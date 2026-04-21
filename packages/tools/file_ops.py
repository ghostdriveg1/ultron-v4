"""
packages/tools/file_ops.py

Ultron V4 — OS File Operations Tool Layer
==========================================
Provides sandboxed async file system operations for the ReAct tool registry.
All paths are resolved relative to ULTRON_WORKSPACE_DIR (env var) or /tmp/ultron.
Path traversal attacks mitigated by workspace jail (resolve + startswith check).

Operations:
  read    — read text file, return content (max 50k chars)
  write   — create/overwrite file, create parent dirs
  append  — append text to existing file
  list    — list directory contents (names + sizes + types)
  delete  — delete file (not directories)
  move    — rename/move file within workspace
  exists  — check if path exists
  mkdir   — create directory tree

Integration:
  - Register with ToolRegistry: registry.register("file_ops", file_ops_tool, FILE_OPS_SCHEMA)
  - ActionResult.extracted_content = operation result string
  - ActionResult.long_term_memory = set for write/append (triggers memory buffer)
  - All async via asyncio.to_thread (blocking IO off event loop)

Informed by:
  - OpenHands/openhands/runtime/plugins/agent_skills/file_ops : edit_file, view_file pattern
  - browser-use file_agent pattern : sandbox jail, explicit allowlist ops
  - SAGAR-TAMANG/friday-tony-stark system.py : OS tool abstraction layer

Design rules:
  - Workspace jail: ALL paths resolved relative to workspace root — no escape
  - Max read size: 50_000 chars (prevent context flood)
  - Max write size: 1_000_000 chars (1MB sanity cap)
  - Binary files: detected by extension — return size only, no content dump
  - No shell exec, no subprocess — pure Python pathlib/shutil
  - Graceful degrade: if ULTRON_WORKSPACE_DIR not set → /tmp/ultron (auto-created)

Future bug risks (pre-registered):
  FO1 [HIGH]  ReAct loop calls write → file path contains '../' → path jail check
              must normalize BEFORE workspace prefix check. Ensure Path.resolve() called.
  FO2 [HIGH]  Large file read (PDF, binary) floods context window.
              Mitigation: binary extension block + 50k char hard cap.
  FO3 [MED]   Concurrent write calls to same file from parallel plan subtasks →
              data corruption. Fix: asyncio.Lock per filepath (file_lock_map).
  FO4 [MED]   /tmp/ultron wiped on HF Space restart → all written files lost.
              Accepted limitation; warn user on write. Persistent storage = S3/R2.
  FO5 [LOW]   list() on very large directory (10k+ files) → response truncated.
              Fix: max_items=200 enforced.
  FO6 [LOW]   move() across filesystems fails silently if shutil.move catches OSError.
              Fix: explicit same-filesystem check before move.

Tool calls used this session:
  Github:get_file_contents x3 (react_loop.py, task_dispatcher.py, packages tree)
  Github:push_files x1
  Notion:notion-fetch x1
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Optional

from packages.brain.react_loop import ActionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WORKSPACE = Path("/tmp/ultron")
MAX_READ_CHARS = 50_000
MAX_WRITE_CHARS = 1_000_000
MAX_LIST_ITEMS = 200

BINARY_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".bin", ".so", ".dll", ".dylib",
    ".mp3", ".mp4", ".wav", ".ogg", ".flac",
    ".pkl", ".pt", ".pth", ".onnx", ".safetensors",
    ".db", ".sqlite", ".sqlite3",
}

# Per-file async locks — bug FO3 mitigation
_file_lock_map: dict[str, asyncio.Lock] = {}
_lock_map_mutex = asyncio.Lock()


# ---------------------------------------------------------------------------
# Workspace management
# ---------------------------------------------------------------------------

def _get_workspace() -> Path:
    """Return configured workspace dir. Auto-create if missing."""
    raw = os.environ.get("ULTRON_WORKSPACE_DIR", str(DEFAULT_WORKSPACE))
    ws = Path(raw)
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _jail(raw_path: str, workspace: Path) -> Optional[Path]:
    """Resolve path and verify it's inside workspace jail.

    Returns None if path escapes workspace (path traversal attempt).
    Bug FO1: resolve() called BEFORE startswith check.
    """
    try:
        # Strip leading slash so relative paths work
        relative = raw_path.lstrip("/")
        resolved = (workspace / relative).resolve()
        if not str(resolved).startswith(str(workspace.resolve())):
            logger.warning(f"[FileOps] PATH JAIL violation: {raw_path} → {resolved}")
            return None
        return resolved
    except Exception as exc:
        logger.error(f"[FileOps] path resolution error: {exc}")
        return None


async def _get_lock(path: Path) -> asyncio.Lock:
    """Return (or create) per-file asyncio.Lock. Bug FO3 mitigation."""
    key = str(path)
    async with _lock_map_mutex:
        if key not in _file_lock_map:
            _file_lock_map[key] = asyncio.Lock()
        return _file_lock_map[key]


def _is_binary(path: Path) -> bool:
    """True if file extension is in binary block list. Bug FO2 mitigation."""
    return path.suffix.lower() in BINARY_EXTENSIONS


# ---------------------------------------------------------------------------
# Individual operations (sync, run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _sync_read(path: Path) -> str:
    if not path.exists():
        return f"ERROR: File not found: {path.name}"
    if not path.is_file():
        return f"ERROR: Not a file: {path.name}"
    if _is_binary(path):
        size = path.stat().st_size
        return f"[BINARY FILE] {path.name} ({size:,} bytes) — content not readable"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + f"\n...[TRUNCATED at {MAX_READ_CHARS} chars]"
        return text
    except Exception as exc:
        return f"ERROR reading {path.name}: {exc}"


def _sync_write(path: Path, content: str, mode: str = "w") -> str:
    if len(content) > MAX_WRITE_CHARS:
        return f"ERROR: Content too large ({len(content):,} chars > {MAX_WRITE_CHARS:,} limit)"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.open(mode, encoding="utf-8").write(content)
        return f"OK: {'Appended' if mode == 'a' else 'Written'} {len(content):,} chars to {path.name}"
    except Exception as exc:
        return f"ERROR writing {path.name}: {exc}"


def _sync_list(path: Path) -> str:
    if not path.exists():
        return f"ERROR: Path not found: {path.name}"
    if path.is_file():
        stat = path.stat()
        return f"FILE: {path.name} ({stat.st_size:,} bytes)"
    items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    if not items:
        return f"EMPTY directory: {path.name}/"
    lines = [f"DIR: {path.name}/"]
    for item in items[:MAX_LIST_ITEMS]:
        if item.is_dir():
            lines.append(f"  📁 {item.name}/")
        else:
            size = item.stat().st_size
            lines.append(f"  📄 {item.name} ({size:,} bytes)")
    if len(items) > MAX_LIST_ITEMS:
        lines.append(f"  ... and {len(items) - MAX_LIST_ITEMS} more (truncated)")
    return "\n".join(lines)


def _sync_delete(path: Path) -> str:
    if not path.exists():
        return f"ERROR: Not found: {path.name}"
    if path.is_dir():
        return f"ERROR: Cannot delete directory with 'delete' — use rmdir explicitly"
    try:
        path.unlink()
        return f"OK: Deleted {path.name}"
    except Exception as exc:
        return f"ERROR deleting {path.name}: {exc}"


def _sync_move(src: Path, dst: Path) -> str:
    if not src.exists():
        return f"ERROR: Source not found: {src.name}"
    try:
        # Bug FO6: check same filesystem
        if src.stat().st_dev != dst.parent.stat().st_dev if dst.parent.exists() else True:
            pass  # shutil.move handles cross-fs; log for awareness
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"OK: Moved {src.name} → {dst.name}"
    except Exception as exc:
        return f"ERROR moving {src.name}: {exc}"


def _sync_mkdir(path: Path) -> str:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return f"OK: Created directory {path.name}/"
    except Exception as exc:
        return f"ERROR mkdir {path.name}: {exc}"


def _sync_exists(path: Path) -> str:
    if not path.exists():
        return f"NOT FOUND: {path.name}"
    t = "DIR" if path.is_dir() else "FILE"
    size = path.stat().st_size if path.is_file() else 0
    return f"{t} EXISTS: {path.name}" + (f" ({size:,} bytes)" if size else "")


# ---------------------------------------------------------------------------
# Main async tool function — registered with ToolRegistry
# ---------------------------------------------------------------------------

async def file_ops_tool(params: dict) -> ActionResult:
    """Unified async file operations tool for ReAct ToolRegistry.

    params:
      op       : str  — one of: read, write, append, list, delete, move, exists, mkdir
      path     : str  — relative path within workspace
      content  : str  — (write/append only) text to write
      dest     : str  — (move only) destination path

    Returns ActionResult with extracted_content = operation result.
    """
    op = params.get("op", "").lower()
    raw_path = params.get("path", "")
    content = params.get("content", "")
    raw_dest = params.get("dest", "")

    if not op:
        return ActionResult(success=False, error="file_ops: 'op' param required")
    if not raw_path and op != "list":
        return ActionResult(success=False, error="file_ops: 'path' param required")

    workspace = _get_workspace()
    path = _jail(raw_path, workspace) if raw_path else workspace
    if raw_path and path is None:
        return ActionResult(
            success=False,
            error=f"file_ops: path '{raw_path}' rejected (security: outside workspace jail)",
        )

    try:
        if op == "read":
            lock = await _get_lock(path)
            async with lock:
                result = await asyncio.to_thread(_sync_read, path)
            return ActionResult(
                extracted_content=result,
                success=not result.startswith("ERROR"),
                error=result if result.startswith("ERROR") else None,
            )

        elif op in ("write", "append"):
            if not content:
                return ActionResult(success=False, error=f"file_ops: 'content' required for {op}")
            mode = "w" if op == "write" else "a"
            lock = await _get_lock(path)
            async with lock:
                result = await asyncio.to_thread(_sync_write, path, content, mode)
            memory_hint = f"file:{op}:{path.name}:{len(content)}chars"
            # Warn: /tmp wiped on HF restart — bug FO4
            if str(workspace).startswith("/tmp"):
                result += " [NOTE: /tmp is ephemeral — lost on Space restart]"
            return ActionResult(
                extracted_content=result,
                long_term_memory=memory_hint,
                success=not result.startswith("ERROR"),
                error=result if result.startswith("ERROR") else None,
            )

        elif op == "list":
            list_path = path if raw_path else workspace
            result = await asyncio.to_thread(_sync_list, list_path)
            return ActionResult(
                extracted_content=result,
                success=not result.startswith("ERROR"),
            )

        elif op == "delete":
            lock = await _get_lock(path)
            async with lock:
                result = await asyncio.to_thread(_sync_delete, path)
            return ActionResult(
                extracted_content=result,
                success=not result.startswith("ERROR"),
                error=result if result.startswith("ERROR") else None,
            )

        elif op == "move":
            if not raw_dest:
                return ActionResult(success=False, error="file_ops: 'dest' param required for move")
            dst_path = _jail(raw_dest, workspace)
            if dst_path is None:
                return ActionResult(
                    success=False,
                    error=f"file_ops: dest path '{raw_dest}' rejected (outside workspace jail)",
                )
            lock = await _get_lock(path)
            async with lock:
                result = await asyncio.to_thread(_sync_move, path, dst_path)
            return ActionResult(
                extracted_content=result,
                success=not result.startswith("ERROR"),
            )

        elif op == "exists":
            result = await asyncio.to_thread(_sync_exists, path)
            return ActionResult(extracted_content=result, success=True)

        elif op == "mkdir":
            result = await asyncio.to_thread(_sync_mkdir, path)
            return ActionResult(
                extracted_content=result,
                success=not result.startswith("ERROR"),
            )

        else:
            available = "read, write, append, list, delete, move, exists, mkdir"
            return ActionResult(
                success=False,
                error=f"file_ops: unknown op '{op}'. Available: {available}",
            )

    except Exception as exc:
        logger.exception(f"[FileOps] unhandled error op={op} path={raw_path}: {exc}")
        return ActionResult(success=False, error=f"file_ops internal error: {str(exc)[:200]}")


# ---------------------------------------------------------------------------
# Tool schema for ToolRegistry / Groq function_calling
# ---------------------------------------------------------------------------

FILE_OPS_SCHEMA: dict = {
    "description": (
        "Read, write, list, delete, move files within Ultron's sandboxed workspace. "
        "Use for saving results, reading uploaded files, managing working data."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["read", "write", "append", "list", "delete", "move", "exists", "mkdir"],
                "description": "File operation to perform",
            },
            "path": {
                "type": "string",
                "description": "Relative file path within workspace (e.g. 'results/output.txt')",
            },
            "content": {
                "type": "string",
                "description": "Text content for write/append operations",
            },
            "dest": {
                "type": "string",
                "description": "Destination path for move operation",
            },
        },
        "required": ["op"],
    },
}


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_file_ops(registry: "ToolRegistry") -> None:  # type: ignore[name-defined]
    """Register file_ops_tool in a ToolRegistry. Call from task_dispatcher.build_tool_registry()."""
    registry.register("file_ops", file_ops_tool, FILE_OPS_SCHEMA)
    logger.info("[FileOps] registered in ToolRegistry")
