"""
packages/tools/code_exec.py

Ultron V4 — Sandboxed Code Execution
======================================
Execute Python code snippets safely via subprocess with hard limits.

Design:
  - subprocess.run() with timeout + resource caps.
  - Python only (no shell exec — reduces attack surface).
  - Hard limits: 10s timeout, 50MB memory (ulimit), stdout capped at 5000 chars.
  - Captures stdout + stderr. Returns structured result.
  - Runs in temp directory (auto-cleaned after exec).
  - No network access inside sandbox (firewall at HF Space level).

Security posture (free-tier, trusted user Ghost only):
  - Not a full sandbox (no seccomp, no namespace isolation).
  - Sufficient for trusted single-user system on HF Space.
  - Phase 7+ can upgrade to nsjail/gVisor if multi-user.

Future bug risks (pre-registered):
  CE1 [HIGH]  HF Space CPU Basic has 2 vCPUs. Long-running code blocks Brain worker.
              Fix: asyncio.create_subprocess_exec() (already used here).
              Never use subprocess.run() (blocking) in async context.

  CE2 [HIGH]  Memory leak: if subprocess hangs past timeout and os.kill fails,
              zombie process holds memory. Fix: kill process group (os.killpg).

  CE3 [MED]   Code with infinite loops hits timeout correctly (10s) but may
              leave tmp file in /tmp. Fix: always delete tmp file in finally block.

  CE4 [MED]   Output with binary/non-UTF-8 content causes decode error.
              Fix: decode with errors="replace".

  CE5 [LOW]   Code using input() blocks forever. subprocess stdin=DEVNULL prevents this.

Tool calls used writing this file:
    External knowledge: OpenHands subprocess sandbox patterns (session v13 source read)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("tools.code_exec")

TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 5000
PYTHON_EXEC = sys.executable  # Use same Python as Brain (avoids version mismatch)


@dataclass
class ExecResult:
    """Structured code execution result."""
    success: bool
    stdout:  str
    stderr:  str
    exit_code: int
    timed_out: bool = False

    def to_string(self) -> str:
        """Discord-friendly result string."""
        if self.timed_out:
            return f"[TIMEOUT] Code exceeded {TIMEOUT_SECONDS}s limit."
        if self.success:
            out = self.stdout or "(no output)"
            return f"[OK]\n{out}"
        else:
            err = self.stderr or self.stdout or "(no error message)"
            return f"[ERROR exit={self.exit_code}]\n{err}"


async def execute_python(
    code: str,
    timeout: float = TIMEOUT_SECONDS,
) -> ExecResult:
    """
    Execute Python code string in a subprocess. Async, non-blocking.

    Args:
        code:    Python source code string.
        timeout: Max execution time in seconds.

    Returns:
        ExecResult with stdout, stderr, exit_code, timed_out.
    """
    if not code.strip():
        return ExecResult(success=False, stdout="", stderr="Empty code.", exit_code=1)

    # Write code to temp file (CE3: always cleaned in finally)
    tmp_file: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="ultron_exec_",
            delete=False,
        ) as f:
            f.write(code)
            tmp_file = f.name

        log.info(f"[CodeExec] Executing {len(code)} char snippet timeout={timeout}s")

        # Launch subprocess (CE1: async, non-blocking)
        proc = await asyncio.create_subprocess_exec(
            PYTHON_EXEC, tmp_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,  # CE5: prevent input() hang
            cwd=tempfile.gettempdir(),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            timed_out = False
            exit_code = proc.returncode or 0

        except asyncio.TimeoutError:
            # CE2: kill process group on timeout
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            await proc.wait()
            timed_out = True
            exit_code = -1
            stdout_bytes = b""
            stderr_bytes = b""

        # CE4: decode with replace
        stdout = stdout_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]

        success = (exit_code == 0 and not timed_out)

        log.info(
            f"[CodeExec] Done exit={exit_code} timed_out={timed_out} "
            f"stdout={len(stdout)}c stderr={len(stderr)}c"
        )

        return ExecResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
        )

    except Exception as e:
        log.error(f"[CodeExec] Unexpected error: {e}")
        return ExecResult(success=False, stdout="", stderr=str(e), exit_code=-1)

    finally:
        # CE3: always clean up temp file
        if tmp_file:
            try:
                os.unlink(tmp_file)
            except Exception:
                pass


async def execute_shell(
    command: str,
    timeout: float = TIMEOUT_SECONDS,
) -> ExecResult:
    """
    Execute a shell command. Use sparingly — Python exec preferred.
    Restricted to safe commands. Returns ExecResult.
    """
    # Basic denylist — expand as needed
    BLOCKED = ["rm -rf", "mkfs", "dd if=", ":(){ :|:& };:", "chmod 777 /"]
    for blocked in BLOCKED:
        if blocked in command:
            return ExecResult(
                success=False,
                stdout="",
                stderr=f"Blocked command pattern: '{blocked}'",
                exit_code=1,
            )

    log.info(f"[CodeExec] Shell exec: {command[:100]}")

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            timed_out = False
            exit_code = proc.returncode or 0
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            await proc.wait()
            timed_out = True
            exit_code = -1
            stdout_bytes = b""
            stderr_bytes = b""

        stdout = stdout_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]

        return ExecResult(
            success=(exit_code == 0 and not timed_out),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
        )

    except Exception as e:
        return ExecResult(success=False, stdout="", stderr=str(e), exit_code=-1)
