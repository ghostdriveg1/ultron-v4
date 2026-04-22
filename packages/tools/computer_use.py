"""
packages/tools/computer_use.py

Ultron V4 — OS Computer Use Tool Layer
=======================================
Provides screenshot capture + mouse/keyboard automation for OS-level control.
This is the Jarvis-level "eyes and hands" on the host machine.

Capabilities:
  screenshot   — capture screen as base64 PNG (or save to workspace file)
  click        — left/right/double click at (x, y) or on text match via OCR
  type         — type text string (keyboard injection)
  hotkey       — send key combination (e.g. ctrl+c, alt+tab, win+d)
  scroll       — scroll wheel at position
  move         — move mouse to coordinates
  get_cursor   — return current cursor (x, y) position
  find_text    — OCR screen, return bounding box of text match (tesseract)
  drag         — click-hold drag from (x1,y1) to (x2,y2)
  clipboard    — get or set clipboard text

Design rules:
  - HF Spaces is headless (no physical display). DISPLAY env var must be set.
    Use Xvfb virtual display (auto-started if not already running).
  - All coordinates are absolute screen pixels.
  - pyautogui is the underlying driver (widely available, pip installable).
  - OCR via pytesseract (tesseract-ocr system package required).
  - Screenshot returns base64 PNG string for LLM vision pass OR saves to
    workspace file for file_ops retrieval.
  - Graceful degrade: if pyautogui not installed -> return error ActionResult.
  - Single asyncio.Lock for all mouse/keyboard ops (serialise automation).
  - All actions run via asyncio.to_thread (blocking IO off event loop).

Informed by:
  - anthropics/computer-use-demo  : screenshot loop, key combo mapping
  - browser-use                   : DOM-first (try DOM before screenshot)
  - OpenHands runtime             : Xvfb setup, display env management

Future bug risks (pre-registered):
  CU1 [HIGH]  HF Space headless — no DISPLAY. Xvfb must be running before any
              pyautogui call. Fix: _ensure_display() called at module init.
              If Xvfb launch fails (no xvfb-run binary), fall back gracefully.
  CU2 [HIGH]  pyautogui.screenshot() on headless without Xvfb raises exception.
              Fix: wrap all calls in try/except, return error ActionResult.
  CU3 [MED]   OCR (tesseract) not installed on HF Space by default.
              Fix: if ImportError -> skip find_text, return unsupported message.
              Add to requirements: pytesseract. System: apt-get tesseract-ocr.
  CU4 [MED]   Long screenshot base64 (1920x1080 PNG ~200KB) floods LLM context.
              Fix: downscale to 1024x768 max before base64 encode.
              Optionally save to file_ops workspace instead of returning inline.
  CU5 [LOW]   Concurrent computer_use calls from parallel plan subtasks cause
              mouse position race conditions. Fix: module-level asyncio.Lock
              serialises all mouse/keyboard ops.
  CU6 [LOW]   type() with unicode chars may fail on some X11 configs.
              Fix: use xdotool as fallback for unicode injection.

Tool calls used this session:
  Github:get_file_contents x4 (main.py, task_dispatcher.py, code_exec.py, browser_agent.py)
  Github:push_files x1
  Notion:notion-fetch x1
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import subprocess
from typing import Optional

from packages.brain.react_loop import ActionResult

log = logging.getLogger("tools.computer_use")

# Module-level lock — serialise all mouse/keyboard ops (CU5)
_automation_lock = asyncio.Lock()

# Max screenshot dimensions to avoid context flood (CU4)
MAX_SCREENSHOT_W = 1024
MAX_SCREENSHOT_H = 768


# ---------------------------------------------------------------------------
# Display management (Xvfb for headless)
# ---------------------------------------------------------------------------

_display_ensured = False


def _ensure_display() -> None:
    """Ensure a virtual display is available for headless environments.

    Sets DISPLAY env var if not already set. Attempts to start Xvfb.
    Bug CU1 mitigation.
    """
    global _display_ensured
    if _display_ensured:
        return

    if os.environ.get("DISPLAY"):
        log.info(f"[ComputerUse] DISPLAY already set: {os.environ['DISPLAY']}")
        _display_ensured = True
        return

    # Try to start Xvfb on :99
    try:
        result = subprocess.run(
            ["xvfb-run", "--help"],
            capture_output=True, timeout=3
        )
        if result.returncode == 0:
            # xvfb-run available — set display env, actual Xvfb started by OS
            os.environ["DISPLAY"] = ":99"
            # Launch Xvfb in background
            subprocess.Popen(
                ["Xvfb", ":99", "-screen", "0", "1280x800x24"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import time; time.sleep(1)  # wait for Xvfb to start
            os.environ["DISPLAY"] = ":99"
            log.info("[ComputerUse] Xvfb started on :99")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        log.warning(
            "[ComputerUse] xvfb-run not found. CU1: headless screenshot will fail. "
            "Add 'Xvfb' to HF Space apt packages."
        )

    _display_ensured = True


# ---------------------------------------------------------------------------
# pyautogui availability check
# ---------------------------------------------------------------------------

def _get_pyautogui():
    """Import pyautogui, return None if not available."""
    try:
        import pyautogui  # type: ignore
        pyautogui.FAILSAFE = False  # disable corner-fail in headless
        return pyautogui
    except ImportError:
        log.warning("[ComputerUse] pyautogui not installed. Add to requirements.")
        return None


# ---------------------------------------------------------------------------
# Individual sync operations (run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _sync_screenshot(save_path: Optional[str] = None) -> str:
    """Capture screenshot. Returns base64 PNG string or file path."""
    _ensure_display()
    pag = _get_pyautogui()
    if pag is None:
        return "ERROR: pyautogui not installed"

    try:
        img = pag.screenshot()

        # Downscale if too large (CU4)
        if img.width > MAX_SCREENSHOT_W or img.height > MAX_SCREENSHOT_H:
            img = img.resize((MAX_SCREENSHOT_W, MAX_SCREENSHOT_H))

        if save_path:
            img.save(save_path, format="PNG")
            return f"Screenshot saved: {save_path}"
        else:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{b64}"

    except Exception as e:
        return f"ERROR: screenshot failed: {e}"


def _sync_click(x: int, y: int, button: str = "left", clicks: int = 1) -> str:
    pag = _get_pyautogui()
    if pag is None:
        return "ERROR: pyautogui not installed"
    try:
        pag.click(x, y, button=button, clicks=clicks, interval=0.1)
        return f"Clicked ({x},{y}) button={button} clicks={clicks}"
    except Exception as e:
        return f"ERROR: click failed: {e}"


def _sync_type(text: str, interval: float = 0.02) -> str:
    pag = _get_pyautogui()
    if pag is None:
        return "ERROR: pyautogui not installed"
    try:
        pag.typewrite(text, interval=interval)
        return f"Typed {len(text)} chars"
    except Exception as e:
        # CU6 fallback: try pyperclip paste for unicode
        try:
            import pyperclip  # type: ignore
            pyperclip.copy(text)
            pag.hotkey("ctrl", "v")
            return f"Typed (via clipboard paste) {len(text)} chars"
        except Exception:
            return f"ERROR: type failed: {e}"


def _sync_hotkey(*keys: str) -> str:
    pag = _get_pyautogui()
    if pag is None:
        return "ERROR: pyautogui not installed"
    try:
        pag.hotkey(*keys)
        return f"Hotkey: {'+'.join(keys)}"
    except Exception as e:
        return f"ERROR: hotkey failed: {e}"


def _sync_scroll(x: int, y: int, clicks: int) -> str:
    pag = _get_pyautogui()
    if pag is None:
        return "ERROR: pyautogui not installed"
    try:
        pag.scroll(clicks, x=x, y=y)
        direction = "up" if clicks > 0 else "down"
        return f"Scrolled {direction} {abs(clicks)} clicks at ({x},{y})"
    except Exception as e:
        return f"ERROR: scroll failed: {e}"


def _sync_move(x: int, y: int) -> str:
    pag = _get_pyautogui()
    if pag is None:
        return "ERROR: pyautogui not installed"
    try:
        pag.moveTo(x, y, duration=0.2)
        return f"Mouse moved to ({x},{y})"
    except Exception as e:
        return f"ERROR: move failed: {e}"


def _sync_get_cursor() -> str:
    pag = _get_pyautogui()
    if pag is None:
        return "ERROR: pyautogui not installed"
    try:
        x, y = pag.position()
        return f"Cursor at ({x},{y})"
    except Exception as e:
        return f"ERROR: get_cursor failed: {e}"


def _sync_drag(x1: int, y1: int, x2: int, y2: int, duration: float = 0.5) -> str:
    pag = _get_pyautogui()
    if pag is None:
        return "ERROR: pyautogui not installed"
    try:
        pag.moveTo(x1, y1)
        pag.dragTo(x2, y2, duration=duration, button="left")
        return f"Dragged ({x1},{y1}) -> ({x2},{y2})"
    except Exception as e:
        return f"ERROR: drag failed: {e}"


def _sync_find_text(query: str) -> str:
    """OCR screen, find text, return bounding box. CU3: requires tesseract."""
    _ensure_display()
    pag = _get_pyautogui()
    if pag is None:
        return "ERROR: pyautogui not installed"
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return "ERROR: pytesseract not installed (CU3). Add pytesseract to requirements + tesseract-ocr to apt packages."

    try:
        img = pag.screenshot()
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        matches = []
        for i, word in enumerate(data["text"]):
            if query.lower() in word.lower():
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                cx = x + w // 2
                cy = y + h // 2
                matches.append(f"'{word}' at ({cx},{cy}) bbox=({x},{y},{w},{h})")
        if not matches:
            return f"Text '{query}' not found on screen"
        return "Found: " + " | ".join(matches[:5])
    except Exception as e:
        return f"ERROR: find_text failed: {e}"


def _sync_clipboard(action: str, text: str = "") -> str:
    """Get or set clipboard content."""
    try:
        import pyperclip  # type: ignore
    except ImportError:
        return "ERROR: pyperclip not installed. Add to requirements."
    try:
        if action == "get":
            content = pyperclip.paste()
            return f"Clipboard: {content[:500]}"
        elif action == "set":
            pyperclip.copy(text)
            return f"Clipboard set: {text[:100]}"
        else:
            return f"ERROR: clipboard action must be 'get' or 'set'"
    except Exception as e:
        return f"ERROR: clipboard failed: {e}"


# ---------------------------------------------------------------------------
# Main async tool function — registered with ToolRegistry
# ---------------------------------------------------------------------------

async def computer_use_tool(params: dict) -> ActionResult:
    """Unified async OS automation tool for ReAct ToolRegistry.

    params:
      op       : str  — screenshot|click|type|hotkey|scroll|move|get_cursor|
                        find_text|drag|clipboard
      x, y     : int  — screen coordinates (click/scroll/move/drag)
      x2, y2   : int  — drag destination
      button   : str  — 'left'|'right'|'middle' (click)
      clicks   : int  — click count or scroll clicks (+ = up, - = down)
      text     : str  — text to type or clipboard set content
      keys     : list — hotkey sequence e.g. ['ctrl', 'c']
      query    : str  — OCR search text (find_text)
      save_path: str  — optional workspace-relative path to save screenshot
      action   : str  — clipboard sub-action: 'get'|'set'

    Returns ActionResult with extracted_content = operation result.
    """
    op = params.get("op", "").lower()
    if not op:
        return ActionResult(success=False, error="computer_use: 'op' param required")

    async with _automation_lock:  # CU5: serialise
        try:
            if op == "screenshot":
                save_path = params.get("save_path") or None
                result = await asyncio.to_thread(_sync_screenshot, save_path)
                # Don't embed huge base64 in long_term_memory
                is_b64 = result.startswith("data:image")
                return ActionResult(
                    extracted_content=result if not is_b64 else f"[SCREENSHOT captured, {len(result)} chars b64]",
                    raw_output=result if is_b64 else None,
                    success=not result.startswith("ERROR"),
                    error=result if result.startswith("ERROR") else None,
                )

            elif op == "click":
                x = int(params.get("x", 0))
                y = int(params.get("y", 0))
                button = params.get("button", "left")
                clicks = int(params.get("clicks", 1))
                result = await asyncio.to_thread(_sync_click, x, y, button, clicks)
                return ActionResult(
                    extracted_content=result,
                    success=not result.startswith("ERROR"),
                    error=result if result.startswith("ERROR") else None,
                )

            elif op == "type":
                text = params.get("text", "")
                if not text:
                    return ActionResult(success=False, error="computer_use: 'text' required for type")
                result = await asyncio.to_thread(_sync_type, text)
                return ActionResult(
                    extracted_content=result,
                    success=not result.startswith("ERROR"),
                    error=result if result.startswith("ERROR") else None,
                )

            elif op == "hotkey":
                keys = params.get("keys", [])
                if not keys:
                    return ActionResult(success=False, error="computer_use: 'keys' list required for hotkey")
                result = await asyncio.to_thread(_sync_hotkey, *keys)
                return ActionResult(
                    extracted_content=result,
                    success=not result.startswith("ERROR"),
                    error=result if result.startswith("ERROR") else None,
                )

            elif op == "scroll":
                x = int(params.get("x", 0))
                y = int(params.get("y", 0))
                clicks = int(params.get("clicks", -3))
                result = await asyncio.to_thread(_sync_scroll, x, y, clicks)
                return ActionResult(
                    extracted_content=result,
                    success=not result.startswith("ERROR"),
                    error=result if result.startswith("ERROR") else None,
                )

            elif op == "move":
                x = int(params.get("x", 0))
                y = int(params.get("y", 0))
                result = await asyncio.to_thread(_sync_move, x, y)
                return ActionResult(
                    extracted_content=result,
                    success=not result.startswith("ERROR"),
                    error=result if result.startswith("ERROR") else None,
                )

            elif op == "get_cursor":
                result = await asyncio.to_thread(_sync_get_cursor)
                return ActionResult(extracted_content=result, success=True)

            elif op == "drag":
                x1 = int(params.get("x", 0))
                y1 = int(params.get("y", 0))
                x2 = int(params.get("x2", 0))
                y2 = int(params.get("y2", 0))
                duration = float(params.get("duration", 0.5))
                result = await asyncio.to_thread(_sync_drag, x1, y1, x2, y2, duration)
                return ActionResult(
                    extracted_content=result,
                    success=not result.startswith("ERROR"),
                    error=result if result.startswith("ERROR") else None,
                )

            elif op == "find_text":
                query = params.get("query", "")
                if not query:
                    return ActionResult(success=False, error="computer_use: 'query' required for find_text")
                result = await asyncio.to_thread(_sync_find_text, query)
                return ActionResult(
                    extracted_content=result,
                    success="ERROR" not in result,
                    error=result if "ERROR" in result else None,
                )

            elif op == "clipboard":
                action = params.get("action", "get")
                text = params.get("text", "")
                result = await asyncio.to_thread(_sync_clipboard, action, text)
                return ActionResult(
                    extracted_content=result,
                    success=not result.startswith("ERROR"),
                    error=result if result.startswith("ERROR") else None,
                )

            else:
                ops = "screenshot,click,type,hotkey,scroll,move,get_cursor,find_text,drag,clipboard"
                return ActionResult(
                    success=False,
                    error=f"computer_use: unknown op '{op}'. Available: {ops}",
                )

        except Exception as e:
            log.exception(f"[ComputerUse] unhandled error op={op}: {e}")
            return ActionResult(success=False, error=f"computer_use internal error: {str(e)[:200]}")


# ---------------------------------------------------------------------------
# Tool schema for ToolRegistry / Groq function_calling
# ---------------------------------------------------------------------------

COMPUTER_USE_SCHEMA: dict = {
    "description": (
        "Control the OS: take screenshots, click, type text, press hotkeys, "
        "scroll, drag, find text on screen via OCR. Use for GUI automation tasks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": [
                    "screenshot", "click", "type", "hotkey", "scroll",
                    "move", "get_cursor", "find_text", "drag", "clipboard",
                ],
                "description": "Operation to perform",
            },
            "x": {"type": "integer", "description": "X coordinate (pixels)"},
            "y": {"type": "integer", "description": "Y coordinate (pixels)"},
            "x2": {"type": "integer", "description": "Drag destination X"},
            "y2": {"type": "integer", "description": "Drag destination Y"},
            "button": {"type": "string", "enum": ["left", "right", "middle"]},
            "clicks": {"type": "integer", "description": "Click count or scroll clicks (+up/-down)"},
            "text": {"type": "string", "description": "Text to type or clipboard content"},
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Hotkey sequence e.g. ['ctrl','c'] or ['alt','tab']",
            },
            "query": {"type": "string", "description": "Text to find on screen via OCR"},
            "save_path": {"type": "string", "description": "Workspace-relative path to save screenshot PNG"},
            "action": {"type": "string", "enum": ["get", "set"], "description": "Clipboard sub-action"},
        },
        "required": ["op"],
    },
}


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_computer_use(registry: "ToolRegistry") -> None:  # type: ignore[name-defined]
    """Register computer_use_tool in a ToolRegistry."""
    registry.register("computer_use", computer_use_tool, COMPUTER_USE_SCHEMA)
    log.info("[ComputerUse] registered in ToolRegistry")
