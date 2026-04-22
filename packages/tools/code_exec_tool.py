"""
packages/tools/code_exec_tool.py

Ultron V4 — Code Exec ToolRegistry Adapter
===========================================
Bridges packages/tools/code_exec.py (ExecResult) into the ReAct
ToolRegistry interface (ActionResult). Also registers shell execution.

This is the file task_dispatcher.py imports instead of the inline stub.
Single import: from packages.tools.code_exec_tool import code_exec_tool, CODE_EXEC_SCHEMA

Design:
  - code_exec_tool() wraps execute_python() -> ActionResult
  - shell_exec_tool() wraps execute_shell() -> ActionResult
  - long_term_memory: set on success (triggers Zilliz buffer)
  - ActionResult.raw_output: full ExecResult for downstream inspection

Future bug risks (pre-registered):
  CT1 [MED]  If code produces output > 5000 chars, truncation happens in
             code_exec.py. ActionResult.extracted_content will be truncated.
             Caller should not assume output is complete for large scripts.
  CT2 [LOW]  shell_exec_tool exposed to LLM. If LLM generates rm -rf style
             commands they are blocked by denylist in execute_shell().
             Denylist must be kept up to date as attack surface grows.

Tool calls used this session:
  Github:get_file_contents x4
  Github:push_files x1
  Notion:notion-fetch x1
"""

from __future__ import annotations

import logging
from typing import Optional

from packages.brain.react_loop import ActionResult
from packages.tools.code_exec import execute_python, execute_shell

log = logging.getLogger("tools.code_exec_tool")


async def code_exec_tool(params: dict) -> ActionResult:
    """Execute Python code. Wraps execute_python() -> ActionResult."""
    code = params.get("code", "")
    if not code:
        return ActionResult(success=False, error="code_exec: 'code' param required")

    timeout = float(params.get("timeout", 10))
    timeout = min(timeout, 30.0)  # hard cap 30s

    log.info(f"[CodeExecTool] executing {len(code)} chars timeout={timeout}s")
    result = await execute_python(code, timeout=timeout)

    output = result.to_string()
    memory = f"code_exec:success:{output[:200]}" if result.success else None

    return ActionResult(
        success=result.success,
        extracted_content=output,
        long_term_memory=memory,
        error=None if result.success else result.stderr[:300] or result.stdout[:300],
        tool_name="code_exec",
        raw_output=result,
    )


async def shell_exec_tool(params: dict) -> ActionResult:
    """Execute shell command. Wraps execute_shell() -> ActionResult."""
    command = params.get("command", "")
    if not command:
        return ActionResult(success=False, error="shell_exec: 'command' param required")

    timeout = float(params.get("timeout", 10))
    timeout = min(timeout, 30.0)

    log.info(f"[ShellExecTool] command: {command[:80]}")
    result = await execute_shell(command, timeout=timeout)

    output = result.to_string()
    return ActionResult(
        success=result.success,
        extracted_content=output,
        error=None if result.success else result.stderr[:300],
        tool_name="shell_exec",
        raw_output=result,
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

CODE_EXEC_SCHEMA: dict = {
    "description": "Execute Python 3 code and return stdout/stderr. Use for calculations, data transforms, scripting.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Valid Python 3 source code"},
            "timeout": {"type": "number", "description": "Max execution seconds (1-30, default 10)"},
        },
        "required": ["code"],
    },
}

SHELL_EXEC_SCHEMA: dict = {
    "description": "Execute a shell command and return output. Use sparingly. Dangerous commands are blocked.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command string"},
            "timeout": {"type": "number", "description": "Max execution seconds (1-30, default 10)"},
        },
        "required": ["command"],
    },
}


def register_code_exec(registry: "ToolRegistry") -> None:  # type: ignore[name-defined]
    """Register code_exec + shell_exec tools in a ToolRegistry."""
    registry.register("code_exec", code_exec_tool, CODE_EXEC_SCHEMA)
    registry.register("shell_exec", shell_exec_tool, SHELL_EXEC_SCHEMA)
    log.info("[CodeExecTool] code_exec + shell_exec registered in ToolRegistry")
