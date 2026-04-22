"""
packages/brain/task_dispatcher.py

Ultron V4 — Task Orchestration Layer
=====================================
Replaces V3's single-shot LLM dispatch with a full ReAct-loop-backed orchestrator.

Responsibilities:
  1. Classify incoming task (search / code / browser / file / computer_use / general)
  2. Hydrate ToolRegistry with ALL concrete tools (search, code_exec, shell_exec,
     browser_fetch, file_ops, file_read, computer_use)
  3. Pull channel AgentState from Redis (persist back after loop finishes)
  4. Call multi-provider key_rotation pool for LLM access (ALL 5 providers)
  5. Run ReActLoop (flash_mode=True for Groq)
  6. Strip internal memory/graph blocks before returning to Discord
  7. Write long_term_memory snippets to Tier2 (Redis buffer → Zilliz flush later)

Informed by:
  - react_loop.py (this repo)                  : ReActLoop, ToolRegistry, AgentState
  - llm_router.py (this repo)                  : make_provider_llm_fn
  - packages/tools/search.py                   : tavily_search (real + DDG fallback)
  - packages/tools/code_exec_tool.py           : code_exec_tool, shell_exec_tool (REAL)
  - packages/tools/file_ops.py                 : file_ops_tool (sandboxed OS file ops)
  - packages/tools/computer_use.py             : computer_use_tool (OS GUI automation)
  - packages/tools/browser_agent.py            : BrowserAgent (Playwright)
  - browser-use/browser-use                    : dispatch pattern, no-vision Groq rule
  - OpenHands/codeact_agent                    : pending_actions, function_calling

Design rules:
  - flash_mode=True default (Groq 8k ctx safe)
  - Per-channel AgentState in Redis (key: ultron:state:{channel_id})
  - LLM calls via make_provider_llm_fn(pool) from llm_router — ALL 5 providers
  - Response to Discord: NEVER leak === MEMORY GRAPH === or [COMPACTED HISTORY] blocks
  - Memory snippets: append to Redis list ultron:mem_buffer:{user_id} (Zilliz flush)
  - max_iterations=5 default (hard ceiling from react_loop ABSOLUTE_MAX=10)
  - search tool: REAL Tavily + DDG fallback
  - code_exec: REAL subprocess execution via code_exec_tool.py
  - file_ops: sandboxed workspace file ops via file_ops.py
  - computer_use: OS GUI automation via computer_use.py (Xvfb required on HF)

Future bug risks (pre-registered):
  D1 [HIGH]   Redis unavailable → AgentState fresh each msg → loop detector clears (B1)
  D2 [HIGH]   Pool None/exhausted → llm_call_fn 401 → max failures → no user message
  D3 [MED]    Keyword classifier ambiguity → wrong first tool → wasted iteration
  D4 [MED]    Redis state TTL not set → accumulates → OOM. Fix: TTL=3600 always.
  D5 [LOW]    mem_buffer unbounded → LTRIM to 100 after every RPUSH
  D6 [LOW]    strip_internal_blocks() regex → add start-of-line anchor
  D7 [MED]    computer_use tool in tool registry increases Groq prompt size by ~200
              tokens (schema). Watch ctx window usage on Groq 8k models.
  D8 [LOW]    BrowserAgent imported lazily inside _tool_browser_agent() — if Playwright
              not installed, error surfaces only at runtime (not startup).
              Fix: check playwright import in build_tool_registry() and skip if missing.

Tool calls used this session:
  Github:get_file_contents x4 (react_loop, task_dispatcher, code_exec, browser_agent)
  Github:push_files x1
  Notion:notion-fetch x1
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Optional

import httpx

from packages.brain.react_loop import (
    ActionResult,
    AgentState,
    LoopStatus,
    ReActLoop,
    ToolRegistry,
)
from packages.brain.llm_router import make_provider_llm_fn
from packages.tools.search import tavily_search
from packages.tools.code_exec_tool import (
    code_exec_tool,
    shell_exec_tool,
    CODE_EXEC_SCHEMA,
    SHELL_EXEC_SCHEMA,
)
from packages.tools.file_ops import file_ops_tool, FILE_OPS_SCHEMA, register_file_ops
from packages.tools.computer_use import computer_use_tool, COMPUTER_USE_SCHEMA, register_computer_use

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDIS_STATE_PREFIX   = "ultron:state:"
REDIS_MEM_BUF_PREFIX = "ultron:mem_buffer:"
REDIS_STATE_TTL      = 3600      # always set TTL (D4)
MEM_BUF_MAX          = 100       # max snippets before trim (D5)
DEFAULT_MAX_ITERATIONS = 5
GROQ_FLASH_MODE      = True      # lock True; never disable for Groq

TASK_TYPE_KEYWORDS: dict[str, list[str]] = {
    "search": [
        "search", "find", "who is", "what is", "when", "latest", "news",
        "price", "weather", "stock", "current", "today", "look up",
    ],
    "code_exec": [
        "run", "execute", "calculate", "compute", "script",
        "python", "bash", "code", "program", "shell",
    ],
    "browser_fetch": [
        "open", "browse", "visit", "url", "website", "scrape",
        "read the page", "fetch url", "go to", "https://", "http://",
    ],
    "file_ops": [
        "read file", "write file", "save file", "open file", "load file",
        "list files", "delete file", "create file", "show file", "my file",
        "from the file", "uploaded", "file path",
    ],
    "computer_use": [
        "screenshot", "click on", "type into", "press", "hotkey",
        "screen", "mouse", "keyboard", "desktop", "window", "gui",
        "find on screen", "drag", "scroll the screen",
    ],
}


# ---------------------------------------------------------------------------
# Internal blocks that must NEVER reach Discord
# ---------------------------------------------------------------------------

_STRIP_PATTERNS = [
    re.compile(r"^=== MEMORY GRAPH ===[\s\S]*?(?=^===|\Z)", re.MULTILINE),
    re.compile(r"^\[COMPACTED HISTORY SUMMARY\][\s\S]*?(?=^\[|\Z)", re.MULTILINE),
    re.compile(r"^\[OBSERVATION\][\s\S]*?(?=^\[|\Z)", re.MULTILINE),
    re.compile(r"^\[LOOP WARNING\].*$", re.MULTILINE),
    re.compile(r"^\[TOOL (ERROR|RESULT|OK)\].*$", re.MULTILINE),
]


def strip_internal_blocks(text: str) -> str:
    for pat in _STRIP_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Browser fetch tool (inline — no Playwright dependency)
# ---------------------------------------------------------------------------

async def _tool_browser_fetch(params: dict) -> ActionResult:
    """Lightweight URL text fetch (no JS rendering, no screenshots)."""
    url = params.get("url", "")
    if not url or not url.startswith(("http://", "https://")):
        return ActionResult(success=False, error="browser_fetch: invalid or missing url")
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "UltronBot/1.0"})
            resp.raise_for_status()
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()[:3000]
            return ActionResult(
                extracted_content=text,
                long_term_memory=f"fetched:{url}",
            )
    except Exception as exc:
        logger.error(f"[browser_fetch] {url}: {exc}")
        return ActionResult(success=False, error=str(exc)[:300])


async def _tool_browser_agent(params: dict) -> ActionResult:
    """Full Playwright BrowserAgent for JS-rendered pages. Lazy import."""
    task = params.get("task", params.get("url", ""))
    if not task:
        return ActionResult(success=False, error="browser_agent: 'task' param required")
    try:
        from packages.tools.browser_agent import BrowserAgent  # D8: lazy import
        agent = BrowserAgent(llm_fn=params.get("_llm_fn"))  # injected by dispatcher
        result_str = await agent.run(task)
        return ActionResult(
            extracted_content=result_str,
            long_term_memory=f"browser_agent:{task[:80]}",
        )
    except ImportError:
        return ActionResult(
            success=False,
            error="browser_agent: playwright not installed (D8). pip install playwright && playwright install chromium",
        )
    except Exception as exc:
        return ActionResult(success=False, error=str(exc)[:300])


# ---------------------------------------------------------------------------
# Tool schemas for search + browser tools
# ---------------------------------------------------------------------------

SEARCH_SCHEMA: dict = {
    "description": "Search the web for current information. Use for news, facts, prices, people.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query, max 100 chars"},
            "max_results": {"type": "integer", "description": "Results count (1-10)", "default": 5},
        },
        "required": ["query"],
    },
}

BROWSER_FETCH_SCHEMA: dict = {
    "description": "Fetch plain text from a URL. For reading articles, documentation pages.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL starting http:// or https://"},
        },
        "required": ["url"],
    },
}

BROWSER_AGENT_SCHEMA: dict = {
    "description": "Full browser automation for JS-heavy pages. Slower — prefer browser_fetch for simple pages.",
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Browser task description or URL to interact with"},
        },
        "required": ["task"],
    },
}


# ---------------------------------------------------------------------------
# ToolRegistry factory — ALL tools registered here
# ---------------------------------------------------------------------------

def build_tool_registry(llm_fn: Any = None) -> ToolRegistry:
    """Build ToolRegistry with all V4 tools.

    Tools:
      search         — Tavily + DDG fallback (REAL)
      code_exec      — subprocess Python execution (REAL)
      shell_exec     — subprocess shell execution (REAL)
      browser_fetch  — httpx URL text fetch (REAL)
      browser_agent  — Playwright full automation (REAL, lazy import)
      file_ops       — sandboxed workspace file operations (REAL)
      computer_use   — OS screenshot + mouse/keyboard automation (REAL)
    """
    registry = ToolRegistry()

    # Search
    registry.register("search", tavily_search, SEARCH_SCHEMA)

    # Code execution (REAL — replaces stub)
    registry.register("code_exec", code_exec_tool, CODE_EXEC_SCHEMA)
    registry.register("shell_exec", shell_exec_tool, SHELL_EXEC_SCHEMA)

    # Browser
    registry.register("browser_fetch", _tool_browser_fetch, BROWSER_FETCH_SCHEMA)
    registry.register("browser_agent", _tool_browser_agent, BROWSER_AGENT_SCHEMA)

    # File ops (REAL — v28)
    register_file_ops(registry)

    # Computer use (REAL — v29)
    register_computer_use(registry)

    logger.info(
        f"[ToolRegistry] registered {len(registry.tool_names)} tools: "
        f"{', '.join(registry.tool_names)}"
    )
    return registry


# ---------------------------------------------------------------------------
# Task-type pre-classifier
# ---------------------------------------------------------------------------

def classify_task(message: str) -> str:
    msg_lower = message.lower()
    scores: dict[str, int] = {t: 0 for t in TASK_TYPE_KEYWORDS}
    for task_type, keywords in TASK_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in msg_lower:
                scores[task_type] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"


# ---------------------------------------------------------------------------
# Redis AgentState persistence
# ---------------------------------------------------------------------------

async def _load_state(redis_client: Any, channel_id: str) -> Optional[dict]:
    if redis_client is None:
        return None
    try:
        key = f"{REDIS_STATE_PREFIX}{channel_id}"
        raw = await redis_client.get(key)
        if raw:
            return json.loads(raw)
    except Exception as exc:
        logger.warning(f"[TaskDispatcher] Redis state load failed: {exc}")
    return None


async def _save_state(redis_client: Any, channel_id: str, state: AgentState) -> None:
    if redis_client is None:
        return
    try:
        key = f"{REDIS_STATE_PREFIX}{channel_id}"
        state_dict = {
            "task":                state.task,
            "n_steps":             state.n_steps,
            "consecutive_failures": state.consecutive_failures,
            "running_memory":      state.running_memory,
            "status":              state.status.value,
            "started_at":          state.started_at,
        }
        await redis_client.set(key, json.dumps(state_dict), ex=REDIS_STATE_TTL)
    except Exception as exc:
        logger.warning(f"[TaskDispatcher] Redis state save failed: {exc}")


async def _buffer_memory(redis_client: Any, user_id: str, snippet: str) -> None:
    if redis_client is None or not snippet:
        return
    try:
        key = f"{REDIS_MEM_BUF_PREFIX}{user_id}"
        await redis_client.rpush(key, snippet)
        await redis_client.ltrim(key, -MEM_BUF_MAX, -1)  # D5
    except Exception as exc:
        logger.warning(f"[TaskDispatcher] mem_buffer write failed: {exc}")


# ---------------------------------------------------------------------------
# TaskDispatcher
# ---------------------------------------------------------------------------

class TaskDispatcher:
    """Orchestrates task execution for Ultron V4."""

    def __init__(
        self,
        pool: Any = None,
        redis: Any = None,
        settings: Any = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        flash_mode: bool = GROQ_FLASH_MODE,
    ) -> None:
        self.pool          = pool
        self.redis         = redis
        self.settings      = settings
        self.max_iterations = max_iterations
        self.flash_mode    = flash_mode

    async def dispatch(
        self,
        message: str,
        channel_id: str,
        user_id: str,
        username: str = "user",
        context: str = "",
    ) -> str:
        """Main entry point. Returns Discord-safe response string."""
        task_type = classify_task(message)
        logger.info(
            f"[TaskDispatcher] channel={channel_id} user={user_id} "
            f"task_type={task_type} msg='{message[:60]}'"
        )

        # Load persisted context
        persisted = await _load_state(self.redis, channel_id)
        initial_context = context
        if persisted and persisted.get("running_memory"):
            initial_context = (
                f"[CONTEXT FROM MEMORY]\n{persisted['running_memory']}\n\n"
                + initial_context
            )

        # Build tools + LLM
        llm_call_fn = await make_provider_llm_fn(self.pool)
        registry    = build_tool_registry(llm_fn=llm_call_fn)

        loop = ReActLoop(
            llm_call_fn=llm_call_fn,
            tool_registry=registry,
            flash_mode=self.flash_mode,
            max_iterations=self.max_iterations,
        )

        final_result: Optional[ActionResult] = None
        try:
            final_result = await loop.run(
                task=message,
                initial_context=initial_context,
            )
        except Exception as exc:
            logger.exception(f"[TaskDispatcher] ReActLoop raised: {exc}")
            final_result = ActionResult(
                is_done=True,
                success=False,
                error=f"Internal error: {str(exc)[:200]}",
            )
        finally:
            _state = AgentState(
                task=message,
                running_memory=(
                    final_result.extracted_content[:500]
                    if final_result and final_result.extracted_content
                    else ""
                ),
            )
            await _save_state(self.redis, channel_id, _state)
            if final_result and final_result.long_term_memory:
                await _buffer_memory(self.redis, user_id, final_result.long_term_memory)

        if final_result is None or (not final_result.success and final_result.error):
            response = "Sorry, I ran into an issue completing that task." + (
                f" ({final_result.error[:100]})" if final_result else ""
            )
        elif final_result.extracted_content:
            response = final_result.extracted_content
        else:
            response = "Task completed, but no output was produced."

        response = strip_internal_blocks(response)
        if len(response) > 1800:
            response = response[:1797] + "..."
        return response


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_dispatcher_instance: Optional[TaskDispatcher] = None


def get_dispatcher(
    pool: Any = None,
    redis: Any = None,
    settings: Any = None,
) -> TaskDispatcher:
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = TaskDispatcher(pool=pool, redis=redis, settings=settings)
        logger.info("[TaskDispatcher] Singleton created")
    return _dispatcher_instance
