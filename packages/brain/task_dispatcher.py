"""
packages/brain/task_dispatcher.py

Ultron V4 — Task Orchestration Layer
=====================================
v31 update: MetacognitionEngine wired into dispatch().
  pre_action_assessment()  called BEFORE ReActLoop
  post_action_reflection() called AFTER  ReActLoop

Design rules unchanged:
  - flash_mode=True default (Groq 8k ctx safe)
  - Per-channel AgentState in Redis (key: ultron:state:{channel_id})
  - LLM calls via make_provider_llm_fn(pool) — ALL 8 providers now
  - Response to Discord: NEVER leak internal blocks
  - max_iterations=5 default

Future bug risks (pre-registered, v31 additions):
  D1 [HIGH]   Redis unavailable → AgentState fresh each msg
  D2 [HIGH]   Pool None/exhausted → llm_call_fn fails
  D3 [MED]    Keyword classifier ambiguity
  D4 [MED]    Redis state TTL not set → OOM
  D5 [LOW]    mem_buffer unbounded
  D6 [LOW]    strip_internal_blocks() regex anchor
  D7 [MED]    computer_use schema inflates Groq prompt ~200 tokens
  D8 [LOW]    BrowserAgent lazy import

Tool calls used this session (v31):
  Github:get_file_contents x3
  Github:push_files x1
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
from packages.brain.meta.engine import get_metacognition
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
REDIS_STATE_TTL      = 3600
MEM_BUF_MAX          = 100
DEFAULT_MAX_ITERATIONS = 5
GROQ_FLASH_MODE      = True

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

_STRIP_PATTERNS = [
    re.compile(r"^=== MEMORY GRAPH ===.*?(?=^===|\Z)", re.MULTILINE | re.DOTALL),
    re.compile(r"^\[COMPACTED HISTORY SUMMARY\].*?(?=^\[|\Z)", re.MULTILINE | re.DOTALL),
    re.compile(r"^\[OBSERVATION\].*?(?=^\[|\Z)", re.MULTILINE | re.DOTALL),
    re.compile(r"^\[LOOP WARNING\].*$", re.MULTILINE),
    re.compile(r"^\[TOOL (ERROR|RESULT|OK)\].*$", re.MULTILINE),
]


def strip_internal_blocks(text: str) -> str:
    for pat in _STRIP_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Browser tools
# ---------------------------------------------------------------------------

async def _tool_browser_fetch(params: dict) -> ActionResult:
    url = params.get("url", "")
    if not url or not url.startswith(("http://", "https://")):
        return ActionResult(success=False, error="browser_fetch: invalid or missing url")
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "UltronBot/1.0"})
            resp.raise_for_status()
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()[:3000]
            return ActionResult(extracted_content=text, long_term_memory=f"fetched:{url}")
    except Exception as exc:
        return ActionResult(success=False, error=str(exc)[:300])


async def _tool_browser_agent(params: dict) -> ActionResult:
    task = params.get("task", params.get("url", ""))
    if not task:
        return ActionResult(success=False, error="browser_agent: 'task' param required")
    try:
        from packages.tools.browser_agent import BrowserAgent
        agent = BrowserAgent(llm_fn=params.get("_llm_fn"))
        result_str = await agent.run(task)
        return ActionResult(
            extracted_content=result_str,
            long_term_memory=f"browser_agent:{task[:80]}",
        )
    except ImportError:
        return ActionResult(
            success=False,
            error="browser_agent: playwright not installed (D8).",
        )
    except Exception as exc:
        return ActionResult(success=False, error=str(exc)[:300])


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA: dict = {
    "description": "Search the web for current information.",
    "parameters": {
        "type": "object",
        "properties": {
            "query":       {"type": "string", "description": "Search query, max 100 chars"},
            "max_results": {"type": "integer", "description": "Results count (1-10)", "default": 5},
        },
        "required": ["query"],
    },
}

BROWSER_FETCH_SCHEMA: dict = {
    "description": "Fetch plain text from a URL.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL http:// or https://"},
        },
        "required": ["url"],
    },
}

BROWSER_AGENT_SCHEMA: dict = {
    "description": "Full browser automation for JS-heavy pages.",
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Task description or URL"},
        },
        "required": ["task"],
    },
}


# ---------------------------------------------------------------------------
# ToolRegistry factory
# ---------------------------------------------------------------------------

def build_tool_registry(llm_fn: Any = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("search",         tavily_search,       SEARCH_SCHEMA)
    registry.register("code_exec",      code_exec_tool,      CODE_EXEC_SCHEMA)
    registry.register("shell_exec",     shell_exec_tool,     SHELL_EXEC_SCHEMA)
    registry.register("browser_fetch",  _tool_browser_fetch, BROWSER_FETCH_SCHEMA)
    registry.register("browser_agent",  _tool_browser_agent, BROWSER_AGENT_SCHEMA)
    register_file_ops(registry)
    register_computer_use(registry)
    logger.info(
        f"[ToolRegistry] {len(registry.tool_names)} tools: "
        f"{', '.join(registry.tool_names)}"
    )
    return registry


# ---------------------------------------------------------------------------
# Task classifier
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
# Redis helpers
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
        await redis_client.ltrim(key, -MEM_BUF_MAX, -1)
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
        self.pool           = pool
        self.redis          = redis
        self.settings       = settings
        self.max_iterations = max_iterations
        self.flash_mode     = flash_mode
        self._metacog       = get_metacognition()

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

        # ── Metacognition: pre-action assessment ───────────────────────────
        pre = await self._metacog.pre_action_assessment(
            action=message,
            context={"channel_id": channel_id, "task_type": task_type},
        )
        logger.info(
            f"[Metacog] mode={pre['mode']} confidence={pre['confidence']} "
            f"approach={pre['recommended_approach']}"
        )
        # ──────────────────────────────────────────────────────────────────

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

        # Wire LLM callback into metacog for deep reflection calls (MC4 guard)
        self._metacog.set_llm_callback(llm_call_fn)

        loop = ReActLoop(
            llm_call_fn=llm_call_fn,
            tool_registry=registry,
            flash_mode=self.flash_mode,
            max_iterations=self.max_iterations,
        )

        final_result: Optional[ActionResult] = None
        success = False
        try:
            final_result = await loop.run(
                task=message,
                initial_context=initial_context,
            )
            success = final_result.success if final_result else False
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

        # ── Metacognition: post-action reflection ──────────────────────────
        outcome_text = (
            final_result.extracted_content
            if final_result and final_result.extracted_content
            else (final_result.error if final_result else "no output")
        ) or "no output"
        asyncio.create_task(
            self._metacog.post_action_reflection(
                action=message,
                outcome=outcome_text,
                success=success,
                context={"channel_id": channel_id},
            )
        )  # non-blocking — don't delay response
        # ──────────────────────────────────────────────────────────────────

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
