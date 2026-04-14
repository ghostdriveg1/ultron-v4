"""
packages/brain/task_dispatcher.py

Ultron V4 — Task Orchestration Layer
=====================================
Replaces V3's single-shot LLM dispatch with a full ReAct-loop-backed orchestrator.

Responsibilities:
  1. Classify incoming task (search / code / browser / file / general / done)
  2. Hydrate ToolRegistry with concrete tools (search=Tavily, code_exec, browser_fetch, file_read)
  3. Pull channel AgentState from Redis (persist back after loop finishes)
  4. Call multi-provider key_rotation pool for LLM access (ALL 5 providers)
  5. Run ReActLoop (flash_mode=True for Groq)
  6. Strip internal memory/graph blocks before returning to Discord
  7. Write long_term_memory snippets to Tier2 (Redis buffer → Zilliz flush later)

Informed by:
  - react_loop.py (this repo)          : ReActLoop, ToolRegistry, AgentState, ActionResult
  - llm_router.py (this repo)          : make_provider_llm_fn, multi-provider routing
  - packages/tools/search.py (this repo): tavily_search — real Tavily + DDG fallback
  - browser-use/browser-use            : dispatch pattern, no-vision Groq rule
  - OpenHands/codeact_agent            : pending_actions, function_calling dispatch
  - SAGAR-TAMANG/friday-tony-stark     : system-level tool dispatch (web.py, system.py pattern)
  - dexterai.org architecture          : domain-specific action routing (intent → specialized handler)

Design rules:
  - flash_mode=True default (Groq 8k ctx safe)
  - Per-channel AgentState in Redis (key: ultron:state:{channel_id})
  - LLM calls via make_provider_llm_fn(pool) from llm_router — ALL 5 providers in pool
  - Response to Discord: NEVER leak === MEMORY GRAPH === or [COMPACTED HISTORY] blocks
  - Memory snippets: append to Redis list ultron:mem_buffer:{user_id} (flushed to Zilliz by memory worker)
  - max_iterations=5 default (hard ceiling from react_loop ABSOLUTE_MAX=10)
  - search tool: REAL Tavily implementation (packages/tools/search.py), DDG fallback

Future bug risks (pre-registered):
  D1 [HIGH]   If Redis is unavailable, AgentState load silently returns fresh state →
              loop_detector window clears → B1 risk from react_loop.py fires.
  D2 [HIGH]   Groq key_rotation pool returns None (all keys exhausted) → llm_call_fn
              gets called with None key → provider raises 401 → consecutive_failures max hit
              → loop aborts with no user-facing error message. Need explicit AllKeysExhausted guard.
  D3 [MED]    task_type classifier uses keyword match → ambiguous tasks ("read the latest news"
              could be search OR browser_fetch) → wrong tool called first → wasted iteration.
              Fix: add a lightweight Groq classify call before loop (1 token, no tools).
  D4 [MED]    Redis state TTL not set → channel states accumulate forever → Redis OOM.
              Fix: always set TTL=3600 on AgentState writes.
  D5 [LOW]    mem_buffer list unbounded per user → if memory worker crashes, buffer grows
              indefinitely. Fix: LTRIM to max 100 items after every RPUSH.
  D6 [LOW]    strip_internal_blocks() regex could accidentally strip valid user content
              if user sends a message starting with "=== ". Add start-of-line anchor.

Tool calls used this session:
  Github:get_file_contents x3 (task_dispatcher.py, llm_router.py, v3 bot)
  Github:create_or_update_file x1
  Github:push_files x2
  Notion:notion-fetch x1
  Notion:notion-update-page x1
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Optional

import httpx  # browser_fetch

from packages.brain.react_loop import (
    ActionResult,
    AgentState,
    LoopStatus,
    ReActLoop,
    ToolRegistry,
)
from packages.brain.llm_router import make_provider_llm_fn  # V4 multi-provider router
from packages.tools.search import tavily_search  # REAL search (replaces stub)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDIS_STATE_PREFIX = "ultron:state:"          # key per channel_id
REDIS_MEM_BUF_PREFIX = "ultron:mem_buffer:"  # list per user_id → Zilliz flush
REDIS_STATE_TTL = 3600                        # seconds  (bug D4: always set TTL)
MEM_BUF_MAX = 100                             # max snippets before trim (bug D5)
DEFAULT_MAX_ITERATIONS = 5
GROQ_FLASH_MODE = True                        # lock True; never disable for Groq

# Task-type keywords for lightweight pre-classifier (bug D3: replace with LLM classify later)
TASK_TYPE_KEYWORDS: dict[str, list[str]] = {
    "search": ["search", "find", "who is", "what is", "when", "latest", "news",
               "price", "weather", "stock", "current", "today", "look up"],
    "code_exec": ["run", "execute", "calculate", "compute", "script",
                  "python", "bash", "code", "program"],
    "browser_fetch": ["open", "browse", "visit", "url", "website", "scrape",
                      "read the page", "fetch url", "go to", "https://", "http://"],
    "file_read": ["read file", "open file", "load file", "show file", "my file",
                  "from the file", "uploaded"],
}


# ---------------------------------------------------------------------------
# Internal blocks that must NEVER reach Discord (privacy + UX)
# ---------------------------------------------------------------------------

_STRIP_PATTERNS = [
    re.compile(r"^=== MEMORY GRAPH ===[\s\S]*?(?=^===|\Z)", re.MULTILINE),
    re.compile(r"^\[COMPACTED HISTORY SUMMARY\][\s\S]*?(?=^\[|\Z)", re.MULTILINE),
    re.compile(r"^\[OBSERVATION\][\s\S]*?(?=^\[|\Z)", re.MULTILINE),
    re.compile(r"^\[LOOP WARNING\].*$", re.MULTILINE),
    re.compile(r"^\[TOOL (ERROR|RESULT|OK)\].*$", re.MULTILINE),
]


def strip_internal_blocks(text: str) -> str:
    """Remove all internal orchestration markers from text before sending to Discord."""
    for pat in _STRIP_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Tools — search is REAL (Tavily), others still stub
# ---------------------------------------------------------------------------

async def _tool_code_exec(params: dict) -> ActionResult:
    """Execute sandboxed Python code via subprocess. STUB — Phase 7."""
    code = params.get("code", "")
    if not code:
        return ActionResult(success=False, error="code_exec: code param missing")
    logger.info(f"[code_exec stub] len={len(code)}")
    return ActionResult(extracted_content=f"[CODE_EXEC STUB] Would run: {code[:200]}...")


async def _tool_browser_fetch(params: dict) -> ActionResult:
    """Fetch URL text content (DOM text, no screenshots — Groq token budget)."""
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
        logger.error(f"[browser_fetch] {url}: {exc}")
        return ActionResult(success=False, error=str(exc)[:300])


async def _tool_file_read(params: dict) -> ActionResult:
    """Read uploaded file from Redis CDN buffer. STUB — Phase 7."""
    file_key = params.get("file_key", "")
    if not file_key:
        return ActionResult(success=False, error="file_read: file_key param missing")
    logger.info(f"[file_read stub] key='{file_key}'")
    return ActionResult(extracted_content=f"[FILE_READ STUB] Key: {file_key} — wire Redis CDN.")


# ---------------------------------------------------------------------------
# Tool schema definitions for Groq function_calling
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict] = {
    "search": {
        "description": "Search the web for current information. Use for news, facts, prices, people.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, max 100 chars"},
                "max_results": {"type": "integer", "description": "Number of results (1-10)", "default": 5},
            },
            "required": ["query"],
        },
    },
    "code_exec": {
        "description": "Execute Python code and return stdout. Use for calculations, data transforms.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Valid Python3 code to execute"},
                "timeout": {"type": "integer", "description": "Max seconds (1-30)", "default": 10},
            },
            "required": ["code"],
        },
    },
    "browser_fetch": {
        "description": "Fetch text content from a URL. Use for reading web pages, articles.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL starting with http:// or https://"},
            },
            "required": ["url"],
        },
    },
    "file_read": {
        "description": "Read a file uploaded by the user to Discord. Provide the file_key.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_key": {"type": "string", "description": "Redis key for the uploaded file"},
            },
            "required": ["file_key"],
        },
    },
}


# ---------------------------------------------------------------------------
# ToolRegistry factory
# ---------------------------------------------------------------------------

def build_tool_registry() -> ToolRegistry:
    """Build and return a ToolRegistry with all V4 tools registered.

    search: REAL (Tavily + DDG fallback via packages/tools/search.py)
    code_exec, browser_fetch, file_read: stubs until Phase 7
    """
    registry = ToolRegistry()
    registry.register("search", tavily_search, TOOL_SCHEMAS["search"])  # REAL
    registry.register("code_exec", _tool_code_exec, TOOL_SCHEMAS["code_exec"])
    registry.register("browser_fetch", _tool_browser_fetch, TOOL_SCHEMAS["browser_fetch"])
    registry.register("file_read", _tool_file_read, TOOL_SCHEMAS["file_read"])
    return registry


# ---------------------------------------------------------------------------
# Task-type pre-classifier
# ---------------------------------------------------------------------------

def classify_task(message: str) -> str:
    """Return best-guess primary tool for a message."""
    msg_lower = message.lower()
    scores: dict[str, int] = {t: 0 for t in TASK_TYPE_KEYWORDS}
    for task_type, keywords in TASK_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in msg_lower:
                scores[task_type] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"


# ---------------------------------------------------------------------------
# Redis AgentState persistence helpers
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
            "task": state.task,
            "n_steps": state.n_steps,
            "consecutive_failures": state.consecutive_failures,
            "running_memory": state.running_memory,
            "status": state.status.value,
            "started_at": state.started_at,
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
# Public interface — TaskDispatcher
# ---------------------------------------------------------------------------

class TaskDispatcher:
    """Orchestrates task execution for Ultron V4."""

    def __init__(
        self,
        pool: Any = None,
        redis: Any = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        flash_mode: bool = GROQ_FLASH_MODE,
    ) -> None:
        self.pool = pool
        self.redis = redis
        self.max_iterations = max_iterations
        self.flash_mode = flash_mode

    async def dispatch(
        self,
        message: str,
        channel_id: str,
        user_id: str,
        context: str = "",
    ) -> str:
        """Main entry point. Returns Discord-safe response string."""
        task_type = classify_task(message)
        logger.info(
            f"[TaskDispatcher] channel={channel_id} user={user_id} "
            f"task_type={task_type} msg='{message[:60]}'"
        )

        persisted = await _load_state(self.redis, channel_id)
        initial_context = context
        if persisted and persisted.get("running_memory"):
            initial_context = (
                f"[CONTEXT FROM MEMORY]\n{persisted['running_memory']}\n\n"
                + initial_context
            )

        registry = build_tool_registry()
        llm_call_fn = await make_provider_llm_fn(self.pool)

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
                running_memory=final_result.extracted_content[:500]
                if final_result and final_result.extracted_content
                else "",
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


def get_dispatcher(pool: Any = None, redis: Any = None) -> TaskDispatcher:
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = TaskDispatcher(pool=pool, redis=redis)
        logger.info("[TaskDispatcher] Singleton created")
    return _dispatcher_instance
