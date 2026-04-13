"""
packages/brain/task_dispatcher.py

Ultron V4 — Task Orchestration Layer
=====================================
Replaces V3's single-shot LLM dispatch with a full ReAct-loop-backed orchestrator.

Responsibilities:
  1. Classify incoming task (search / code / browser / file / general / done)
  2. Hydrate ToolRegistry with concrete stub tools (search, code_exec, browser_fetch, file_read)
  3. Pull channel AgentState from Redis (persist back after loop finishes)
  4. Call Groq key_rotation pool for LLM access
  5. Run ReActLoop (flash_mode=True for Groq)
  6. Strip internal memory/graph blocks before returning to Discord
  7. Write long_term_memory snippets to Tier2 (Redis buffer → Zilliz flush later)

Informed by:
  - react_loop.py (this repo)          : ReActLoop, ToolRegistry, AgentState, ActionResult
  - browser-use/browser-use            : dispatch pattern, no-vision Groq rule
  - OpenHands/codeact_agent            : pending_actions, function_calling dispatch
  - SAGAR-TAMANG/friday-tony-stark     : system-level tool dispatch (web.py, system.py pattern)
  - dexterai.org architecture          : domain-specific action routing (intent → specialized handler)
  - manus.im / GenSpark / MiniMax      : multi-step orchestration, streaming UX, tool-call chaining

Design rules:
  - flash_mode=True default (Groq 8k ctx safe)
  - Per-channel AgentState in Redis (key: ultron:state:{channel_id})
  - Groq key_rotation: call pool.get_key() before every LLM call
  - Response to Discord: NEVER leak === MEMORY GRAPH === or [COMPACTED HISTORY] blocks
  - Memory snippets: append to Redis list ultron:mem_buffer:{user_id} (flushed to Zilliz by memory worker)
  - max_iterations=5 default (hard ceiling from react_loop ABSOLUTE_MAX=10)
  - Tool stubs: all 4 tools are real async functions with TODO internals
    (search → Tavily free tier, code_exec → subprocess sandbox, browser_fetch → httpx, file_read → Redis CDN)

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
  Github:get_file_contents x2 (react_loop.py, friday/tools/),
  Github:push_files x1,
  web_fetch x1 (dexterai.org),
  Notion:notion-fetch x1,
  Notion:notion-update-page x1
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict
from typing import Any, Optional

import httpx  # browser_fetch + Tavily search stub

from packages.brain.react_loop import (
    ActionResult,
    AgentState,
    LoopStatus,
    ReActLoop,
    ToolRegistry,
)

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
    """Remove all internal orchestration markers from text before sending to Discord.

    Bug D6: start-of-line anchor (^) in patterns prevents stripping valid user content.
    """
    for pat in _STRIP_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Tool stubs — real async functions (internals TODO, interface LOCKED)
# ---------------------------------------------------------------------------

async def _tool_search(params: dict) -> ActionResult:
    """Web search via Tavily free-tier API.

    params: {query: str, max_results: int = 5}
    TODO: inject TAVILY_API_KEY from config. Currently returns stub result.
    Groq tool_use pattern (friday/tools/web.py style): call API → extract snippets → return.
    """
    query = params.get("query", "")
    # max_results = params.get("max_results", 5)
    if not query:
        return ActionResult(success=False, error="search: query param missing")

    # TODO: replace stub with real Tavily call
    # async with httpx.AsyncClient() as client:
    #     r = await client.post("https://api.tavily.com/search",
    #                           json={"api_key": TAVILY_KEY, "query": query,
    #                                 "max_results": max_results})
    #     data = r.json()
    #     snippets = [r["content"] for r in data.get("results", [])]
    #     return ActionResult(extracted_content="\n\n".join(snippets[:3]))

    logger.info(f"[search stub] query='{query}'")
    return ActionResult(
        extracted_content=f"[SEARCH STUB] Results for: {query} — wire Tavily key to activate.",
        long_term_memory=f"search:{query}",
    )


async def _tool_code_exec(params: dict) -> ActionResult:
    """Execute sandboxed Python code via subprocess.

    params: {code: str, timeout: int = 10}
    Runs code in a subprocess with stdout/stderr capture.
    TODO: add resource limits (CPU/memory) and disallow imports of os.system, subprocess.
    Friday/friday-tony-stark pattern: subprocess call → capture → return.
    """
    code = params.get("code", "")
    timeout = min(int(params.get("timeout", 10)), 30)  # hard cap 30s
    if not code:
        return ActionResult(success=False, error="code_exec: code param missing")

    # TODO: replace stub with real subprocess sandbox
    # proc = await asyncio.create_subprocess_exec(
    #     "python3", "-c", code,
    #     stdout=asyncio.subprocess.PIPE,
    #     stderr=asyncio.subprocess.PIPE,
    # )
    # stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    # if proc.returncode != 0:
    #     return ActionResult(success=False, error=stderr.decode()[:1000])
    # return ActionResult(extracted_content=stdout.decode()[:2000])

    logger.info(f"[code_exec stub] code length={len(code)}")
    return ActionResult(
        extracted_content=f"[CODE_EXEC STUB] Would run: {code[:200]}...",
    )


async def _tool_browser_fetch(params: dict) -> ActionResult:
    """Fetch a URL and return text content (no vision, DOM text only for Groq token budget).

    params: {url: str, selector: str = None}
    Uses httpx + basic HTML strip. No screenshots (10x token saving vs vision).
    browser-use pattern: DOM text only for Groq (vision=False hard rule).
    TODO: upgrade to playwright for JS-rendered pages.
    """
    url = params.get("url", "")
    if not url or not url.startswith(("http://", "https://")):
        return ActionResult(success=False, error="browser_fetch: invalid or missing url")

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "UltronBot/1.0"})
            resp.raise_for_status()
            # Strip HTML tags (basic)
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()[:3000]
            return ActionResult(
                extracted_content=text,
                long_term_memory=f"fetched:{url}",
            )
    except Exception as exc:
        logger.error(f"[browser_fetch] {url}: {exc}")
        return ActionResult(success=False, error=str(exc)[:300])


async def _tool_file_read(params: dict) -> ActionResult:
    """Read a file from Redis CDN buffer (files uploaded via Discord → stored in Redis).

    params: {file_key: str}  — Redis key set by discord_bot on upload
    TODO: implement Redis GET + base64 decode for binary files.
    """
    file_key = params.get("file_key", "")
    if not file_key:
        return ActionResult(success=False, error="file_read: file_key param missing")

    # TODO: replace stub with Redis GET
    # redis_client = get_redis()  # from shared config
    # content = await redis_client.get(file_key)
    # if not content:
    #     return ActionResult(success=False, error=f"file_read: key '{file_key}' not found in Redis")
    # return ActionResult(extracted_content=content.decode()[:3000])

    logger.info(f"[file_read stub] key='{file_key}'")
    return ActionResult(
        extracted_content=f"[FILE_READ STUB] Key: {file_key} — wire Redis to activate.",
    )


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
    """Build and return a ToolRegistry with all V4 stub tools registered.

    Called once per dispatch. Registry is lightweight — no shared state issues.
    """
    registry = ToolRegistry()
    registry.register("search", _tool_search, TOOL_SCHEMAS["search"])
    registry.register("code_exec", _tool_code_exec, TOOL_SCHEMAS["code_exec"])
    registry.register("browser_fetch", _tool_browser_fetch, TOOL_SCHEMAS["browser_fetch"])
    registry.register("file_read", _tool_file_read, TOOL_SCHEMAS["file_read"])
    return registry


# ---------------------------------------------------------------------------
# Task-type pre-classifier (keyword-based, replace with LLM in Phase 3 — bug D3)
# ---------------------------------------------------------------------------

def classify_task(message: str) -> str:
    """Return best-guess primary tool for a message.

    Returns one of: search, code_exec, browser_fetch, file_read, general.
    Bug D3: ambiguous tasks route incorrectly. Replace with 1-call Groq classify later.
    """
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
    """Load AgentState dict from Redis. Returns None if not found or Redis down.

    Bug D1: Redis unavailable → returns None → fresh AgentState created →
    loop_detector window clears → react_loop.py B1 risk fires.
    """
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
    """Persist AgentState to Redis. Always set TTL (bug D4)."""
    if redis_client is None:
        return
    try:
        key = f"{REDIS_STATE_PREFIX}{channel_id}"
        # Serialize only serializable fields
        state_dict = {
            "task": state.task,
            "n_steps": state.n_steps,
            "consecutive_failures": state.consecutive_failures,
            "running_memory": state.running_memory,
            "status": state.status.value,
            "started_at": state.started_at,
            # results/message_history intentionally NOT persisted (too large, rebuilt fresh)
        }
        await redis_client.set(key, json.dumps(state_dict), ex=REDIS_STATE_TTL)
    except Exception as exc:
        logger.warning(f"[TaskDispatcher] Redis state save failed: {exc}")


async def _buffer_memory(redis_client: Any, user_id: str, snippet: str) -> None:
    """Append memory snippet to Redis buffer list. Trim to MEM_BUF_MAX (bug D5)."""
    if redis_client is None or not snippet:
        return
    try:
        key = f"{REDIS_MEM_BUF_PREFIX}{user_id}"
        await redis_client.rpush(key, snippet)
        await redis_client.ltrim(key, -MEM_BUF_MAX, -1)  # keep last 100
    except Exception as exc:
        logger.warning(f"[TaskDispatcher] mem_buffer write failed: {exc}")


# ---------------------------------------------------------------------------
# Groq LLM call wrapper — integrates with key_rotation pool
# ---------------------------------------------------------------------------

async def _make_groq_llm_fn(pool: Any):
    """Return an async llm_call_fn bound to the key_rotation pool.

    The returned function is passed to ReActLoop as llm_call_fn.
    Bug D2: if pool.get_key() returns None (all exhausted), the call will fail
    with 401. The AllKeysExhausted guard below converts this to a clean error.

    pool expected interface:
      key_obj = await pool.get_key()  → {key, provider, model}
      await pool.report_success(key_obj.key_id)
      await pool.report_failure(key_obj.key_id)
    """
    async def llm_call_fn(messages: list[dict], tools: list[dict]) -> Optional[dict]:
        if pool is None:
            logger.error("[TaskDispatcher] No key pool provided — cannot call LLM")
            return None

        key_obj = None
        try:
            key_obj = await pool.get_key()  # raises AllKeysExhaustedError if empty
        except Exception as exc:
            # Bug D2: AllKeysExhausted → return None → consecutive_failures increments
            logger.error(f"[TaskDispatcher] Key pool exhausted: {exc}")
            return None

        if key_obj is None:
            logger.error("[TaskDispatcher] pool.get_key() returned None")
            return None

        try:
            import httpx as _httpx  # local import to avoid circular
            payload = {
                "model": key_obj.get("model", "llama-3.3-70b-versatile"),
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.3,
                "response_format": {"type": "json_object"},  # Groq JSON mode
            }
            # Only add tools if non-empty (Groq rejects empty tools array)
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            async with _httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key_obj['key']}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    await pool.report_failure(key_obj["key_id"])
                    logger.warning(f"[TaskDispatcher] Groq {resp.status_code} → key failure reported")
                    return None

                resp.raise_for_status()
                data = resp.json()
                await pool.report_success(key_obj["key_id"])

                # Extract content from Groq response
                choice = data["choices"][0]["message"]
                # If tool_calls present, extract first tool call args as content
                if choice.get("tool_calls"):
                    tc = choice["tool_calls"][0]
                    fn_args = tc["function"].get("arguments", "{}")
                    # Wrap as action dict so react_loop can parse it
                    try:
                        args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        args = {}
                    return {
                        "content": json.dumps({
                            "memory": "",
                            "action_type": tc["function"]["name"],
                            "action_params": args,
                        })
                    }

                return {"content": choice.get("content", "{}")}

        except Exception as exc:
            if key_obj:
                try:
                    await pool.report_failure(key_obj["key_id"])
                except Exception:
                    pass
            logger.error(f"[TaskDispatcher] LLM call error: {exc}")
            return None

    return llm_call_fn


# ---------------------------------------------------------------------------
# Public interface — TaskDispatcher
# ---------------------------------------------------------------------------

class TaskDispatcher:
    """Orchestrates task execution for Ultron V4.

    Usage (from discord_bot.py or FastAPI handler)::

        dispatcher = TaskDispatcher(pool=key_pool, redis=redis_client)
        response = await dispatcher.dispatch(
            message="What's the current Bitcoin price?",
            channel_id="1234567890",
            user_id="ghost_uid",
        )
        # response is a clean string, safe to send to Discord

    Design (Dexterai-inspired):
      - Each incoming message → classified by intent → routed to appropriate tool chain
      - ReAct loop handles multi-step (search → fetch → synthesize)
      - Response stripped of all internal blocks before return
    """

    def __init__(
        self,
        pool: Any = None,              # key_rotation pool instance (V3-compatible interface)
        redis: Any = None,             # aioredis or upstash async client
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
        """Main entry point. Returns Discord-safe response string.

        Steps:
          1. Classify task intent
          2. Load channel state from Redis
          3. Build ToolRegistry + LLM fn
          4. Run ReActLoop
          5. Buffer memory snippets
          6. Save state back to Redis
          7. Strip internal blocks → return
        """
        task_type = classify_task(message)
        logger.info(
            f"[TaskDispatcher] channel={channel_id} user={user_id} "
            f"task_type={task_type} msg='{message[:60]}'"
        )

        # Load persisted state (AgentState continuity across messages)
        # Bug D1: if Redis down, fresh state created, B1 risk from react_loop fires
        persisted = await _load_state(self.redis, channel_id)
        initial_context = context
        if persisted and persisted.get("running_memory"):
            initial_context = (
                f"[CONTEXT FROM MEMORY]\n{persisted['running_memory']}\n\n"
                + initial_context
            )

        # Build components
        registry = build_tool_registry()
        llm_call_fn = await _make_groq_llm_fn(self.pool)

        loop = ReActLoop(
            llm_call_fn=llm_call_fn,
            tool_registry=registry,
            flash_mode=self.flash_mode,
            max_iterations=self.max_iterations,
        )

        # Run ReAct loop
        # Bug B4 from react_loop: if loop crashes, memory write below never fires.
        # Wrap in try/finally to guarantee state save.
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
            # Always save state + buffer memory (bug B4 fix)
            # Reconstruct minimal AgentState for persistence
            _state = AgentState(
                task=message,
                running_memory=final_result.extracted_content[:500]
                if final_result and final_result.extracted_content
                else "",
            )
            await _save_state(self.redis, channel_id, _state)

            if final_result and final_result.long_term_memory:
                await _buffer_memory(self.redis, user_id, final_result.long_term_memory)

        # Build user-facing response
        if final_result is None or (not final_result.success and final_result.error):
            response = (
                f"Sorry, I ran into an issue completing that task."
                + (f" ({final_result.error[:100]})" if final_result else "")
            )
        elif final_result.extracted_content:
            response = final_result.extracted_content
        else:
            response = "Task completed, but no output was produced."

        # Strip all internal orchestration blocks (bug D6: start-of-line anchored)
        response = strip_internal_blocks(response)

        # Discord 2000-char limit guard
        if len(response) > 1800:
            response = response[:1797] + "..."

        return response


# ---------------------------------------------------------------------------
# Module-level singleton factory (optional convenience for main.py)
# ---------------------------------------------------------------------------

_dispatcher_instance: Optional[TaskDispatcher] = None


def get_dispatcher(pool: Any = None, redis: Any = None) -> TaskDispatcher:
    """Return or create the global TaskDispatcher singleton.

    Called from main.py on startup. Pass pool + redis once.
    """
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = TaskDispatcher(pool=pool, redis=redis)
        logger.info("[TaskDispatcher] Singleton created")
    return _dispatcher_instance
