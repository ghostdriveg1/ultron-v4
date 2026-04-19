"""
packages/brain/main.py

Ultron V4 — FastAPI Brain Entrypoint
======================================
Startup sequence:
  1. Settings validated (fail loud if required env vars missing)
  2. KeyPool built from config_loader.build_pool_config()
  3. TaskDispatcher instantiated with pool
  4. Memory pipeline: Embedder + ZillizStore + RaptorTree + MemoryWorker
  5. LifecycleEngine + GroundTruthStore + RDLoop (if Redis available)
  5c. SpacePromoter (optional — health-check loop + CF KV promotion)
  6. Sentinel instantiated (if GEMINI_SENTINEL_KEY set)
  7. Council instantiated (always — uses general pool)
  8. Background tasks: health-ping + memory flush worker
  9. FastAPI app begins serving on port 7860 (HF Spaces standard)
  10. Discord bot: blocking .run() in daemon thread (lifecycle-aware)

Endpoints:
  POST /infer                   — Discord bot -> Brain. Auth: X-Ultron-Token header.
  GET  /health                  — CF Worker keep-alive + Sentinel audit. No auth.
  POST /sentinel/event          — Sentinel writes incident/routing decision. Auth required.
  GET  /keys                    — Pool status + key counts per provider (website dashboard).
  GET  /memory/stm/{channel_id} — Redis STM context viewer for website Memory tab.
  GET  /rd/history/{user_id}    — R&D loop implemented improvements for website.

Design decisions:
  - asynccontextmanager lifespan (FastAPI 0.93+ pattern). No @app.on_event.
  - All shared state (pool, dispatcher) stored in app.state — no module-level globals.
  - Auth via simple token comparison (ULTRON_AUTH_TOKEN). Replace with JWT in v5.
  - /health is intentionally unauthenticated — CF Worker + Sentinel ping without token.
  - Structured JSON responses everywhere. Discord-formatted strings only at bot layer.
  - Request IDs injected via middleware for distributed tracing readiness.
  - Discord bot runs in daemon thread (discord.py owns its own asyncio event loop).
  - SpacePromoter runs as asyncio task inside FastAPI's event loop (pure async).

Future bug risks (pre-registered):
  M1 [HIGH]   HF Spaces can spin up MULTIPLE workers for the same Space on scale events.
              app.state is per-worker — two workers get two independent KeyPool instances.
              Both track failures independently -> provider quotas burned 2x faster.
              Fix (future): move failure state to Redis (pool P1) so all workers share.

  M2 [HIGH]   /infer has no request queuing. If 10 Discord messages arrive simultaneously,
              10 ReAct loops start concurrently. Each uses 3-5 LLM calls -> 30-50 concurrent
              API hits -> mass 429 storm. Fix: asyncio.Semaphore(max_concurrent=3) on /infer.

  M3 [MED]    X-Ultron-Token comparison is timing-attack vulnerable (str == str).
              Fix: use hmac.compare_digest() instead. Low priority for free-tier.

  M4 [MED]    TaskDispatcher.dispatch() is assumed to be defined. If import fails
              (react_loop.py or llm_router.py has a bug), entire app crashes at startup.
              Fix: wrap imports in try/except at startup -> log error -> fail with 503
              instead of silent crash.

  M5 [LOW]    /sentinel/event SentinelEvent Pydantic model added (v22).
              Now wired to sentinel.handle_event(). If Sentinel is INACTIVE
              (no GEMINI_SENTINEL_KEY), event is logged only (no Gemini call).

  M6 [LOW]    Lifespan background task (health_ping) has no cancellation guard.
              If the task raises an exception, it dies silently — no restart.
              Fix: wrap task body in try/except + re-schedule on error.

  CL1 [HIGH]  Memory worker (MemoryWorker) is optional — if ZILLIZ_URI or ZILLIZ_TOKEN
              missing, worker is skipped. Verify settings check doesn't raise on missing
              optional vars. config.py warns but does not fail on optional Zilliz vars.

  CL2 [MED]   MemoryWorker shares Redis client with discord_bot.py — if bot is in same
              process and calls LTRIM on wrong key, worker may lose buffered messages.
              Fix: worker uses strictly prefixed keys (ultron:mem_buffer:*).

  CL3 [MED]   llm_fn passed to RaptorTree is make_provider_llm_fn(pool). Pool may be
              exhausted when RAPTOR summariser calls it. AllKeysExhaustedError caught
              in raptor.py._summarise_cluster() -> returns None (partial tree).

  CL4 [LOW]   Redis client in main.py lifespan created via redis.asyncio.from_url().
              If REDIS_URL not set, Redis init will fail at startup. Should degrade
              gracefully (disable memory worker) rather than crash entire Brain.

  CL5 [MED]   LifecycleEngine._locks dict grows unbounded across users in long-running
              process. Each new user_id adds an asyncio.Lock. In high-traffic scenarios
              (100+ users) this leaks memory. Fix: use WeakValueDictionary or LRU cache.

  CL6 [LOW]   RDLoop.run() is not started as a background task in main.py — it is
              triggered externally (post-task completion). If called from /infer handler,
              it blocks the response. Fix: always asyncio.create_task() for RDLoop.run().

  M7 [MED]    Discord bot thread holds a reference to redis_client (aioredis). aioredis
              client is created in FastAPI's asyncio event loop. Bot thread runs its own
              event loop (discord.py). Cross-loop Redis calls from bot thread will raise
              "bound to different event loop". Fix v26: bot thread calls Brain /infer HTTP
              (already the architecture) — never calls Redis directly from bot thread.
              No issue in current design; flag for future if bot ever goes direct-Redis.

  M8 [LOW]    SpacePromoter _promoter_stop event created in lifespan scope; if lifespan
              exits before promoter task starts (edge case on fast shutdown), stop event
              is set before task reads it — task exits immediately. Acceptable: only on
              startup crash scenarios.

Tool calls used writing this file (v26):
    Github:get_file_contents x1 (main.py current state + sha)
    Github:get_file_contents x1 (discord_bot.py — run() signature)
    Github:get_file_contents x1 (space_promoter.py — SpacePromoter.run() signature)
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from packages.brain.key_rotation.config_loader import build_pool_config
from packages.brain.key_rotation.pool import KeyPool
from packages.brain.task_dispatcher import TaskDispatcher
from packages.brain.llm_router import make_provider_llm_fn
from packages.shared.config import get_settings
from packages.shared.exceptions import AllKeysExhaustedError, SentinelKeyUnavailableError
from packages.brain import discord_bot as _discord_bot  # type: ignore
from packages.infrastructure.space_promoter import SpacePromoter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    message: str
    channel_id: str
    user_id: str
    guild_id: Optional[str] = None
    username: Optional[str] = None


class InferResponse(BaseModel):
    reply: str
    channel_id: str
    request_id: str
    latency_ms: float


class SentinelEvent(BaseModel):
    event_type: str            # "space_failure" | "routing_override" | "health_check" | "weekly_audit" | "project_plan"
    payload: dict[str, Any]
    timestamp: Optional[float] = None


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(request: Request, settings_token: str) -> None:
    """Validate X-Ultron-Token header. Raises 401 if invalid or missing."""
    if not settings_token:
        logger.warning("[Auth] ULTRON_AUTH_TOKEN not set — skipping auth (DEV MODE).")
        return

    token = request.headers.get("X-Ultron-Token", "")
    if not hmac.compare_digest(token, settings_token):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Ultron-Token.")


# ---------------------------------------------------------------------------
# Background task: health ping (keep HF Space warm)
# ---------------------------------------------------------------------------

async def _health_ping_loop(brain_url: str, interval_seconds: int = 43200) -> None:
    """Ping /health every N seconds to keep HF Space warm."""
    await asyncio.sleep(60)
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                resp = await client.get(f"{brain_url}/health")
                logger.debug(f"[HealthPing] self-ping -> {resp.status_code}")
            except Exception as e:
                logger.warning(f"[HealthPing] self-ping failed: {e}")
            await asyncio.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# Lifespan: startup + shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context."""
    logger.info("[Startup] Ultron V4 Brain starting...")
    startup_start = time.monotonic()

    # ── Step 1: Settings ──────────────────────────────────────────────────
    try:
        settings = get_settings()
    except RuntimeError as e:
        logger.critical(f"[Startup] Settings validation FAILED: {e}")
        raise

    # ── Step 2–3: KeyPool ─────────────────────────────────────────────────
    pool_config = build_pool_config(settings)
    pool = KeyPool(pool_config)

    # ── Step 4: TaskDispatcher ────────────────────────────────────────────
    dispatcher = TaskDispatcher(pool=pool, settings=settings)

    # ── Resolve brain_url early (needed by RDLoop + health ping) ──────────
    _brain_url = (
        "https://ghostdrive1-ultron1.hf.space"
        if not os.environ.get("LOCAL_DEV")
        else f"http://localhost:{getattr(settings, 'brain_port', 7860)}"
    )

    # ── Step 5: Memory pipeline (optional — degrades gracefully if Zilliz unset) ──
    memory_worker_task: Optional[asyncio.Task] = None
    redis_client = None

    zilliz_uri   = getattr(settings, "zilliz_uri", "") or os.environ.get("ZILLIZ_URI", "")
    zilliz_token = getattr(settings, "zilliz_token", "") or os.environ.get("ZILLIZ_TOKEN", "")
    redis_url    = getattr(settings, "redis_url", "") or os.environ.get("REDIS_URL", "")

    if zilliz_uri and zilliz_token and redis_url:
        try:
            import redis.asyncio as aioredis  # type: ignore
            from packages.memory.embedder import Embedder
            from packages.memory.tier2_zilliz import ZillizStore
            from packages.memory.raptor import RaptorTree
            from packages.memory.worker import MemoryWorker

            redis_client = aioredis.from_url(redis_url, decode_responses=False)
            embedder     = Embedder()
            zilliz_store = ZillizStore(uri=zilliz_uri, token=zilliz_token)
            llm_fn       = make_provider_llm_fn(pool)
            raptor_tree  = RaptorTree(embedder=embedder, zilliz_store=zilliz_store, llm_fn=llm_fn)
            mem_worker   = MemoryWorker(redis=redis_client, embedder=embedder, raptor_tree=raptor_tree)

            memory_worker_task = asyncio.create_task(mem_worker.run())

            app.state.embedder     = embedder
            app.state.zilliz_store = zilliz_store
            app.state.raptor_tree  = raptor_tree
            app.state.redis        = redis_client

            logger.info("[Startup] Memory pipeline: Embedder + ZillizStore + RaptorTree + MemoryWorker ACTIVE")
        except Exception as e:
            logger.warning(f"[Startup] Memory pipeline init failed (non-fatal): {e}")
    else:
        logger.warning("[Startup] Memory pipeline DISABLED — ZILLIZ_URI/ZILLIZ_TOKEN/REDIS_URL not set")

        # Still init Redis alone for context windows + council state (if URL available)
        if redis_url:
            try:
                import redis.asyncio as aioredis  # type: ignore
                redis_client = aioredis.from_url(redis_url, decode_responses=False)
                app.state.redis = redis_client
                logger.info("[Startup] Redis-only client active (no Zilliz)")
            except Exception as e:
                logger.warning(f"[Startup] Redis init failed (non-fatal): {e}")

    # ── Step 5b: Lifecycle + GroundTruth + RDLoop (optional — needs Redis) ──
    lifecycle = None
    gt_store  = None
    rd_loop   = None

    if redis_client is not None:
        try:
            from packages.memory.lifecycle import LifecycleEngine
            from packages.memory.ground_truth import GroundTruthStore
            from packages.brain.rd_loop import RDLoop

            discord_webhook = os.environ.get("DISCORD_WEBHOOK_URL", "") or None

            lifecycle = LifecycleEngine(redis_client)
            gt_store  = GroundTruthStore(redis_client)
            rd_loop   = RDLoop(
                redis_client=redis_client,
                lifecycle=lifecycle,
                brain_url=_brain_url,
                auth_token=getattr(settings, "ultron_auth_token", ""),
                discord_webhook=discord_webhook,
            )

            app.state.lifecycle = lifecycle
            app.state.gt_store  = gt_store
            app.state.rd_loop   = rd_loop

            logger.info("[Startup] LifecycleEngine + GroundTruthStore + RDLoop: ACTIVE")
        except Exception as e:
            logger.warning(f"[Startup] Lifecycle/GT/RDLoop init failed (non-fatal): {e}")
    else:
        logger.warning("[Startup] Lifecycle/GT/RDLoop DISABLED — Redis not available")

    # ── Step 5c: SpacePromoter (optional — health-check + CF KV promotion) ─
    _promoter_stop: asyncio.Event = asyncio.Event()
    _promoter_task: Optional[asyncio.Task] = None
    try:
        promoter = SpacePromoter(redis_client=redis_client)
        _promoter_task = asyncio.create_task(promoter.run(_promoter_stop))
        app.state.promoter = promoter
        logger.info("[Startup] SpacePromoter: ACTIVE")
    except Exception as e:
        logger.warning(f"[Startup] SpacePromoter init failed (non-fatal): {e}")

    # ── Step 6: Sentinel (optional — degrades gracefully if key unset) ────
    sentinel = None
    try:
        from packages.brain.sentinel import build_sentinel
        sentinel = build_sentinel(settings)
        if sentinel:
            logger.info("[Startup] Sentinel: ACTIVE (Gemini 2.5 Pro dedicated key)")
        else:
            logger.warning("[Startup] Sentinel: INACTIVE (GEMINI_SENTINEL_KEY not set)")
    except Exception as e:
        logger.warning(f"[Startup] Sentinel init failed (non-fatal): {e}")

    # ── Step 7: Council (always active — uses general pool) ───────────────
    try:
        from packages.brain.council import Council
        council = Council(
            pool=pool,
            redis=getattr(app.state, "redis", None),
        )
        app.state.council = council
        logger.info("[Startup] Council Mode: ACTIVE")
    except Exception as e:
        logger.warning(f"[Startup] Council init failed (non-fatal): {e}")

    # ── Step 8: Store core state ──────────────────────────────────────────
    app.state.settings   = settings
    app.state.pool       = pool
    app.state.dispatcher = dispatcher
    app.state.sentinel   = sentinel
    app.state.start_time = startup_start

    # ── Step 9: Background health ping ────────────────────────────────────
    _ping_task = asyncio.create_task(
        _health_ping_loop(_brain_url, interval_seconds=43200)
    )

    # ── Step 10: Discord bot (daemon thread — discord.py owns its own event loop) ──
    # M7: bot only calls Brain /infer via HTTP, never touches Redis directly —
    # no cross-loop aioredis issue. Passing redis_client as reference is safe
    # because bot module receives it but only the on_message handler uses it
    # for Redis calls that run inside its own thread's event loop via discord.py.
    # NOTE: discord_bot._ctx_append/_ctx_get use asyncio internally. They run
    # in the bot's own event loop (created by discord.py in the bot thread) —
    # NOT in FastAPI's loop. The aioredis client created here is bound to
    # FastAPI's loop. WORKAROUND: bot module must create its OWN aioredis client
    # internally if Redis calls needed. For now, Redis is passed but aioredis
    # may raise cross-loop errors — tracked as M7. Mitigation: pass redis=None
    # until bot-side Redis is refactored to create its own client.
    _bot_lifecycle = getattr(app.state, "lifecycle", None)
    _bot_thread = threading.Thread(
        target=_discord_bot.run,
        kwargs={"redis": None, "lifecycle": _bot_lifecycle},  # M7: redis=None until bot refactor
        daemon=True,
        name="ultron-discord-bot",
    )
    _bot_thread.start()
    logger.info("[Startup] Discord bot thread started.")

    elapsed = (time.monotonic() - startup_start) * 1000
    lifecycle_active = hasattr(app.state, "lifecycle") and app.state.lifecycle is not None
    logger.info(
        f"[Startup] Ultron V4 Brain READY in {elapsed:.1f}ms. "
        f"Pool general={len(pool.general)} sentinel={'ACTIVE' if sentinel else 'INACTIVE'} "
        f"council=ACTIVE memory={'ACTIVE' if memory_worker_task else 'INACTIVE'} "
        f"lifecycle={'ACTIVE' if lifecycle_active else 'INACTIVE'} "
        f"promoter={'ACTIVE' if _promoter_task else 'INACTIVE'} "
        f"discord_bot=ACTIVE"
    )

    # ── Yield: serve requests ─────────────────────────────────────────────
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("[Shutdown] Cancelling background tasks...")
    _promoter_stop.set()  # signal SpacePromoter to exit cleanly
    _ping_task.cancel()
    if memory_worker_task:
        memory_worker_task.cancel()
        try:
            await memory_worker_task
        except asyncio.CancelledError:
            pass
    if _promoter_task:
        _promoter_task.cancel()
        try:
            await _promoter_task
        except asyncio.CancelledError:
            pass
    try:
        await _ping_task
    except asyncio.CancelledError:
        pass
    if redis_client:
        await redis_client.aclose()
    if hasattr(app.state, "zilliz_store"):
        await app.state.zilliz_store.close()
    # Discord bot thread is daemon — dies with process. No explicit join needed.
    logger.info("[Shutdown] Ultron V4 Brain stopped cleanly.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ultron V4 Brain",
    version="4.0.0",
    description="Ultron V4 — Agentic AI OS Brain. FastAPI + ReAct + Multi-provider LLM pool.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = req_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(request: Request) -> JSONResponse:
    pool: KeyPool = request.app.state.pool
    pool_status = await pool.status()
    uptime = time.monotonic() - request.app.state.start_time
    status = "ok" if pool_status["general_available"] > 0 else "degraded"

    promoter_status = None
    if hasattr(request.app.state, "promoter"):
        try:
            promoter_status = request.app.state.promoter.get_status()
        except Exception:
            pass

    return JSONResponse({
        "status":             status,
        "uptime_seconds":     round(uptime, 1),
        "version":            "4.0.0",
        "memory_pipeline":    hasattr(request.app.state, "raptor_tree"),
        "sentinel_active":    request.app.state.sentinel is not None,
        "council_active":     hasattr(request.app.state, "council"),
        "lifecycle_active":   hasattr(request.app.state, "lifecycle") and request.app.state.lifecycle is not None,
        "promoter_active":    hasattr(request.app.state, "promoter"),
        "promoter":           promoter_status,
        "pool": {
            "general_available":  pool_status["general_available"],
            "general_total":      len(pool_status["general"]),
            "sentinel_available": pool_status["sentinel_available"],
            "sentinel_total":     len(pool_status["sentinel"]),
        },
    })


@app.post("/infer", response_model=InferResponse)
async def infer(body: InferRequest, request: Request) -> InferResponse:
    """Main Discord -> Brain endpoint."""
    req_id  = getattr(request.state, "request_id", "?")
    t_start = time.monotonic()

    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    dispatcher: TaskDispatcher = request.app.state.dispatcher

    # Fire lifecycle.ingest() as background task (non-blocking) — CL6 mitigation
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is not None:
        asyncio.create_task(
            lifecycle.ingest(
                user_id=body.user_id,
                channel_id=body.channel_id,
                raw_text=body.message,
                metadata={"source": "infer", "username": body.username or "user"},
            )
        )

    try:
        reply = await dispatcher.dispatch(
            message=body.message,
            channel_id=body.channel_id,
            user_id=body.user_id,
            username=body.username or "user",
        )
    except AllKeysExhaustedError as e:
        logger.error(f"[/infer] req_id={req_id} AllKeysExhaustedError: {e}")
        raise HTTPException(
            status_code=503,
            detail="All LLM keys are exhausted or in cooldown. Try again later."
        )
    except Exception as e:
        logger.exception(f"[/infer] req_id={req_id} Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Brain error.")

    latency_ms = (time.monotonic() - t_start) * 1000
    logger.info(
        f"[/infer] req_id={req_id} channel={body.channel_id} "
        f"latency={latency_ms:.0f}ms reply_len={len(reply)}"
    )

    return InferResponse(
        reply=reply,
        channel_id=body.channel_id,
        request_id=req_id,
        latency_ms=round(latency_ms, 1),
    )


@app.post("/sentinel/event")
async def sentinel_event(body: SentinelEvent, request: Request) -> JSONResponse:
    """
    Sentinel writes routing decisions and incident reports here.
    Fully wired in v22: delegates to Sentinel.handle_event() if Sentinel active.
    If Sentinel inactive (no GEMINI_SENTINEL_KEY), logs event and returns 200.
    """
    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    ts       = body.timestamp or time.time()
    sentinel = request.app.state.sentinel

    logger.info(
        f"[Sentinel] event_type={body.event_type} "
        f"ts={ts:.0f} payload_keys={list(body.payload.keys())}"
    )

    if sentinel is None:
        logger.warning(
            f"[Sentinel] Event received but Sentinel INACTIVE "
            f"(GEMINI_SENTINEL_KEY not set). event_type={body.event_type}"
        )
        return JSONResponse({
            "status": "logged_only",
            "reason": "Sentinel inactive — set GEMINI_SENTINEL_KEY",
            "event_type": body.event_type,
        })

    try:
        result = await sentinel.handle_event(
            event_type=body.event_type,
            payload=body.payload,
        )
        return JSONResponse({"status": "ok", "event_type": body.event_type, **result})
    except Exception as e:
        logger.error(f"[Sentinel] handle_event failed: {e}")
        return JSONResponse(
            {"status": "error", "error": str(e), "event_type": body.event_type},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Website API endpoints
# ---------------------------------------------------------------------------

@app.get("/keys")
async def keys_status(request: Request) -> JSONResponse:
    """
    Returns per-provider key pool status for the website Credentials dashboard.
    Auth required.
    """
    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    pool: KeyPool = request.app.state.pool
    pool_status = await pool.status()

    # Build per-provider breakdown
    providers: dict[str, dict] = {}
    for key_info in pool_status.get("general", []):
        provider = key_info.get("provider", "unknown")
        if provider not in providers:
            providers[provider] = {"total": 0, "available": 0, "in_cooldown": 0}
        providers[provider]["total"] += 1
        if key_info.get("available", False):
            providers[provider]["available"] += 1
        else:
            providers[provider]["in_cooldown"] += 1

    sentinel_keys = pool_status.get("sentinel", [])
    sentinel_available = sum(1 for k in sentinel_keys if k.get("available", False))

    return JSONResponse({
        "general": {
            "providers": providers,
            "total": len(pool_status.get("general", [])),
            "available": pool_status.get("general_available", 0),
        },
        "sentinel": {
            "total": len(sentinel_keys),
            "available": sentinel_available,
        },
    })


@app.get("/memory/stm/{channel_id}")
async def memory_stm(channel_id: str, request: Request) -> JSONResponse:
    """
    Returns the STM (short-term memory) context for a channel.
    Used by website Memory tab — STM view.
    Auth required.
    """
    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return JSONResponse({"error": "Redis not available"}, status_code=503)

    # Read raw Redis context window (set by discord_bot.py)
    ctx_key = f"ultron:ctx:{channel_id}"
    try:
        entries = await redis.lrange(ctx_key, 0, -1)
        messages = [
            e.decode() if isinstance(e, bytes) else e
            for e in entries
        ]
    except Exception as e:
        logger.warning(f"[/memory/stm] Redis read failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

    # Also return lifecycle STM if available
    lifecycle = getattr(request.app.state, "lifecycle", None)
    lifecycle_cells: List[dict] = []
    if lifecycle is not None:
        try:
            # Use channel_id as user_id proxy for STM lookup (cells stored per user_id)
            cells = await lifecycle.get_stm(channel_id)
            lifecycle_cells = [c.to_dict() for c in cells]
        except Exception as e:
            logger.warning(f"[/memory/stm] lifecycle.get_stm failed: {e}")

    return JSONResponse({
        "channel_id": channel_id,
        "context_window": messages,
        "context_window_count": len(messages),
        "lifecycle_cells": lifecycle_cells,
        "lifecycle_cell_count": len(lifecycle_cells),
    })


@app.get("/rd/history/{user_id}")
async def rd_history(user_id: str, request: Request) -> JSONResponse:
    """
    Returns the R&D loop implemented improvements for a user.
    Used by website Projects tab — R&D history view.
    Auth required.
    """
    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    rd_loop = getattr(request.app.state, "rd_loop", None)
    if rd_loop is None:
        return JSONResponse({"error": "RDLoop not initialized — Redis required"}, status_code=503)

    try:
        limit = int(request.query_params.get("limit", 20))
        improvements = await rd_loop.get_history(user_id, limit=limit)
        state = await rd_loop.get_state(user_id)
    except Exception as e:
        logger.warning(f"[/rd/history] failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({
        "user_id": user_id,
        "improvements": [i.to_dict() for i in improvements],
        "improvement_count": len(improvements),
        "rd_state": state.to_dict() if state else None,
    })


# ---------------------------------------------------------------------------
# Entrypoint (uvicorn)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "packages.brain.main:app",
        host="0.0.0.0",
        port=getattr(settings, "brain_port", 7860),
        log_level="info",
        access_log=True,
    )
