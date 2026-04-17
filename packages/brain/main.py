"""
packages/brain/main.py

Ultron V4 — FastAPI Brain Entrypoint
======================================
Startup sequence:
  1. Settings validated (fail loud if required env vars missing)
  2. KeyPool built from config_loader.build_pool_config()
  3. TaskDispatcher instantiated with pool
  4. Memory pipeline: Embedder + ZillizStore + RaptorTree + MemoryWorker
  5. Background tasks: health-ping + memory flush worker
  6. FastAPI app begins serving on port 7860 (HF Spaces standard)

Endpoints:
  POST /infer          — Discord bot → Brain. Auth: X-Ultron-Token header.
  GET  /health         — CF Worker keep-alive + Sentinel audit. No auth.
  POST /sentinel/event — Sentinel writes incident/routing decision. Auth required.

Design decisions:
  - asynccontextmanager lifespan (FastAPI 0.93+ pattern). No @app.on_event.
  - All shared state (pool, dispatcher) stored in app.state — no module-level globals.
  - Auth via simple token comparison (ULTRON_AUTH_TOKEN). Replace with JWT in v5.
  - /health is intentionally unauthenticated — CF Worker + Sentinel ping without token.
  - Structured JSON responses everywhere. Discord-formatted strings only at bot layer.
  - Request IDs injected via middleware for distributed tracing readiness.

Future bug risks (pre-registered):
  M1 [HIGH]   HF Spaces can spin up MULTIPLE workers for the same Space on scale events.
              app.state is per-worker — two workers get two independent KeyPool instances.
              Both track failures independently → provider quotas burned 2x faster.
              Fix (future): move failure state to Redis (pool P1) so all workers share.

  M2 [HIGH]   /infer has no request queuing. If 10 Discord messages arrive simultaneously,
              10 ReAct loops start concurrently. Each uses 3-5 LLM calls → 30-50 concurrent
              API hits → mass 429 storm. Fix: asyncio.Semaphore(max_concurrent=3) on /infer.

  M3 [MED]    X-Ultron-Token comparison is timing-attack vulnerable (str == str).
              Fix: use hmac.compare_digest() instead. Low priority for free-tier.

  M4 [MED]    TaskDispatcher.dispatch() is assumed to be defined. If import fails
              (react_loop.py or llm_router.py has a bug), entire app crashes at startup.
              Fix: wrap imports in try/except at startup → log error → fail with 503
              instead of silent crash.

  M5 [LOW]    /sentinel/event accepts any JSON body currently. No schema validation.
              A malformed payload from Sentinel (future) will KeyError inside the handler.
              Fix: add Pydantic model for SentinelEvent when Sentinel layer is written.

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
              in raptor.py._summarise_cluster() — returns None (partial tree).

  CL4 [LOW]   Redis client in main.py lifespan created via redis.asyncio.from_url().
              If REDIS_URL not set, Redis init will fail at startup. Should degrade
              gracefully (disable memory worker) rather than crash entire Brain.

Tool calls used writing this file:
    Github:get_file_contents x1 (pool.py — confirmed KeyPool.status() shape)
    Github:get_file_contents x1 (task_dispatcher.py — confirmed TaskDispatcher.__init__ + dispatch sig)
    External read: litellm proxy_server.py (lifespan pattern, startup sequence)
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

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
    event_type: str            # "routing_decision" | "incident" | "health_check" | "weekly_audit"
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
# Background task: health ping (Sentinel keep-alive)
# ---------------------------------------------------------------------------

async def _health_ping_loop(brain_url: str, interval_seconds: int = 43200) -> None:
    """Ping /health every N seconds to keep HF Space warm."""
    await asyncio.sleep(60)
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                resp = await client.get(f"{brain_url}/health")
                logger.debug(f"[HealthPing] self-ping → {resp.status_code}")
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

            # Store in app.state for future access (e.g. /infer memory inject)
            app.state.embedder     = embedder
            app.state.zilliz_store = zilliz_store
            app.state.raptor_tree  = raptor_tree
            app.state.redis        = redis_client

            logger.info("[Startup] Memory pipeline: Embedder + ZillizStore + RaptorTree + MemoryWorker ACTIVE")
        except Exception as e:
            logger.warning(f"[Startup] Memory pipeline init failed (non-fatal): {e}")
    else:
        logger.warning("[Startup] Memory pipeline DISABLED — ZILLIZ_URI/ZILLIZ_TOKEN/REDIS_URL not set")

    # ── Step 6: Store core state ──────────────────────────────────────────
    app.state.settings   = settings
    app.state.pool       = pool
    app.state.dispatcher = dispatcher
    app.state.start_time = startup_start

    # ── Step 7: Background health ping ────────────────────────────────────
    _brain_url = (
        "https://ghostdrive1-ultron1.hf.space"
        if not os.environ.get("LOCAL_DEV")
        else f"http://localhost:{getattr(settings, 'brain_port', 7860)}"
    )
    _ping_task = asyncio.create_task(
        _health_ping_loop(_brain_url, interval_seconds=43200)
    )

    elapsed = (time.monotonic() - startup_start) * 1000
    logger.info(
        f"[Startup] Ultron V4 Brain ready in {elapsed:.1f}ms. "
        f"Port={getattr(settings, 'brain_port', 7860)}. "
        f"General keys={len(pool.general)}. "
        f"Sentinel={'ACTIVE' if pool.sentinel else 'INACTIVE'}."
    )

    # ── Yield: serve requests ─────────────────────────────────────────────
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("[Shutdown] Cancelling background tasks...")
    _ping_task.cancel()
    if memory_worker_task:
        memory_worker_task.cancel()
        try:
            await memory_worker_task
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

    return JSONResponse({
        "status":             status,
        "uptime_seconds":     round(uptime, 1),
        "version":            "4.0.0",
        "memory_pipeline":    hasattr(request.app.state, "raptor_tree"),
        "pool": {
            "general_available":  pool_status["general_available"],
            "general_total":      len(pool_status["general"]),
            "sentinel_available": pool_status["sentinel_available"],
            "sentinel_total":     len(pool_status["sentinel"]),
        },
    })


@app.post("/infer", response_model=InferResponse)
async def infer(body: InferRequest, request: Request) -> InferResponse:
    """Main Discord → Brain endpoint.

    Bug M2: no concurrency cap. Add asyncio.Semaphore in next session.
    """
    req_id  = getattr(request.state, "request_id", "?")
    t_start = time.monotonic()

    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    dispatcher: TaskDispatcher = request.app.state.dispatcher

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
    """Sentinel writes routing decisions and incident reports here. Full wiring in Phase 4."""
    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    ts = body.timestamp or time.time()
    logger.info(
        f"[Sentinel] event_type={body.event_type} "
        f"ts={ts:.0f} payload_keys={list(body.payload.keys())}"
    )
    return JSONResponse({"status": "received", "event_type": body.event_type})


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
