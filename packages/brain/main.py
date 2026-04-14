"""
packages/brain/main.py

Ultron V4 — FastAPI Brain Entrypoint
======================================
Startup sequence:
  1. Settings validated (fail loud if required env vars missing)
  2. KeyPool built from config_loader.build_pool_config()
  3. TaskDispatcher instantiated with pool
  4. Background health-ping task started (Sentinel keep-alive)
  5. FastAPI app begins serving on port 7860 (HF Spaces standard)

Endpoints:
  POST /infer          — Discord bot → Brain. Auth: X-Ultron-Token header.
  GET  /health         — CF Worker keep-alive + Sentinel audit. No auth.
  POST /sentinel/event — Sentinel writes incident/routing decision. Auth required.

Design decisions:
  - asynccontextmanager lifespan (FastAPI 0.93+ pattern). No @app.on_event.
  - All shared state (pool, dispatcher) stored in app.state — no module-level globals.
  - Auth via simple token comparison (ULTRON_AUTH_TOKEN). Replace with JWT in v5.
  - /health is intentionally unauthenticated — CF Worker pings it every 24h.
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

Tool calls used writing this file:
    Github:get_file_contents x1 (pool.py — confirmed KeyPool.status() shape)
    Github:get_file_contents x1 (task_dispatcher.py — confirmed TaskDispatcher.__init__ + dispatch sig)
    External read: litellm proxy_server.py (lifespan pattern, startup sequence)
"""

from __future__ import annotations

import asyncio
import hmac
import logging
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
    """Validate X-Ultron-Token header. Raises 401 if invalid or missing.

    Bug M3: str equality — replace with hmac.compare_digest for timing safety.
    """
    if not settings_token:
        # No token configured — skip auth (dev mode). Log every time.
        logger.warning("[Auth] ULTRON_AUTH_TOKEN not set — skipping auth (DEV MODE).")
        return

    token = request.headers.get("X-Ultron-Token", "")
    # Bug M3: should be hmac.compare_digest but token is plain str here
    if not hmac.compare_digest(token, settings_token):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Ultron-Token.")


# ---------------------------------------------------------------------------
# Background task: health ping (Sentinel keep-alive)
# ---------------------------------------------------------------------------

async def _health_ping_loop(brain_url: str, interval_seconds: int = 86400) -> None:
    """Ping /health every N seconds to keep HF Space warm.

    CF Worker also pings every 24h via GitHub Actions — this is a belt-and-suspenders
    self-ping from inside the Space.

    Bug M6: no restart-on-error guard yet. Wrap in try/except in future.
    """
    await asyncio.sleep(60)   # Initial delay — let startup finish
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
    """FastAPI lifespan context. Runs startup before yield, shutdown after.

    Startup order (strict — each step depends on previous):
      1. Parse + validate Settings (fail loud)
      2. Build KeyPool config dict
      3. Instantiate KeyPool
      4. Instantiate TaskDispatcher with pool
      5. Start background health-ping task
      6. Yield → app serves requests

    Shutdown:
      7. Cancel background tasks
      8. Log shutdown

    Bug M4: if any import at top of file failed, app crashes here before startup.
    """
    logger.info("[Startup] Ultron V4 Brain starting...")
    startup_start = time.monotonic()

    # ── Step 1: Settings ──────────────────────────────────────────────────
    try:
        settings = get_settings()
    except RuntimeError as e:
        logger.critical(f"[Startup] Settings validation FAILED: {e}")
        raise

    # ── Step 2: KeyPool config ────────────────────────────────────────────
    pool_config = build_pool_config(settings)

    # ── Step 3: KeyPool ───────────────────────────────────────────────────
    pool = KeyPool(pool_config)

    # ── Step 4: TaskDispatcher ────────────────────────────────────────────
    dispatcher = TaskDispatcher(pool=pool, settings=settings)

    # ── Step 5: Store in app.state (no module-level globals) ──────────────
    app.state.settings   = settings
    app.state.pool       = pool
    app.state.dispatcher = dispatcher
    app.state.start_time = startup_start

    # ── Step 6: Background tasks ──────────────────────────────────────────
    # Self-ping to prevent HF Space from going to sleep mid-day
    # Use the public HF Space URL if set, otherwise localhost
    _brain_url = (
        f"https://ghostdrive1-ultron1.hf.space"
        if not __import__("os").environ.get("LOCAL_DEV")
        else f"http://localhost:{settings.brain_port}"
    )
    _ping_task = asyncio.create_task(
        _health_ping_loop(_brain_url, interval_seconds=43200)  # 12hr self-ping
    )

    elapsed = (time.monotonic() - startup_start) * 1000
    logger.info(
        f"[Startup] Ultron V4 Brain ready in {elapsed:.1f}ms. "
        f"Port={settings.brain_port}. "
        f"General keys={len(pool.general)}. "
        f"Sentinel={'ACTIVE' if pool.sentinel else 'INACTIVE'}."
    )

    # ── Yield: serve requests ─────────────────────────────────────────────
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("[Shutdown] Cancelling background tasks...")
    _ping_task.cancel()
    try:
        await _ping_task
    except asyncio.CancelledError:
        pass
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

# CORS — open for now (CF Worker and Discord bot are only callers)
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
    """Inject X-Request-ID into every request + response for tracing."""
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
    """CF Worker keep-alive + Sentinel audit endpoint.

    Returns:
        {
          "status": "ok" | "degraded",
          "uptime_seconds": float,
          "pool": {general_available, sentinel_available, ...},
          "version": "4.0.0"
        }

    'degraded' if general_available == 0 (all keys tripped).
    Intentionally unauthenticated — CF Worker + Sentinel ping without token.
    """
    pool: KeyPool = request.app.state.pool
    pool_status = await pool.status()

    uptime = time.monotonic() - request.app.state.start_time
    status = "ok" if pool_status["general_available"] > 0 else "degraded"

    return JSONResponse({
        "status":             status,
        "uptime_seconds":     round(uptime, 1),
        "version":            "4.0.0",
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

    Flow:
      1. Auth check (X-Ultron-Token)
      2. Delegate to TaskDispatcher.dispatch()
      3. Return Discord-safe reply string

    Bug M2: no concurrency cap. Add asyncio.Semaphore in next session.

    Raises:
        401: bad/missing token
        503: all LLM keys exhausted (AllKeysExhaustedError)
        500: unexpected error
    """
    req_id  = getattr(request.state, "request_id", "?")
    t_start = time.monotonic()

    settings: object = request.app.state.settings
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
    """Sentinel writes routing decisions and incident reports here.

    Currently: log + acknowledge. Full Sentinel integration in Phase 4.

    Bug M5: no schema validation on body.payload — any dict accepted.
    """
    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    ts = body.timestamp or time.time()
    logger.info(
        f"[Sentinel] event_type={body.event_type} "
        f"ts={ts:.0f} payload_keys={list(body.payload.keys())}"
    )

    # Future Phase 4: route by event_type:
    #   routing_decision → write to CF KV
    #   incident         → write Notion incident page
    #   weekly_audit     → write Notion weekly report page
    #   health_check     → update routing table

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
        port=settings.brain_port,
        log_level="info",
        access_log=True,
    )
