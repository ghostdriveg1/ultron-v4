"""
packages/brain/main.py

Ultron V4 — FastAPI Brain Entrypoint
======================================
v31 update:
  Step 5e: get_metacognition() init — MetacognitionEngine singleton
  Step 5f: StructuredStore (Supabase Tier4) init — graceful degrade
  New endpoint: GET /metacog/state — cognitive state dashboard

Startup sequence:
  1. Settings validated
  2. KeyPool built (now 8 providers: +SambaNova +Fireworks +HF)
  3. TaskDispatcher (metacog wired inside)
  4. Memory pipeline: Embedder + ZillizStore + RaptorTree + MemoryWorker
  5. LifecycleEngine + GroundTruthStore + RDLoop
  5c. SpacePromoter
  5d. PlannerAgent
  5e. MetacognitionEngine [v31]
  5f. StructuredStore / Supabase Tier4 [v31]
  6. Sentinel
  7. Council
  8. Background tasks
  9. FastAPI serve port 7860
  10. Discord bot daemon thread

Endpoints:
  POST /infer              — dispatch via TaskDispatcher
  POST /plan               — multi-step via PlannerAgent
  GET  /health             — pool status + component states
  POST /sentinel/event     — Sentinel event handler
  GET  /keys               — pool key status
  GET  /memory/stm/{id}    — STM viewer
  GET  /rd/history/{id}    — R&D history
  GET  /infra/events       — SpacePromoter events
  GET  /metacog/state      — MetacognitionEngine state [v31]

Future bug risks (pre-registered, v31 additions):
  All existing M1-M9, CL1-CL6 apply.
  SB1 [HIGH] Supabase client is sync — wrapped in asyncio.to_thread(). See tier4_supabase.py.
  MC2 [MED]  post_action_reflection concurrent dict mutation — asyncio.Lock in metacog.

Tool calls used writing this file (v31):
  Github:get_file_contents x1 (main.py)
  Github:push_files x1 (batch commit)
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
from packages.brain.planner import PlannerAgent, get_planner
from packages.brain.meta.engine import get_metacognition
from packages.shared.config import get_settings
from packages.shared.exceptions import AllKeysExhaustedError, SentinelKeyUnavailableError
from packages.infrastructure.space_promoter import SpacePromoter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_plan_semaphore = asyncio.Semaphore(1)  # M9 mitigation


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    message:    str
    channel_id: str
    user_id:    str
    guild_id:   Optional[str] = None
    username:   Optional[str] = None


class InferResponse(BaseModel):
    reply:      str
    channel_id: str
    request_id: str
    latency_ms: float


class PlanRequest(BaseModel):
    goal:       str
    channel_id: str
    user_id:    str
    username:   Optional[str] = None
    context:    Optional[str] = ""


class PlanResponse(BaseModel):
    reply:         str
    channel_id:    str
    user_id:       str
    request_id:    str
    latency_ms:    float
    subtask_count: int


class SentinelEvent(BaseModel):
    event_type: str
    payload:    dict[str, Any]
    timestamp:  Optional[float] = None


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(request: Request, settings_token: str) -> None:
    if not settings_token:
        logger.warning("[Auth] ULTRON_AUTH_TOKEN not set — DEV MODE.")
        return
    token = request.headers.get("X-Ultron-Token", "")
    if not hmac.compare_digest(token, settings_token):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Ultron-Token.")


# ---------------------------------------------------------------------------
# Background task: health ping
# ---------------------------------------------------------------------------

async def _health_ping_loop(brain_url: str, interval_seconds: int = 43200) -> None:
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
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Startup] Ultron V4 Brain starting...")
    startup_start = time.monotonic()

    # Step 1: Settings
    try:
        settings = get_settings()
    except RuntimeError as e:
        logger.critical(f"[Startup] Settings FAILED: {e}")
        raise

    # Step 2-3: KeyPool + TaskDispatcher
    pool_config = build_pool_config(settings)
    pool        = KeyPool(pool_config)
    dispatcher  = TaskDispatcher(pool=pool, settings=settings)

    _brain_url = (
        "https://ghostdrive1-ultron1.hf.space"
        if not os.environ.get("LOCAL_DEV")
        else f"http://localhost:{getattr(settings, 'brain_port', 7860)}"
    )

    # Step 4: Memory pipeline
    memory_worker_task: Optional[asyncio.Task] = None
    redis_client = None

    zilliz_uri   = getattr(settings, "zilliz_uri",   "") or os.environ.get("ZILLIZ_URI",   "")
    zilliz_token = getattr(settings, "zilliz_token", "") or os.environ.get("ZILLIZ_TOKEN", "")
    redis_url    = getattr(settings, "redis_url",    "") or os.environ.get("REDIS_URL",    "")

    if zilliz_uri and zilliz_token and redis_url:
        try:
            import redis.asyncio as aioredis
            from packages.memory.embedder    import Embedder
            from packages.memory.tier2_zilliz import ZillizStore
            from packages.memory.raptor      import RaptorTree
            from packages.memory.worker      import MemoryWorker

            redis_client  = aioredis.from_url(redis_url, decode_responses=False)
            embedder      = Embedder()
            zilliz_store  = ZillizStore(uri=zilliz_uri, token=zilliz_token)
            llm_fn        = make_provider_llm_fn(pool)
            raptor_tree   = RaptorTree(embedder=embedder, zilliz_store=zilliz_store, llm_fn=llm_fn)
            mem_worker    = MemoryWorker(redis=redis_client, embedder=embedder, raptor_tree=raptor_tree)

            memory_worker_task    = asyncio.create_task(mem_worker.run())
            app.state.embedder    = embedder
            app.state.zilliz_store = zilliz_store
            app.state.raptor_tree = raptor_tree
            app.state.redis       = redis_client
            logger.info("[Startup] Memory pipeline: ACTIVE")
        except Exception as e:
            logger.warning(f"[Startup] Memory pipeline init failed (non-fatal): {e}")
    else:
        logger.warning("[Startup] Memory pipeline DISABLED")
        if redis_url:
            try:
                import redis.asyncio as aioredis
                redis_client    = aioredis.from_url(redis_url, decode_responses=False)
                app.state.redis = redis_client
                logger.info("[Startup] Redis-only client active")
            except Exception as e:
                logger.warning(f"[Startup] Redis init failed (non-fatal): {e}")

    # Step 5b: Lifecycle + GroundTruth + RDLoop
    lifecycle = None
    gt_store  = None
    rd_loop   = None

    if redis_client is not None:
        try:
            from packages.memory.lifecycle    import LifecycleEngine
            from packages.memory.ground_truth import GroundTruthStore
            from packages.brain.rd_loop       import RDLoop

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

    # Step 5c: SpacePromoter
    _promoter_stop: asyncio.Event = asyncio.Event()
    _promoter_task: Optional[asyncio.Task] = None
    try:
        promoter = SpacePromoter(redis_client=redis_client)
        _promoter_task = asyncio.create_task(promoter.run(_promoter_stop))
        app.state.promoter = promoter
        logger.info("[Startup] SpacePromoter: ACTIVE")
    except Exception as e:
        logger.warning(f"[Startup] SpacePromoter init failed (non-fatal): {e}")

    # Step 5d: PlannerAgent
    try:
        planner = get_planner(pool=pool, redis=redis_client)
        app.state.planner = planner
        logger.info("[Startup] PlannerAgent: ACTIVE")
    except Exception as e:
        logger.warning(f"[Startup] PlannerAgent init failed (non-fatal): {e}")

    # Step 5e: MetacognitionEngine [v31]
    try:
        metacog = get_metacognition()
        app.state.metacog = metacog
        logger.info("[Startup] MetacognitionEngine: ACTIVE")
    except Exception as e:
        logger.warning(f"[Startup] MetacognitionEngine init failed (non-fatal): {e}")

    # Step 5f: StructuredStore / Supabase Tier4 [v31]
    try:
        from packages.memory.tier4_supabase import get_structured_store
        supabase_url = os.environ.get("SUPABASE_URL", "")
        supabase_key = os.environ.get("SUPABASE_KEY", "")
        tier4 = get_structured_store(supabase_url=supabase_url, supabase_key=supabase_key)
        await tier4.initialize()
        app.state.tier4 = tier4
        logger.info(
            f"[Startup] StructuredStore (Tier4): "
            f"{'ACTIVE' if tier4.get_status()['connected'] else 'DEGRADED (no creds)'}"
        )
    except Exception as e:
        logger.warning(f"[Startup] StructuredStore init failed (non-fatal): {e}")

    # Step 6: Sentinel
    sentinel = None
    try:
        from packages.brain.sentinel import build_sentinel
        sentinel = build_sentinel(settings)
        if sentinel:
            logger.info("[Startup] Sentinel: ACTIVE (Gemini 2.5 Pro)")
        else:
            logger.warning("[Startup] Sentinel: INACTIVE (GEMINI_SENTINEL_KEY not set)")
    except Exception as e:
        logger.warning(f"[Startup] Sentinel init failed (non-fatal): {e}")

    # Step 7: Council
    try:
        from packages.brain.council import Council
        council = Council(pool=pool, redis=getattr(app.state, "redis", None))
        app.state.council = council
        logger.info("[Startup] Council Mode: ACTIVE")
    except Exception as e:
        logger.warning(f"[Startup] Council init failed (non-fatal): {e}")

    # Step 8: Core state
    app.state.settings   = settings
    app.state.pool       = pool
    app.state.dispatcher = dispatcher
    app.state.sentinel   = sentinel
    app.state.start_time = startup_start

    # Step 9: Health ping background task
    _ping_task = asyncio.create_task(
        _health_ping_loop(_brain_url, interval_seconds=43200)
    )

    # Step 10: Discord bot
    _bot_thread = None
    if getattr(settings, "discord_token", None):
        try:
            import packages.brain.discord_bot as _discord_bot
            _bot_lifecycle = getattr(app.state, "lifecycle", None)
            _bot_thread = threading.Thread(
                target=_discord_bot.run,
                kwargs={"redis": None, "lifecycle": _bot_lifecycle},
                daemon=True,
                name="ultron-discord-bot",
            )
            _bot_thread.start()
            logger.info("[Startup] Discord bot thread started.")
        except Exception as e:
            logger.warning(f"[Startup] Discord bot failed (non-fatal): {e}")
    else:
        logger.warning("[Startup] Discord bot SKIPPED — website-only mode.")

    elapsed = (time.monotonic() - startup_start) * 1000
    logger.info(
        f"[Startup] Ultron V4 Brain READY in {elapsed:.1f}ms. "
        f"pool_general={len(pool.general)} "
        f"sentinel={'ACTIVE' if sentinel else 'INACTIVE'} "
        f"memory={'ACTIVE' if memory_worker_task else 'INACTIVE'} "
        f"metacog={'ACTIVE' if hasattr(app.state, 'metacog') else 'INACTIVE'} "
        f"tier4={'ACTIVE' if hasattr(app.state, 'tier4') and getattr(app.state, 'tier4', None) and app.state.tier4.get_status()['connected'] else 'DEGRADED'} "
        f"discord_bot={'ACTIVE' if _bot_thread else 'SKIPPED'}"
    )

    yield

    # Shutdown
    logger.info("[Shutdown] Cancelling background tasks...")
    _promoter_stop.set()
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
    logger.info("[Shutdown] Ultron V4 Brain stopped cleanly.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ultron V4 Brain",
    version="4.0.0",
    description="Ultron V4 — Agentic AI OS Brain.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


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
    pool_status   = await pool.status()
    uptime        = time.monotonic() - request.app.state.start_time
    status        = "ok" if pool_status["general_available"] > 0 else "degraded"

    promoter_status = None
    if hasattr(request.app.state, "promoter"):
        try:
            promoter_status = request.app.state.promoter.get_status()
        except Exception:
            pass

    tier4_connected = False
    if hasattr(request.app.state, "tier4"):
        try:
            tier4_connected = request.app.state.tier4.get_status()["connected"]
        except Exception:
            pass

    return JSONResponse({
        "status":           status,
        "uptime_seconds":   round(uptime, 1),
        "version":          "4.0.0",
        "memory_pipeline":  hasattr(request.app.state, "raptor_tree"),
        "sentinel_active":  request.app.state.sentinel is not None,
        "council_active":   hasattr(request.app.state, "council"),
        "lifecycle_active": hasattr(request.app.state, "lifecycle") and request.app.state.lifecycle is not None,
        "promoter_active":  hasattr(request.app.state, "promoter"),
        "planner_active":   hasattr(request.app.state, "planner"),
        "metacog_active":   hasattr(request.app.state, "metacog"),
        "tier4_connected":  tier4_connected,
        "promoter":         promoter_status,
        "pool": {
            "general_available":  pool_status["general_available"],
            "general_total":      len(pool_status["general"]),
            "sentinel_available": pool_status["sentinel_available"],
            "sentinel_total":     len(pool_status["sentinel"]),
        },
    })


@app.post("/infer", response_model=InferResponse)
async def infer(body: InferRequest, request: Request) -> InferResponse:
    req_id  = getattr(request.state, "request_id", "?")
    t_start = time.monotonic()

    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    dispatcher: TaskDispatcher = request.app.state.dispatcher

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
        raise HTTPException(status_code=503, detail="All LLM keys exhausted.")
    except Exception as e:
        logger.exception(f"[/infer] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Brain error.")

    latency_ms = (time.monotonic() - t_start) * 1000
    return InferResponse(
        reply=reply,
        channel_id=body.channel_id,
        request_id=req_id,
        latency_ms=round(latency_ms, 1),
    )


@app.post("/plan", response_model=PlanResponse)
async def plan(body: PlanRequest, request: Request) -> PlanResponse:
    req_id  = getattr(request.state, "request_id", "?")
    t_start = time.monotonic()

    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    planner: Optional[PlannerAgent] = getattr(request.app.state, "planner", None)
    if planner is None:
        raise HTTPException(status_code=503, detail="PlannerAgent not initialized.")

    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is not None:
        asyncio.create_task(
            lifecycle.ingest(
                user_id=body.user_id,
                channel_id=body.channel_id,
                raw_text=body.goal,
                metadata={"source": "plan", "username": body.username or "user"},
            )
        )

    try:
        async with _plan_semaphore:
            reply = await planner.run(
                goal=body.goal,
                channel_id=body.channel_id,
                user_id=body.user_id,
                initial_context=body.context or "",
            )
    except AllKeysExhaustedError:
        raise HTTPException(status_code=503, detail="All LLM keys exhausted.")
    except Exception as e:
        logger.exception(f"[/plan] PlannerAgent raised: {e}")
        raise HTTPException(status_code=500, detail="Planner error.")

    latency_ms    = (time.monotonic() - t_start) * 1000
    subtask_count = reply.count("**[") or 1

    return PlanResponse(
        reply=reply,
        channel_id=body.channel_id,
        user_id=body.user_id,
        request_id=req_id,
        latency_ms=round(latency_ms, 1),
        subtask_count=subtask_count,
    )


@app.post("/sentinel/event")
async def sentinel_event(body: SentinelEvent, request: Request) -> JSONResponse:
    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    ts       = body.timestamp or time.time()
    sentinel = request.app.state.sentinel

    logger.info(f"[Sentinel] event_type={body.event_type} payload_keys={list(body.payload.keys())}")

    if sentinel is None:
        return JSONResponse({
            "status":     "logged_only",
            "reason":     "Sentinel inactive",
            "event_type": body.event_type,
        })

    try:
        result = await sentinel.handle_event(event_type=body.event_type, payload=body.payload)
        return JSONResponse({"status": "ok", "event_type": body.event_type, **result})
    except Exception as e:
        return JSONResponse(
            {"status": "error", "error": str(e), "event_type": body.event_type},
            status_code=500,
        )


@app.get("/keys")
async def keys_status(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    pool: KeyPool = request.app.state.pool
    pool_status   = await pool.status()

    providers: dict[str, dict] = {}
    for key_info in pool_status.get("general", []):
        provider = key_info.get("provider", "unknown")
        if provider not in providers:
            providers[provider] = {"total": 0, "available": 0, "in_cooldown": 0}
        providers[provider]["total"] += 1
        if not key_info.get("tripped", False):
            providers[provider]["available"] += 1
        else:
            providers[provider]["in_cooldown"] += 1

    sentinel_keys      = pool_status.get("sentinel", [])
    sentinel_available = sum(1 for k in sentinel_keys if not k.get("tripped", False))

    return JSONResponse({
        "general": {
            "providers": providers,
            "total":     len(pool_status.get("general", [])),
            "available": pool_status.get("general_available", 0),
        },
        "sentinel": {
            "total":     len(sentinel_keys),
            "available": sentinel_available,
        },
    })


@app.get("/memory/stm/{channel_id}")
async def memory_stm(channel_id: str, request: Request) -> JSONResponse:
    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return JSONResponse({"error": "Redis not available"}, status_code=503)

    ctx_key = f"ultron:ctx:{channel_id}"
    try:
        entries  = await redis.lrange(ctx_key, 0, -1)
        messages = [e.decode() if isinstance(e, bytes) else e for e in entries]
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    lifecycle = getattr(request.app.state, "lifecycle", None)
    lifecycle_cells: List[dict] = []
    if lifecycle is not None:
        try:
            cells = await lifecycle.get_stm(channel_id)
            lifecycle_cells = [c.to_dict() for c in cells]
        except Exception:
            pass

    return JSONResponse({
        "channel_id":           channel_id,
        "context_window":       messages,
        "context_window_count": len(messages),
        "lifecycle_cells":      lifecycle_cells,
        "lifecycle_cell_count": len(lifecycle_cells),
    })


@app.get("/rd/history/{user_id}")
async def rd_history(user_id: str, request: Request) -> JSONResponse:
    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    rd_loop = getattr(request.app.state, "rd_loop", None)
    if rd_loop is None:
        return JSONResponse({"error": "RDLoop not initialized"}, status_code=503)

    try:
        limit        = int(request.query_params.get("limit", 20))
        improvements = await rd_loop.get_history(user_id, limit=limit)
        state        = await rd_loop.get_state(user_id)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({
        "user_id":           user_id,
        "improvements":      [i.to_dict() for i in improvements],
        "improvement_count": len(improvements),
        "rd_state":          state.to_dict() if state else None,
    })


@app.get("/infra/events")
async def infra_events(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return JSONResponse([], status_code=200)

    try:
        raw_entries = await redis.lrange("ultron:infra:events", 0, -1)
        events = []
        for entry in raw_entries:
            try:
                decoded = entry.decode() if isinstance(entry, bytes) else entry
                events.append(json.loads(decoded))
            except Exception:
                events.append({"raw": str(entry)})
        return JSONResponse(events)
    except Exception as e:
        return JSONResponse([], status_code=200)


@app.get("/metacog/state")
async def metacog_state(request: Request) -> JSONResponse:
    """MetacognitionEngine cognitive state — for dashboard / debugging. [v31]"""
    settings = request.app.state.settings
    _check_auth(request, getattr(settings, "ultron_auth_token", ""))

    metacog = getattr(request.app.state, "metacog", None)
    if metacog is None:
        return JSONResponse({"error": "MetacognitionEngine not initialized"}, status_code=503)

    return JSONResponse(metacog.get_state())


# ---------------------------------------------------------------------------
# Entrypoint
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
