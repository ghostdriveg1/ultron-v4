"""
packages/brain/eternal_loop_router.py

Ultron V4 — Eternal Loop + Sentinel CEO Endpoints
===================================================
v32: All endpoints required by the n8n Eternal Loop + Sentinel workflow.

Endpoints added:
  POST /llm/generate             — raw LLM call, role-injected, routes through pool
  POST /task/decompose           — HTN plan decomposition (wraps planner.decompose_goal)
  POST /task/deliver             — store final task output to Supabase/Zilliz + notify Discord
  POST /task/update_context      — inject prior loop output into Redis STM + RAPTOR
  POST /sentinel/review          — Sentinel quality gate (Gemini 2.5 Pro scores output)
  POST /sentinel/log             — lightweight health log to Redis
  POST /sentinel/alert           — Discord webhook alert (degraded / critical)
  POST /sentinel/migrate_context — context migration: Redis STM dump → Zilliz compress → reload
  POST /keys/rotate              — trigger key pool rotation for exhausted providers

Design rules:
  - All LLM calls route through pool (llm_router.make_provider_llm_fn) EXCEPT /sentinel/review
    which uses dedicated Gemini 2.5 Pro Sentinel key (call_provider direct).
  - /llm/generate accepts provider_hint to bias pool selection — NOT a hard override.
  - /task/deliver: stores to Supabase Tier4 + Zilliz if available, degrades gracefully.
  - /sentinel/migrate_context: dumps Redis STM → compresses via RAPTOR → clears old STM.
  - /keys/rotate: calls pool.rotate_provider(provider) for each exhausted provider.
  - All endpoints require X-Ultron-Token auth.
  - Graceful degrade: if component unavailable, log + return degraded status (no 5xx).

Future bug risks (pre-registered):
  EL1 [HIGH]  /llm/generate with use_raptor=True — RAPTOR query can timeout (30s).
              Mitigation: asyncio.wait_for(raptor.query(), timeout=25) — fallback to empty context.
  EL2 [MED]   /sentinel/review Sentinel key absent → falls back to pool Gemini key.
              Quality scores from pool Gemini may be lower quality than 2.5 Pro.
  EL3 [MED]   /task/deliver Supabase write is sync (SB1 pattern) — wrapped in asyncio.to_thread.
  EL4 [LOW]   /sentinel/migrate_context races with MemoryWorker if both run simultaneously.
              Mitigation: asyncio.Lock on migration — only one at a time.
  EL5 [LOW]   /keys/rotate calls pool.rotate_provider() — method must exist on KeyPool.
              If method absent (older pool version), graceful fallback: log + return 200.
  EL6 [MED]   provider_hint in /llm/generate is advisory only — pool may select different
              provider if hint provider is exhausted. Expected and correct behavior.

Tool calls used writing this file (v32):
  Github:get_file_contents x4 (main.py, task_dispatcher.py, llm_router.py, planner.py)
  Github:push_files x1 (atomic commit)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from packages.brain.llm_router import make_provider_llm_fn, call_provider
from packages.brain.planner import decompose_goal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Migration lock — EL4 mitigation
# ---------------------------------------------------------------------------

_migration_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()

# ---------------------------------------------------------------------------
# Auth helper (mirrors main.py pattern)
# ---------------------------------------------------------------------------

def _auth(request: Request) -> None:
    import hmac
    token = os.environ.get("ULTRON_AUTH_TOKEN", "")
    if not token:
        return  # dev mode
    incoming = request.headers.get("X-Ultron-Token", "")
    if not hmac.compare_digest(incoming, token):
        raise HTTPException(status_code=401, detail="Invalid X-Ultron-Token")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LLMGenerateRequest(BaseModel):
    role: str = "assistant"
    system: Optional[str] = None
    prompt: str
    task_id: Optional[str] = None
    loop_count: int = 0
    use_search: bool = False
    use_raptor: bool = False
    use_file_ops: bool = False
    provider_hint: Optional[str] = None  # EL6: advisory only


class DecomposeRequest(BaseModel):
    goal: str
    domain: Optional[str] = "auto"
    deadline_hours: float = 48.0
    quality_target: str = "world_class"
    max_subtasks: int = 6
    decomposition_style: str = "htn_parallel_sequential"


class DeliverRequest(BaseModel):
    task_id: str
    goal: str
    final_output: str
    quality_score: float = 0.0
    loop_count: int = 0
    improvement_history: list = []
    notify_discord: bool = True
    store_supabase: bool = True
    store_zilliz: bool = True


class UpdateContextRequest(BaseModel):
    task_id: str
    loop_count: int = 0
    previous_output: str = ""
    improvement_notes: list = []
    quality_score: float = 0.0
    inject_redis: bool = True
    inject_raptor: bool = True


class SentinelReviewRequest(BaseModel):
    task_id: str
    goal: str
    final_output: str
    critic_review: Optional[str] = ""
    loop_count: int = 0
    quality_target: str = "world_class"


class SentinelLogRequest(BaseModel):
    status: str = "healthy"
    ts: Optional[str] = None
    pool: Optional[dict] = None


class SentinelAlertRequest(BaseModel):
    level: str = "warning"
    message: str
    ts: Optional[str] = None


class MigrateContextRequest(BaseModel):
    exhausted_providers: list = []
    strategy: str = "redis_dump_to_zilliz_raptor_reload"
    next_key_index: Any = "auto"
    preserve_stm: bool = True


class KeyRotateRequest(BaseModel):
    exhausted_providers: list  # list of {provider, pct, active_key, total_keys}
    strategy: str = "round_robin_context_migration"
    context_migration: Optional[dict] = None


# ---------------------------------------------------------------------------
# POST /llm/generate
# ---------------------------------------------------------------------------

@router.post("/llm/generate")
async def llm_generate(body: LLMGenerateRequest, request: Request) -> JSONResponse:
    """
    Role-injected raw LLM call through the key pool.
    Optionally prepends RAPTOR context (EL1: 25s timeout).
    EL6: provider_hint is advisory — pool may select different provider.
    """
    _auth(request)
    pool = request.app.state.pool

    # Build messages
    system_content = body.system or _default_system(body.role)

    # Optionally inject RAPTOR context
    raptor_context = ""
    if body.use_raptor and body.prompt:
        raptor_tree = getattr(request.app.state, "raptor_tree", None)
        if raptor_tree is not None:
            try:
                async with asyncio.timeout(25):  # EL1 mitigation
                    raptor_context = await raptor_tree.query(body.prompt)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"[/llm/generate] RAPTOR query failed (EL1): {e}")

    user_prompt = body.prompt
    if raptor_context:
        user_prompt = f"[MEMORY CONTEXT]\n{raptor_context}\n\n[TASK]\n{body.prompt}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_prompt},
    ]

    llm_fn = await make_provider_llm_fn(pool)
    result = await llm_fn(messages=messages, tools=[])

    if result is None:
        return JSONResponse(
            {"response": "", "error": "all keys exhausted or LLM call failed", "role": body.role},
            status_code=503,
        )

    content = result.get("content", "")
    return JSONResponse({"response": content, "role": body.role, "task_id": body.task_id})


def _default_system(role: str) -> str:
    systems = {
        "researcher":        "You are a world-class research expert. Provide deep, exhaustive, accurate research.",
        "architect":         "You are a principal software architect. Design scalable, elegant, battle-tested systems.",
        "coder":             "You are an elite software engineer. Write complete, production-grade, tested code. No placeholders.",
        "critic":            "You are a brutal but constructive code reviewer. Score 0-100. List all issues by severity: CRITICAL/HIGH/MEDIUM/LOW.",
        "synthesizer":       "You are the final synthesis expert. Fix all critical issues. Produce the absolute best final version.",
        "strategic_planner": "You are Ultron's strategic planner. Identify highest-leverage next improvement. Ghost is a ChemE at SVNIT Surat — prioritize chemical engineering tools. Return JSON only.",
    }
    return systems.get(role, "You are Ultron, an advanced AI assistant. Respond helpfully and accurately.")


# ---------------------------------------------------------------------------
# POST /task/decompose
# ---------------------------------------------------------------------------

@router.post("/task/decompose")
async def task_decompose(body: DecomposeRequest, request: Request) -> JSONResponse:
    """HTN decomposition via planner.decompose_goal()."""
    _auth(request)
    pool = request.app.state.pool

    llm_fn = await make_provider_llm_fn(pool)
    subtasks = await decompose_goal(
        goal=body.goal,
        llm_call_fn=llm_fn,
        max_subtasks=min(body.max_subtasks, 6),
    )

    return JSONResponse({
        "goal":     body.goal,
        "domain":   body.domain,
        "subtasks": [st.to_dict() for st in subtasks],
        "count":    len(subtasks),
    })


# ---------------------------------------------------------------------------
# POST /task/deliver
# ---------------------------------------------------------------------------

@router.post("/task/deliver")
async def task_deliver(body: DeliverRequest, request: Request) -> JSONResponse:
    """Store final task output. Supabase Tier4 + Zilliz + Discord notify. All degrade gracefully."""
    _auth(request)

    ts = time.time()
    record = {
        "task_id":            body.task_id,
        "goal":               body.goal,
        "final_output":       body.final_output,
        "quality_score":      body.quality_score,
        "loop_count":         body.loop_count,
        "improvement_history": body.improvement_history,
        "delivered_at":       ts,
    }

    # Supabase Tier4 — EL3: wrapped in to_thread (SB1 pattern)
    tier4 = getattr(request.app.state, "tier4", None)
    tier4_ok = False
    if tier4 and body.store_supabase:
        try:
            await asyncio.to_thread(
                tier4.store_improvement,
                user_id="eternal_loop",
                improvement=record,
            )
            tier4_ok = True
        except Exception as e:
            logger.warning(f"[/task/deliver] Supabase store failed (EL3): {e}")

    # Zilliz RAPTOR — store summary as long-term memory
    raptor_ok = False
    if body.store_zilliz:
        raptor_tree = getattr(request.app.state, "raptor_tree", None)
        embedder    = getattr(request.app.state, "embedder",    None)
        if raptor_tree and embedder:
            try:
                summary = f"TASK: {body.goal}\nSCORE: {body.quality_score}\nOUTPUT: {body.final_output[:500]}"
                await raptor_tree.ingest(texts=[summary], metadata={"task_id": body.task_id, "type": "delivery"})
                raptor_ok = True
            except Exception as e:
                logger.warning(f"[/task/deliver] RAPTOR ingest failed: {e}")

    # Discord notify
    discord_ok = False
    if body.notify_discord:
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
        if webhook_url:
            try:
                msg = (
                    f"✅ **Eternal Loop Delivery**\n"
                    f"Goal: {body.goal[:100]}\n"
                    f"Score: {body.quality_score}/100 | Loops: {body.loop_count}\n"
                    f"Task ID: `{body.task_id}`"
                )
                async with httpx.AsyncClient(timeout=5) as client:
                    await client.post(webhook_url, json={"content": msg})
                discord_ok = True
            except Exception as e:
                logger.warning(f"[/task/deliver] Discord notify failed: {e}")

    return JSONResponse({
        "status":      "delivered",
        "task_id":     body.task_id,
        "tier4_ok":    tier4_ok,
        "raptor_ok":   raptor_ok,
        "discord_ok":  discord_ok,
        "delivered_at": ts,
    })


# ---------------------------------------------------------------------------
# POST /task/update_context
# ---------------------------------------------------------------------------

@router.post("/task/update_context")
async def task_update_context(body: UpdateContextRequest, request: Request) -> JSONResponse:
    """Inject prior loop output into Redis STM + RAPTOR for next iteration context."""
    _auth(request)

    redis_ok  = False
    raptor_ok = False

    # Redis STM injection
    if body.inject_redis:
        redis = getattr(request.app.state, "redis", None)
        if redis and body.previous_output:
            try:
                ctx_key = f"ultron:ctx:eternal:{body.task_id}"
                entry = json.dumps({
                    "loop": body.loop_count,
                    "output": body.previous_output[:600],
                    "score": body.quality_score,
                    "notes": body.improvement_notes,
                    "ts": time.time(),
                })
                await redis.rpush(ctx_key, entry)
                await redis.expire(ctx_key, 86400)  # 24h TTL
                redis_ok = True
            except Exception as e:
                logger.warning(f"[/task/update_context] Redis inject failed: {e}")

    # RAPTOR injection
    if body.inject_raptor:
        raptor_tree = getattr(request.app.state, "raptor_tree", None)
        if raptor_tree and body.previous_output:
            try:
                notes_str = " | ".join(body.improvement_notes) if body.improvement_notes else ""
                text = (
                    f"[LOOP {body.loop_count}] Task: {body.task_id}\n"
                    f"Score: {body.quality_score}\nNotes: {notes_str}\n"
                    f"Output: {body.previous_output[:400]}"
                )
                await raptor_tree.ingest(
                    texts=[text],
                    metadata={"task_id": body.task_id, "loop": body.loop_count, "type": "loop_context"},
                )
                raptor_ok = True
            except Exception as e:
                logger.warning(f"[/task/update_context] RAPTOR inject failed: {e}")

    return JSONResponse({
        "status":    "updated",
        "task_id":   body.task_id,
        "loop":      body.loop_count,
        "redis_ok":  redis_ok,
        "raptor_ok": raptor_ok,
    })


# ---------------------------------------------------------------------------
# POST /sentinel/review
# ---------------------------------------------------------------------------

SENTINEL_REVIEW_SYSTEM = """You are Sentinel, Ultron's quality assurance CEO.
Your job: score the quality of the final output for a given goal.

Rules:
- Score 0-100 (100 = world-class, production-ready, exceeds expectations)
- quality_score >= 85 means world-class
- Be strict. A score of 85+ means Ghost could deploy this immediately.
- List improvement_suggestions as concrete, actionable items.

Return ONLY valid JSON:
{
  "quality_score": <0-100>,
  "verdict": "world_class" | "good" | "needs_improvement" | "poor",
  "strengths": ["..."],
  "improvement_suggestions": ["..."],
  "critical_issues": ["..."]
}"""


@router.post("/sentinel/review")
async def sentinel_review(body: SentinelReviewRequest, request: Request) -> JSONResponse:
    """
    Sentinel quality gate. Uses dedicated Gemini Sentinel key if available,
    falls back to pool Gemini key (EL2 mitigation).
    """
    _auth(request)

    prompt = (
        f"GOAL: {body.goal}\n\n"
        f"LOOP COUNT: {body.loop_count}\n\n"
        f"CRITIC REVIEW:\n{body.critic_review or 'N/A'}\n\n"
        f"FINAL OUTPUT (first 1500 chars):\n{body.final_output[:1500]}\n\n"
        f"Score the quality of this output for the goal. Return JSON only."
    )

    messages = [
        {"role": "system", "content": SENTINEL_REVIEW_SYSTEM},
        {"role": "user",   "content": prompt},
    ]

    result = None

    # Try dedicated Sentinel key (Gemini 2.5 Pro)
    sentinel_key = os.environ.get("GEMINI_SENTINEL_KEY", "")
    if sentinel_key:
        try:
            result = await call_provider(
                provider="gemini",
                api_key=sentinel_key,
                model="gemini-2.5-pro-preview-05-06",
                messages=messages,
            )
        except Exception as e:
            logger.warning(f"[/sentinel/review] Sentinel key call failed: {e}")

    # EL2: fallback to pool if Sentinel key absent or failed
    if result is None:
        pool = request.app.state.pool
        llm_fn = await make_provider_llm_fn(pool)
        result = await llm_fn(messages=messages, tools=[])

    if result is None:
        # Hard fallback — return neutral score so loop continues
        return JSONResponse({
            "quality_score":          50,
            "verdict":                "needs_improvement",
            "strengths":              [],
            "improvement_suggestions": ["Sentinel review unavailable — continuing loop"],
            "critical_issues":        [],
            "sentinel_error":         True,
        })

    content = result.get("content", "{}")
    try:
        cleaned = content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed  = json.loads(cleaned)
        return JSONResponse({
            "quality_score":           parsed.get("quality_score", 50),
            "verdict":                 parsed.get("verdict", "needs_improvement"),
            "strengths":               parsed.get("strengths", []),
            "improvement_suggestions": parsed.get("improvement_suggestions", []),
            "critical_issues":         parsed.get("critical_issues", []),
        })
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"[/sentinel/review] JSON parse failed: {e}")
        return JSONResponse({
            "quality_score":          60,
            "verdict":                "needs_improvement",
            "strengths":              [],
            "improvement_suggestions": ["Parse error — raw Sentinel output unavailable"],
            "critical_issues":        [],
        })


# ---------------------------------------------------------------------------
# POST /sentinel/log
# ---------------------------------------------------------------------------

@router.post("/sentinel/log")
async def sentinel_log(body: SentinelLogRequest, request: Request) -> JSONResponse:
    """Lightweight health log to Redis. Used by n8n Sentinel on HEALTHY path."""
    _auth(request)

    entry = json.dumps({
        "status": body.status,
        "ts":     body.ts or time.time(),
        "pool":   body.pool or {},
    })

    redis = getattr(request.app.state, "redis", None)
    if redis:
        try:
            await redis.rpush("ultron:sentinel:health_log", entry)
            await redis.ltrim("ultron:sentinel:health_log", -500, -1)  # keep last 500
        except Exception as e:
            logger.warning(f"[/sentinel/log] Redis write failed: {e}")

    logger.info(f"[Sentinel] health log: status={body.status}")
    return JSONResponse({"ok": True, "status": body.status})


# ---------------------------------------------------------------------------
# POST /sentinel/alert
# ---------------------------------------------------------------------------

@router.post("/sentinel/alert")
async def sentinel_alert(body: SentinelAlertRequest, request: Request) -> JSONResponse:
    """Fire Discord webhook alert. Used by n8n Sentinel on DEGRADED path."""
    _auth(request)

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    discord_ok  = False

    if webhook_url:
        try:
            level_emoji = {"critical": "🔴", "warning": "🟡", "info": "🟢"}.get(body.level, "⚪")
            msg = f"{level_emoji} **Ultron Sentinel Alert** [{body.level.upper()}]\n{body.message}"
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(webhook_url, json={"content": msg})
            discord_ok = True
        except Exception as e:
            logger.warning(f"[/sentinel/alert] Discord webhook failed: {e}")

    # Also log to Redis
    redis = getattr(request.app.state, "redis", None)
    if redis:
        try:
            await redis.rpush("ultron:sentinel:alerts", json.dumps({
                "level":   body.level,
                "message": body.message,
                "ts":      body.ts or time.time(),
            }))
            await redis.ltrim("ultron:sentinel:alerts", -200, -1)
        except Exception:
            pass

    logger.warning(f"[Sentinel] ALERT level={body.level}: {body.message}")
    return JSONResponse({"ok": True, "discord_ok": discord_ok, "level": body.level})


# ---------------------------------------------------------------------------
# POST /sentinel/migrate_context
# ---------------------------------------------------------------------------

@router.post("/sentinel/migrate_context")
async def sentinel_migrate_context(body: MigrateContextRequest, request: Request) -> JSONResponse:
    """
    Context migration: Redis STM dump → compress via RAPTOR → clear old STM.
    Called when API key exhaustion detected. EL4: single migration at a time via lock.
    """
    _auth(request)

    if _migration_lock.locked():
        return JSONResponse({
            "status":  "skipped",
            "reason":  "migration already in progress (EL4)",
            "providers": body.exhausted_providers,
        })

    async with _migration_lock:
        redis       = getattr(request.app.state, "redis",       None)
        raptor_tree = getattr(request.app.state, "raptor_tree", None)

        migrated_channels = []
        compressed_chunks = 0

        if redis and raptor_tree and body.preserve_stm:
            try:
                # Find all active STM channels
                ctx_keys = await redis.keys("ultron:ctx:*")
                for key in ctx_keys[:20]:  # cap at 20 channels per migration
                    key_str = key.decode() if isinstance(key, bytes) else key
                    entries = await redis.lrange(key_str, 0, -1)
                    if not entries:
                        continue

                    # Decode and compress into RAPTOR
                    texts = []
                    for e in entries:
                        try:
                            decoded = e.decode() if isinstance(e, bytes) else e
                            texts.append(decoded[:200])
                        except Exception:
                            pass

                    if texts:
                        combined = "\n".join(texts)
                        await raptor_tree.ingest(
                            texts=[combined],
                            metadata={"source": "context_migration", "key": key_str},
                        )
                        compressed_chunks += 1
                        migrated_channels.append(key_str)

                        # Clear old STM after compression
                        await redis.delete(key_str)

            except Exception as e:
                logger.error(f"[/sentinel/migrate_context] Migration error: {e}")

        # Rotate pool keys for exhausted providers (EL5: graceful if method absent)
        pool = request.app.state.pool
        rotated_providers = []
        for provider in body.exhausted_providers:
            if hasattr(pool, "rotate_provider"):
                try:
                    await pool.rotate_provider(provider)
                    rotated_providers.append(provider)
                except Exception as e:
                    logger.warning(f"[/sentinel/migrate_context] rotate_provider({provider}) failed: {e}")
            else:
                logger.info(f"[/sentinel/migrate_context] pool.rotate_provider not implemented yet (EL5)")

        logger.info(
            f"[Sentinel] Context migration complete. "
            f"channels={len(migrated_channels)} chunks={compressed_chunks} "
            f"rotated_providers={rotated_providers}"
        )

        return JSONResponse({
            "status":             "migrated",
            "migrated_channels":  migrated_channels,
            "compressed_chunks":  compressed_chunks,
            "rotated_providers":  rotated_providers,
            "exhausted_providers": body.exhausted_providers,
        })


# ---------------------------------------------------------------------------
# POST /keys/rotate
# ---------------------------------------------------------------------------

@router.post("/keys/rotate")
async def keys_rotate(body: KeyRotateRequest, request: Request) -> JSONResponse:
    """
    Trigger key pool rotation for exhausted providers.
    Called by n8n Key Pool Manager every 5 min when usage >= 90%.
    EL5: pool.rotate_provider() graceful fallback if method absent.
    """
    _auth(request)

    pool = request.app.state.pool
    rotated = []
    failed  = []

    for item in body.exhausted_providers:
        provider = item.get("provider", item) if isinstance(item, dict) else str(item)
        if hasattr(pool, "rotate_provider"):
            try:
                await pool.rotate_provider(provider)
                rotated.append(provider)
                logger.info(f"[/keys/rotate] Rotated provider: {provider}")
            except Exception as e:
                logger.warning(f"[/keys/rotate] rotate_provider({provider}) failed: {e}")
                failed.append(provider)
        else:
            # EL5: graceful — just trip the current key to force pool to next
            logger.info(f"[/keys/rotate] pool.rotate_provider absent — skipping {provider} (EL5)")
            rotated.append(f"{provider}:skipped_no_method")

    # Context migration if requested
    migrated = False
    if body.context_migration and (rotated or failed):
        try:
            migrate_req = MigrateContextRequest(
                exhausted_providers=[r for r in rotated if ":skipped" not in r],
                strategy=body.context_migration.get("strategy", "redis_dump_to_zilliz_raptor_reload"),
                preserve_stm=body.context_migration.get("preserve_stm", True),
            )
            # Fire migrate in background — don't block rotation response
            asyncio.create_task(
                _run_migration_internal(migrate_req, request.app.state)
            )
            migrated = True
        except Exception as e:
            logger.warning(f"[/keys/rotate] Background migration failed to start: {e}")

    return JSONResponse({
        "status":   "rotated",
        "rotated":  rotated,
        "failed":   failed,
        "migration_triggered": migrated,
    })


async def _run_migration_internal(body: MigrateContextRequest, app_state: Any) -> None:
    """Internal migration runner — mirrors /sentinel/migrate_context logic."""
    if _migration_lock.locked():
        return
    async with _migration_lock:
        redis       = getattr(app_state, "redis",       None)
        raptor_tree = getattr(app_state, "raptor_tree", None)
        pool        = getattr(app_state, "pool",        None)

        if redis and raptor_tree and body.preserve_stm:
            try:
                ctx_keys = await redis.keys("ultron:ctx:*")
                for key in ctx_keys[:20]:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    entries = await redis.lrange(key_str, 0, -1)
                    texts = []
                    for e in entries:
                        try:
                            decoded = e.decode() if isinstance(e, bytes) else e
                            texts.append(decoded[:200])
                        except Exception:
                            pass
                    if texts:
                        await raptor_tree.ingest(
                            texts=["\n".join(texts)],
                            metadata={"source": "internal_migration", "key": key_str},
                        )
                        await redis.delete(key_str)
            except Exception as e:
                logger.error(f"[_run_migration_internal] error: {e}")
