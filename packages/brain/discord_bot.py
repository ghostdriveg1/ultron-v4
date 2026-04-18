"""
packages/brain/discord_bot.py

Ultron V4 — Discord Interface Layer
=====================================
Thin bot that bridges Discord <-> FastAPI Brain (/infer endpoint).
Never calls TaskDispatcher or LLM directly — Brain is the only LLM surface.

Features (V4 over V3):
  - Per-channel Redis rolling context window (20 messages)
  - Prefix commands: !help !status !memory !council !clear !ping
  - Chunked message send (Discord 2000-char limit)
  - Typing indicator on all slow ops
  - Attachment handling: uploads file_url/filename to Brain
  - Rate-limit: 10 req/min per user (in-memory sliding window)
  - No internal block leaks: strip_internal_blocks applied on ALL responses
  - No fake behaviors: if Brain returns error, say so plainly
  - LifecycleEngine.ingest() called on every user message (v25)

V4 design rules:
  - Bot is stateless except for rate-limit counters and Redis context writes
  - All intelligence lives in Brain (FastAPI) — bot is a relay
  - ALLOWED_USERS is the ONLY auth surface — fail closed
  - typing() wraps ALL slow calls without exception
  - NEVER send === MEMORY GRAPH === or [COMPACTED HISTORY] blocks to user

Future bug risks (pre-registered):
  BOT1 [HIGH]   Redis context write fails silently -> context window empty -> B1/D1 fire
                Fix: log warning, continue without context (degrade gracefully)
  BOT2 [HIGH]   Discord rate-limit on bulk sends (5+ chunks in 1s) -> 429 from Discord API
                Fix: asyncio.sleep(0.5) between chunks if len(chunks) > 2
  BOT3 [MED]    Brain /health timeout on !status -> bot hangs under slow HF Space wake
                Fix: timeout=8s on health check, return "Brain waking..." on timeout
  BOT4 [MED]    ALLOWED_USERS env parsed at import -> adding user requires restart
                Fix: re-parse on each message (minimal perf cost, big ops win)
  BOT5 [LOW]    on_message fires for bot's own replies (if intents wrong) -> infinite loop
                Fix: if msg.author == bot.user: return — MUST be first check
  BOT6 [LOW]    Context window RPUSH/LTRIM non-atomic -> concurrent messages corrupt window
                Fix: use Redis pipeline() for atomic push+trim pair
  BOT7 [MED]    lifecycle.ingest() called per message but user_id != channel_id in lifecycle
                keys. If lifecycle.get_stm(user_id) called with channel_id from /memory/stm,
                returns empty list. Fix: pass user_id to ingest, not channel_id as proxy.
                (Tracked as CL5 in main.py — lifecycle stores cells per user_id)

Tool calls used this session (v25):
  Github:get_file_contents x1 (discord_bot.py current state + sha)
  Github:get_file_contents x1 (lifecycle.py interface — ingest signature)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from collections import defaultdict
from typing import Optional

import discord
import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — all from env, fail loud if critical vars missing
# ---------------------------------------------------------------------------

DISCORD_BOT_TOKEN: str = os.environ.get("DISCORD_BOT_TOKEN", "")
BRAIN_URL: str = os.environ.get("BRAIN_URL", "http://localhost:7860").rstrip("/")
INTERNAL_AUTH_TOKEN: str = os.environ.get("INTERNAL_AUTH_TOKEN", "")
GHOST_USER_ID: str = os.environ.get("DISCORD_GHOST_USER_ID", "")

# Parse allowed users fresh on each access to avoid restart-on-change (BOT4 mitigation)
def _get_allowed_users() -> set[str]:
    raw = os.environ.get("ALLOWED_DISCORD_USERS", GHOST_USER_ID)
    return {u.strip() for u in raw.split(",") if u.strip()}

# Rate limiting: 10 req/min per user (in-memory, resets on restart — acceptable for single-Space)
_rate_window: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_MAX = 10
RATE_LIMIT_WINDOW = 60.0  # seconds

# Per-channel context window: 20 messages max (Redis key: ultron:ctx:{channel_id})
CTX_KEY_PREFIX = "ultron:ctx:"
CTX_MAX_MESSAGES = 20
CTX_TTL = 7200  # 2 hours idle expiry

# Discord message limits
DISCORD_MAX_CHARS = 1990
CHUNK_DELAY = 0.5  # seconds between chunks if > 2 (BOT2 mitigation)

# Internal blocks that must NEVER reach Discord — mirrors task_dispatcher strip logic
_STRIP_PATTERNS = [
    re.compile(r"^=== MEMORY GRAPH ===.*?(?=^===|\Z)", re.MULTILINE | re.DOTALL),
    re.compile(r"^\[COMPACTED HISTORY SUMMARY\].*?(?=^\[|\Z)", re.MULTILINE | re.DOTALL),
    re.compile(r"^\[OBSERVATION\].*?(?=^\[|\Z)", re.MULTILINE | re.DOTALL),
    re.compile(r"^\[LOOP WARNING\].*$", re.MULTILINE),
    re.compile(r"^\[TOOL (ERROR|RESULT|OK)\].*$", re.MULTILINE),
]


def _strip(text: str) -> str:
    """Remove all internal orchestration markers before sending to Discord."""
    for pat in _STRIP_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


def _chunk(text: str, size: int = DISCORD_MAX_CHARS) -> list[str]:
    """Split text into Discord-safe chunks. Never empty."""
    if not text:
        return ["(empty response)"]
    return [text[i : i + size] for i in range(0, len(text), size)]


def _is_rate_limited(user_id: str) -> bool:
    """Return True if user has exceeded 10 req/min."""
    now = time.monotonic()
    window = _rate_window[user_id]
    # Evict old timestamps
    _rate_window[user_id] = [t for t in window if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_window[user_id]) >= RATE_LIMIT_MAX:
        return True
    _rate_window[user_id].append(now)
    return False


# ---------------------------------------------------------------------------
# Brain HTTP client
# ---------------------------------------------------------------------------

def _headers(user_id: str) -> dict[str, str]:
    return {
        "X-Auth-Token": INTERNAL_AUTH_TOKEN,
        "X-User-Id": user_id,
        "Content-Type": "application/json",
    }


async def _call_brain(
    path: str,
    payload: dict,
    user_id: str,
    timeout: float = 90.0,
) -> dict:
    """POST to Brain FastAPI. Returns parsed JSON or error dict."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{BRAIN_URL}{path}",
                headers=_headers(user_id),
                json=payload,
            )
        if r.status_code == 429:
            return {"error": "\u26a0\ufe0f Rate limited. Try again in a moment."}
        if r.status_code == 503:
            return {"error": "\u26a0\ufe0f All LLM keys exhausted. Try again later."}
        if r.status_code == 401:
            return {"error": "\u26d4 Auth failed. Check INTERNAL_AUTH_TOKEN."}
        if r.status_code not in (200, 201):
            return {"error": f"Brain {r.status_code}: {r.text[:200]}"}
        return r.json()
    except httpx.TimeoutException:
        return {"error": "\u23f1\ufe0f Brain timed out. HF Space may be waking up — try again in 30s."}
    except Exception as exc:
        logger.exception(f"[Bot] _call_brain {path} failed: {exc}")
        return {"error": f"Bot error: {str(exc)[:200]}"}


async def _get_health(user_id: str) -> dict:
    """GET /health — separate method, shorter timeout (BOT3 mitigation)."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{BRAIN_URL}/health", headers=_headers(user_id))
        if r.status_code == 200:
            return r.json()
        return {"error": f"Health {r.status_code}"}
    except httpx.TimeoutException:
        return {"error": "Brain waking up... try !status in 30s"}
    except Exception as exc:
        return {"error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# Redis context window helpers (optional — bot works without Redis)
# ---------------------------------------------------------------------------

async def _ctx_append(redis, channel_id: str, role: str, content: str) -> None:
    """Append message to per-channel context list. Atomic push+trim (BOT6 mitigation)."""
    if redis is None:
        return
    key = f"{CTX_KEY_PREFIX}{channel_id}"
    entry = f"{role}: {content[:500]}"
    try:
        pipe = redis.pipeline()
        pipe.rpush(key, entry)
        pipe.ltrim(key, -CTX_MAX_MESSAGES, -1)
        pipe.expire(key, CTX_TTL)
        await pipe.execute()
    except Exception as exc:
        logger.warning(f"[Bot] ctx_append failed (BOT1): {exc}")  # degrade gracefully


async def _ctx_get(redis, channel_id: str) -> str:
    """Return last N messages from channel context as formatted string."""
    if redis is None:
        return ""
    key = f"{CTX_KEY_PREFIX}{channel_id}"
    try:
        entries = await redis.lrange(key, 0, -1)
        return "\n".join(e.decode() if isinstance(e, bytes) else e for e in entries)
    except Exception as exc:
        logger.warning(f"[Bot] ctx_get failed (BOT1): {exc}")
        return ""


# ---------------------------------------------------------------------------
# Lifecycle ingest helper
# ---------------------------------------------------------------------------

async def _lifecycle_ingest(
    lifecycle,
    user_id: str,
    channel_id: str,
    text: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Fire lifecycle.ingest() as best-effort background call.
    Never raises — BOT7 mitigation: pass user_id (not channel_id) as the lifecycle key.
    """
    if lifecycle is None:
        return
    try:
        await lifecycle.ingest(
            user_id=user_id,
            channel_id=channel_id,
            raw_text=text,
            metadata=metadata or {},
        )
    except Exception as exc:
        logger.warning(f"[Bot] lifecycle.ingest failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def _cmd_help(msg: discord.Message) -> None:
    await msg.reply(
        "**Ultron V4**\n"
        "`!help` — this message\n"
        "`!status` — system health + key pool\n"
        "`!memory` — Tier 1 context summary\n"
        "`!council <brief>` — Council Mode (MOA multi-expert)\n"
        "`!clear` — wipe channel context window\n"
        "`!ping` — latency check"
    )


async def _cmd_status(msg: discord.Message, user_id: str) -> None:
    async with msg.channel.typing():
        data = await _get_health(user_id)
    if "error" in data:
        await msg.reply(f"\u26a0\ufe0f {data['error']}")
        return
    pool = data.get("pool", {})
    total = pool.get("total", 0)
    available = pool.get("available", 0)
    uptime = data.get("uptime_seconds", 0)
    uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m"
    lifecycle_status = "\u2705" if data.get("lifecycle_active") else "\u274c"
    lines = [
        f"**Ultron V4** | Uptime: {uptime_str} | Keys: {available}/{total} | Lifecycle: {lifecycle_status}",
    ]
    # Show per-provider pool summary if available
    providers = pool.get("providers", {})
    for provider, stats in providers.items():
        avail_icon = "\u2705" if stats.get("available", 0) > 0 else "\u274c"
        lines.append(f"  {avail_icon} `{provider}` — {stats.get('available', 0)}/{stats.get('total', 0)} keys")
    await msg.reply("\n".join(lines))


async def _cmd_memory(msg: discord.Message, user_id: str) -> None:
    async with msg.channel.typing():
        result = await _call_brain(
            "/infer",
            {
                "message": "Summarize the current Tier 1 memory context. Be concise.",
                "user_id": user_id,
                "channel_id": str(msg.channel.id),
            },
            user_id,
        )
    text = _strip(result.get("response", result.get("error", "No memory data.")))
    chunks = _chunk(text)
    for i, c in enumerate(chunks):
        if i > 0:
            await asyncio.sleep(CHUNK_DELAY)
        await msg.channel.send(c)


async def _cmd_council(msg: discord.Message, user_id: str, args: str) -> None:
    if not args:
        await msg.reply("Usage: `!council <project brief>`")
        return
    await msg.reply("\u26a1 Assembling council... (~30-60s)")
    async with msg.channel.typing():
        result = await _call_brain(
            "/council",
            {"project_brief": args, "domain": "general", "phase": "start"},
            user_id,
            timeout=120.0,
        )
    synthesis = _strip(result.get("synthesis", result.get("error", "Council failed.")))
    header = "**\u26a1 Council Report:**\n"
    chunks = _chunk(header + synthesis)
    for i, c in enumerate(chunks):
        if i > 0:
            await asyncio.sleep(CHUNK_DELAY)
        await msg.channel.send(c)


async def _cmd_clear(msg: discord.Message, redis, channel_id: str) -> None:
    if redis is not None:
        try:
            await redis.delete(f"{CTX_KEY_PREFIX}{channel_id}")
        except Exception as exc:
            logger.warning(f"[Bot] ctx clear failed: {exc}")
    await msg.reply("\U0001f5d1\ufe0f Channel context cleared.")


async def _cmd_ping(msg: discord.Message) -> None:
    latency_ms = round(msg._state._get_websocket(msg.guild).latency * 1000)
    await msg.reply(f"\U0001f3d3 Pong! Latency: {latency_ms}ms")


# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------

def build_bot(redis=None, lifecycle=None) -> discord.Client:
    """Build and return the Discord client.

    Args:
        redis: Optional async Redis client (e.g. upstash_redis.Redis or aioredis.Redis).
               If None, context window is disabled but bot still works.
        lifecycle: Optional LifecycleEngine instance. If set, ingest() is called on
                   every user message for STM/MTM/Foresight pipeline. (v25)
    """
    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    bot = discord.Client(intents=intents)

    @bot.event
    async def on_ready() -> None:
        logger.info(f"[Bot] {bot.user} online | Brain: {BRAIN_URL}")
        print(f"[Ultron V4] {bot.user} | Brain: {BRAIN_URL}")

    @bot.event
    async def on_message(msg: discord.Message) -> None:
        # BOT5: must be first check
        if msg.author == bot.user:
            return

        user_id = str(msg.author.id)

        # Auth — re-parse env each time (BOT4 mitigation)
        if user_id not in _get_allowed_users():
            return

        content = msg.content.strip()
        if not content and not msg.attachments:
            return

        # Rate limit
        if _is_rate_limited(user_id):
            await msg.reply("\u26a0\ufe0f Slow down — max 10 messages/minute.")
            return

        channel_id = str(msg.channel.id)

        # Command routing
        if content.startswith("!"):
            parts = content.split(None, 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if cmd == "!help":
                await _cmd_help(msg)
            elif cmd == "!status":
                await _cmd_status(msg, user_id)
            elif cmd == "!memory":
                await _cmd_memory(msg, user_id)
            elif cmd == "!council":
                await _cmd_council(msg, user_id, args)
            elif cmd == "!clear":
                await _cmd_clear(msg, redis, channel_id)
            elif cmd == "!ping":
                await _cmd_ping(msg)
            else:
                await msg.reply(f"Unknown command: `{cmd}`. Try `!help`.")
            return

        # Build context string from Redis window
        context = await _ctx_get(redis, channel_id)

        # Handle attachments: pass URL + filename to Brain
        attachment_info = ""
        if msg.attachments:
            att = msg.attachments[0]  # handle first attachment
            attachment_info = f"\n[ATTACHMENT: {att.filename} | {att.url}]"

        full_message = content + attachment_info

        # Fire lifecycle.ingest() as non-blocking background task (v25)
        # BOT7: pass user_id as lifecycle key, channel_id for context grouping
        asyncio.create_task(
            _lifecycle_ingest(
                lifecycle,
                user_id=user_id,
                channel_id=channel_id,
                text=full_message,
                metadata={"source": "discord", "username": str(msg.author)},
            )
        )

        # Save user message to context window BEFORE call
        await _ctx_append(redis, channel_id, "user", full_message)

        # Dispatch to Brain
        async with msg.channel.typing():
            result = await _call_brain(
                "/infer",
                {
                    "message": full_message,
                    "user_id": user_id,
                    "channel_id": channel_id,
                    "context": context,
                },
                user_id,
            )

        if "error" in result:
            await msg.reply(result["error"])
            return

        response = _strip(result.get("response", ""))
        if not response:
            response = "(no response)"

        # Save bot response to context window
        await _ctx_append(redis, channel_id, "assistant", response[:500])

        # Also ingest bot response into lifecycle (for Foresight context)
        asyncio.create_task(
            _lifecycle_ingest(
                lifecycle,
                user_id=user_id,
                channel_id=channel_id,
                text=f"[ULTRON RESPONSE] {response[:500]}",
                metadata={"source": "discord_response"},
            )
        )

        # Send chunked
        chunks = _chunk(response)
        for i, chunk in enumerate(chunks):
            if i > 0 and len(chunks) > 2:
                await asyncio.sleep(CHUNK_DELAY)  # BOT2 mitigation
            if i == 0:
                await msg.reply(chunk)
            else:
                await msg.channel.send(chunk)

    return bot


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(redis=None, lifecycle=None) -> None:
    """Start the Discord bot. Called from main.py lifespan or standalone."""
    if not DISCORD_BOT_TOKEN:
        logger.error("[Bot] DISCORD_BOT_TOKEN not set — bot disabled")
        print("[ERROR] DISCORD_BOT_TOKEN not set", file=sys.stderr)
        return
    bot = build_bot(redis=redis, lifecycle=lifecycle)
    bot.run(DISCORD_BOT_TOKEN, log_handler=None)  # logging already configured
