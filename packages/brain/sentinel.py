"""
packages/brain/sentinel.py

Ultron V4 — Sentinel God Layer
===============================
Authority: Highest. Never answers user queries. Only watches, decides, writes, repairs.

Responsibilities:
  - KV routing table read/write (CF Worker routes based on this)
  - Space health checks (<50ms, every request path)
  - Failure detection → promote backup → write incident to Notion → Discord DM Ghost
  - Weekly audit cron (triggered by GH Actions): reads logs → Gemini 1M context → Notion page
  - /sentinel/event handler integration: receives structured events from main.py

Design:
  - Dedicated Gemini 2.5 Pro key (GEMINI_SENTINEL_KEY). NEVER shares with general pool.
  - All CF KV ops via REST API (no SDK dependency).
  - Notion writes via direct REST API (Notion-Version: 2022-06-28).
  - Discord DM to Ghost on any critical event.
  - All methods are async. Sentinel is instantiated once in main.py lifespan.

Future bug risks (pre-registered):
  S1 [HIGH]  Gemini 2.5 Pro rate limit: 2 RPM on free tier. Weekly audit = 1 call.
             But concurrent failure events (burst) can cause 429.
             Fix: asyncio.Semaphore(1) on _call_sentinel(). Queues instead of drops.

  S2 [HIGH]  CF KV _kv_put with stale routing table: if two Sentinel instances (multi-worker
             M1 scenario) both detect failure simultaneously, both promote backup.
             Second write is safe (idempotent) but both fire Discord DMs.
             Fix: use CF KV conditional write (If-Match ETag) when available.

  S3 [MED]   Notion REST write requires NOTION_SENTINEL_PAGE_ID and NOTION_TOKEN env vars.
             If missing, incident write silently skips. Sentinel still DMs Ghost.
             Fix: log loud warning at startup if Notion vars unset.

  S4 [MED]   Discord DM: if bot token revoked, DM fails silently. Sentinel still writes Notion.
             Fix: add fallback webhook URL (SENTINEL_WEBHOOK_URL) as second DM channel.

  S5 [LOW]   weekly_audit() fetches Supabase logs up to 800k chars. Gemini 1M context
             can handle but API timeout=120s may be insufficient for large log sets.
             Fix: increase timeout to 240s for weekly audit call specifically.

Tool calls used writing this file:
    Github:get_file_contents x1 (ultron-v3/packages/brain/sentinel.py — reference patterns)
    Github:get_file_contents x1 (ultron-v4/packages/brain/main.py — confirmed app.state shape)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

import httpx

log = logging.getLogger("sentinel")

SENTINEL_MODEL    = "gemini-2.5-pro-preview-05-06"
ROUTING_TABLE_KEY = "ultron:routing:v4"

# Notion API constants
NOTION_API_BASE    = "https://api.notion.com/v1"
NOTION_API_VERSION = "2022-06-28"


class Sentinel:
    """
    God Layer. Non-technical CTO. Watches everything. Controls routing.
    Detects failures. Writes Notion. DMs Ghost. Zero user-facing latency.

    Instantiated in main.py lifespan if GEMINI_SENTINEL_KEY is set.
    """

    def __init__(
        self,
        sentinel_key: str,
        *,
        cf_account_id: str = "",
        cf_namespace_id: str = "",
        cf_kv_token: str = "",
        hf_primary_url: str = "",
        hf_backup_url: str = "",
        discord_bot_token: str = "",
        discord_ghost_uid: str = "",
        notion_token: str = "",
        notion_incident_page_id: str = "",
        supabase_url: str = "",
        supabase_key: str = "",
    ) -> None:
        self._key                 = sentinel_key
        self._cf_account_id       = cf_account_id
        self._cf_namespace_id     = cf_namespace_id
        self._cf_kv_token         = cf_kv_token
        self._hf_primary_url      = hf_primary_url
        self._hf_backup_url       = hf_backup_url
        self._discord_bot_token   = discord_bot_token
        self._discord_ghost_uid   = discord_ghost_uid
        self._notion_token        = notion_token
        self._notion_incident_pid = notion_incident_page_id
        self._supabase_url        = supabase_url
        self._supabase_key        = supabase_key
        self._gemini_lock         = asyncio.Semaphore(1)  # S1 guard: 1 Gemini call at a time

    # ──────────────────────────────────────────────────────────────────────
    # Gemini 2.5 Pro — dedicated call
    # ──────────────────────────────────────────────────────────────────────

    async def _call_sentinel(
        self,
        prompt: str,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> str:
        """Call Gemini 2.5 Pro via sentinel key. Rate-limited by asyncio.Semaphore(1)."""
        async with self._gemini_lock:  # S1 guard
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{SENTINEL_MODEL}:generateContent"
            )
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.3,
                },
                "systemInstruction": {
                    "parts": [{
                        "text": (
                            "You are Sentinel, the God-layer AI of the Ultron system. "
                            "You are the non-technical CTO. You never answer user queries. "
                            "You only analyze, decide, and write structured reports. "
                            "Be concise, precise, and technical. Use markdown headers."
                        )
                    }]
                },
            }
            try:
                async with httpx.AsyncClient(timeout=timeout) as c:
                    r = await c.post(url, params={"key": self._key}, json=payload)
                    r.raise_for_status()
                    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            except httpx.HTTPStatusError as e:
                log.error(f"[Sentinel] Gemini API error: {e.response.status_code} {e.response.text[:200]}")
                raise
            except Exception as e:
                log.error(f"[Sentinel] Gemini call failed: {e}")
                raise

    # ──────────────────────────────────────────────────────────────────────
    # Cloudflare KV — routing table
    # ──────────────────────────────────────────────────────────────────────

    def _kv_base(self) -> str:
        return (
            f"https://api.cloudflare.com/client/v4/accounts/{self._cf_account_id}"
            f"/storage/kv/namespaces/{self._cf_namespace_id}/values"
        )

    def _kv_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._cf_kv_token}"}

    async def _kv_get(self, key: str) -> Optional[str]:
        if not self._cf_kv_token:
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{self._kv_base()}/{key}", headers=self._kv_headers())
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.text
        except Exception as e:
            log.warning(f"[Sentinel] KV get failed key={key}: {e}")
            return None

    async def _kv_put(self, key: str, value: str) -> bool:
        if not self._cf_kv_token:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.put(
                    f"{self._kv_base()}/{key}",
                    headers=self._kv_headers(),
                    content=value,
                )
                r.raise_for_status()
                return True
        except Exception as e:
            log.error(f"[Sentinel] KV put failed key={key}: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Routing table
    # ──────────────────────────────────────────────────────────────────────

    async def get_routing_table(self) -> dict:
        raw = await self._kv_get(ROUTING_TABLE_KEY)
        if not raw:
            return {
                "primary":    self._hf_primary_url,
                "backup":     self._hf_backup_url,
                "updated_at": str(time.time()),
                "version":    "v4",
            }
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            log.error("[Sentinel] Routing table corrupt JSON — using defaults")
            return {"primary": self._hf_primary_url, "backup": self._hf_backup_url}

    async def set_primary_space(self, url: str) -> bool:
        table = await self.get_routing_table()
        table["primary"]    = url
        table["updated_at"] = str(time.time())
        ok = await self._kv_put(ROUTING_TABLE_KEY, json.dumps(table))
        if ok:
            log.info(f"[Sentinel] Routing updated → primary={url}")
        return ok

    async def check_space_health(self, url: str, timeout: float = 5.0) -> bool:
        """Fast health check. Called on every request path — must be <50ms in p99."""
        if not url:
            return False
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.get(f"{url}/health")
                return r.status_code == 200
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Failure handling
    # ──────────────────────────────────────────────────────────────────────

    async def handle_space_failure(
        self,
        failed_url: str,
        backup_url: str,
        error: str,
    ) -> None:
        """
        Called when primary Space fails health check.
        Flow: promote backup → Gemini analysis → Notion incident page → Discord DM.
        """
        log.critical(f"[Sentinel] Space failure: failed={failed_url} promoting={backup_url}")

        # 1. Promote backup to primary
        await self.set_primary_space(backup_url)

        # 2. Gemini incident analysis (best-effort)
        analysis = ""
        try:
            analysis = await self._call_sentinel(
                f"## Ultron Space Failure Incident\n\n"
                f"Failed Space: {failed_url}\n"
                f"Backup Promoted: {backup_url}\n"
                f"Error: {error}\n\n"
                f"Write structured incident report: Summary | Root Cause Hypothesis | "
                f"Impact | Recovery Steps | Prevention. Max 300 words.",
                max_tokens=512,
            )
        except Exception as e:
            analysis = f"Gemini analysis unavailable: {e}"

        # 3. Write Notion incident page
        await self._write_notion_incident(
            title=f"[INCIDENT] Space Failure — {time.strftime('%Y-%m-%d %H:%M UTC')}",
            content=(
                f"**Failed:** {failed_url}\n"
                f"**Promoted:** {backup_url}\n"
                f"**Error:** {error}\n\n"
                f"## Sentinel Analysis\n\n{analysis}"
            ),
        )

        # 4. Discord DM Ghost
        msg = (
            f"🚨 **SENTINEL INCIDENT**\n"
            f"`{failed_url}` FAILED.\n"
            f"Promoted `{backup_url}` to primary.\n\n"
            f"**Analysis:**\n{analysis[:1200]}"
        )
        await self._discord_dm(msg)

    # ──────────────────────────────────────────────────────────────────────
    # Weekly audit
    # ──────────────────────────────────────────────────────────────────────

    async def weekly_audit(self) -> str:
        """
        Full week log read → Gemini 1M context analysis → Notion page → Discord DM.
        Triggered by GH Actions cron (Sunday 23:59).
        """
        log.info("[Sentinel] Weekly audit starting...")
        logs_text = "(Supabase not configured)"

        if self._supabase_url and self._supabase_key:
            try:
                since = time.time() - 7 * 86_400
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.get(
                        f"{self._supabase_url}/rest/v1/logs",
                        params={
                            "ts":     f"gte.{since}",
                            "select": "ts,component,level,msg,extra",
                            "order":  "ts.asc",
                            "limit":  "10000",
                        },
                        headers={
                            "apikey":        self._supabase_key,
                            "Authorization": f"Bearer {self._supabase_key}",
                        },
                    )
                    logs_text = json.dumps(r.json(), separators=(",", ":"))[:800_000]
            except Exception as e:
                logs_text = f"Log fetch error: {e}"

        try:
            report = await self._call_sentinel(
                f"## Weekly Audit — Ultron V4 System\n\n"
                f"Date: {time.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                f"LOGS (7 days):\n{logs_text}\n\n"
                f"Write structured report:\n"
                f"## Component Health\n## LLM Stats\n## Memory Hit Rates\n"
                f"## Top 5 Errors + Root Cause Hypothesis\n"
                f"## Key Rotation Events\n## Sentinel Recommendations for Next Version\n\n"
                f"Max 1000 words.",
                max_tokens=1800,
                timeout=240.0,  # S5: longer timeout for large log sets
            )
        except Exception as e:
            report = f"Weekly audit Gemini call failed: {e}"

        # Write to Notion
        await self._write_notion_incident(
            title=f"📊 Sentinel Weekly Audit — {time.strftime('%Y-%m-%d')}",
            content=report,
        )

        # DM Ghost
        dm_msg = f"📊 **SENTINEL WEEKLY AUDIT**\n\n{report[:1800]}"
        await self._discord_dm(dm_msg)

        log.info("[Sentinel] Weekly audit complete.")
        return report

    # ──────────────────────────────────────────────────────────────────────
    # Event handler (called from main.py /sentinel/event)
    # ──────────────────────────────────────────────────────────────────────

    async def handle_event(self, event_type: str, payload: dict) -> dict:
        """
        Central event dispatcher. Called by main.py /sentinel/event endpoint.

        event_type:
          "space_failure"     → handle_space_failure()
          "routing_override"  → set_primary_space(url)
          "health_check"      → check_space_health(url)
          "weekly_audit"      → weekly_audit()
          "project_plan"      → generate_project_plan(brief)
        """
        log.info(f"[Sentinel] handle_event type={event_type} payload_keys={list(payload.keys())}")

        if event_type == "space_failure":
            await self.handle_space_failure(
                failed_url=payload.get("failed_url", ""),
                backup_url=payload.get("backup_url", self._hf_backup_url),
                error=payload.get("error", "unknown"),
            )
            return {"status": "failover_complete"}

        elif event_type == "routing_override":
            url = payload.get("url", "")
            ok  = await self.set_primary_space(url)
            return {"status": "ok" if ok else "kv_write_failed", "primary": url}

        elif event_type == "health_check":
            url    = payload.get("url", self._hf_primary_url)
            result = await self.check_space_health(url)
            return {"url": url, "healthy": result}

        elif event_type == "weekly_audit":
            report = await self.weekly_audit()
            return {"status": "complete", "report_preview": report[:200]}

        elif event_type == "project_plan":
            brief  = payload.get("brief", "")
            plan   = await self.generate_project_plan(brief)
            return {"status": "complete", "plan": plan}

        else:
            log.warning(f"[Sentinel] Unknown event_type={event_type}")
            return {"status": "unknown_event_type"}

    # ──────────────────────────────────────────────────────────────────────
    # Project plan generation
    # ──────────────────────────────────────────────────────────────────────

    async def generate_project_plan(self, brief: str) -> str:
        return await self._call_sentinel(
            f"## Project Operational Plan\n\n"
            f"Brief: {brief}\n\n"
            f"Generate: DevOps strategy | Memory architecture | MOA config | "
            f"Tool assignments | Success criteria | Risk assessment. Max 600 words.",
            max_tokens=1200,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Notion write
    # ──────────────────────────────────────────────────────────────────────

    async def _write_notion_incident(
        self,
        title: str,
        content: str,
    ) -> None:
        """
        Create a new child page under NOTION_SENTINEL_PAGE_ID.
        S3 guard: if notion creds not set, warn and skip.
        """
        if not self._notion_token or not self._notion_incident_pid:
            log.warning("[Sentinel] Notion creds unset — skipping incident write (S3)")
            return

        headers = {
            "Authorization":  f"Bearer {self._notion_token}",
            "Notion-Version": NOTION_API_VERSION,
            "Content-Type":   "application/json",
        }

        body = {
            "parent":     {"page_id": self._notion_incident_pid},
            "properties": {
                "title": {
                    "title": [{"type": "text", "text": {"content": title}}]
                }
            },
            "children": [
                {
                    "object": "block",
                    "type":   "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": content[:2000]}  # Notion block limit
                        }]
                    },
                }
            ],
        }

        # If content > 2000 chars, add a second paragraph block
        if len(content) > 2000:
            body["children"].append({
                "object": "block",
                "type":   "paragraph",
                "paragraph": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": content[2000:4000]}
                    }]
                },
            })

        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.post(
                    f"{NOTION_API_BASE}/pages",
                    headers=headers,
                    json=body,
                )
                r.raise_for_status()
                log.info(f"[Sentinel] Notion incident page created: {title}")
        except httpx.HTTPStatusError as e:
            log.error(
                f"[Sentinel] Notion write failed: {e.response.status_code} "
                f"{e.response.text[:200]}"
            )
        except Exception as e:
            log.error(f"[Sentinel] Notion write exception: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Discord DM
    # ──────────────────────────────────────────────────────────────────────

    async def _discord_dm(self, message: str) -> None:
        """DM Ghost via Discord bot. S4: silent fail if token revoked."""
        if not self._discord_bot_token or not self._discord_ghost_uid:
            log.warning("[Sentinel] Discord bot/UID not set — skipping DM (S4)")
            return

        headers = {
            "Authorization": f"Bot {self._discord_bot_token}",
            "Content-Type":  "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                dm = await c.post(
                    "https://discord.com/api/v10/users/@me/channels",
                    headers=headers,
                    json={"recipient_id": self._discord_ghost_uid},
                )
                if dm.status_code not in (200, 201):
                    log.error(f"[Sentinel] DM channel creation failed: {dm.status_code}")
                    return
                channel_id = dm.json()["id"]
                # Chunk into 1990-char Discord safe pieces
                for chunk in [message[i:i + 1990] for i in range(0, len(message), 1990)]:
                    await c.post(
                        f"https://discord.com/api/v10/channels/{channel_id}/messages",
                        headers=headers,
                        json={"content": chunk},
                    )
        except Exception as e:
            log.error(f"[Sentinel] Discord DM failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Factory — called from main.py lifespan
# ──────────────────────────────────────────────────────────────────────────────

def build_sentinel(settings: Any) -> Optional["Sentinel"]:
    """
    Build Sentinel from settings. Returns None if GEMINI_SENTINEL_KEY not set.
    Called during FastAPI lifespan — non-fatal if Sentinel unavailable.
    """
    import os
    sentinel_key = (
        getattr(settings, "gemini_sentinel_key", "")
        or os.environ.get("GEMINI_SENTINEL_KEY", "")
    )
    if not sentinel_key:
        log.warning("[Sentinel] GEMINI_SENTINEL_KEY not set — Sentinel INACTIVE")
        return None

    return Sentinel(
        sentinel_key=sentinel_key,
        cf_account_id=getattr(settings, "cf_account_id", "") or os.environ.get("CF_ACCOUNT_ID", ""),
        cf_namespace_id=getattr(settings, "cf_kv_namespace_id", "") or os.environ.get("CF_KV_NAMESPACE_ID", ""),
        cf_kv_token=getattr(settings, "cf_kv_api_token", "") or os.environ.get("CF_KV_API_TOKEN", ""),
        hf_primary_url=os.environ.get("HF_PRIMARY_URL", "https://ghostdrive1-ultron1.hf.space"),
        hf_backup_url=os.environ.get("HF_BACKUP_URL", ""),
        discord_bot_token=getattr(settings, "discord_bot_token", "") or os.environ.get("DISCORD_BOT_TOKEN", ""),
        discord_ghost_uid=os.environ.get("DISCORD_GHOST_UID", ""),
        notion_token=os.environ.get("NOTION_TOKEN", ""),
        notion_incident_page_id=os.environ.get("NOTION_SENTINEL_PAGE_ID", ""),
        supabase_url=getattr(settings, "supabase_url", "") or os.environ.get("SUPABASE_URL", ""),
        supabase_key=getattr(settings, "supabase_key", "") or os.environ.get("SUPABASE_KEY", ""),
    )
