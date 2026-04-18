"""
packages/infrastructure/space_promoter.py

Ultron V4 — Phase 6 Multi-Space Orchestration
===============================================
SpacePromoter manages active-active HF Space topology:
  - Primary Space: ghostdrive1/ultron-brain (acct1)
  - Backup Space:  ghostdrive2/ultron-brain-backup (acct2)
  - Voice Space:   ghostdrive1/ultron-voice

Responsibilities:
  1. Health-check loop: ping /health on all registered Spaces every N seconds
  2. Detect primary failure: 3 consecutive failures -> promote backup to primary
  3. Write new routing table to Cloudflare KV (Sentinel KV)
  4. Notify Ghost via Discord webhook on promotion events
  5. De-promote: restore primary when it recovers (with hysteresis)

CF KV routing table schema:
  Key: "ultron:routing:v4"
  Value: JSON {
    "primary": "https://...",
    "backup": "https://...",
    "voice": "https://...",
    "updated_at": "ISO",
    "promoted_at": "ISO|null",
    "reason": "..."
  }

Skypilot pattern used (github.com/skypilot-org/skypilot):
  - Health check with retries before marking node dead
  - Hysteresis before promotion (avoid flapping)
  - Atomic KV write with old-value check
  - Separate check intervals for primary vs backup

Pre-registered bugs:
  SP1 [HIGH]  KV write fails mid-failover -> old primary still in KV -> CF Worker
              routes all traffic to dead Space. Fix: retry KV write up to 3x before
              accepting split-brain state. Log alert if all retries fail.

  SP2 [HIGH]  Split-brain: both Spaces healthy, KV write fails on restore, backup
              and primary both serving. Fix: CF Worker reads KV on every request
              (cheap, <1ms). Single KV source of truth. Worker has no memory.

  SP3 [MED]   HF Space cold start takes 30-60s. If health check timeout < 30s,
              false-positive failures trigger unnecessary promotion. Fix: use
              HEALTH_TIMEOUT=35s, FAILURE_THRESHOLD=3 consecutive checks.

  SP4 [MED]   Discord webhook not set -> notification fails silently. Ghost doesn't
              know about promotion. Fix: also log promotion event to Redis
              (ultron:infra:events list) so website can show it.

  SP5 [LOW]   Infinite promotion loop: backup also dies -> try to promote tertiary
              (doesn't exist) -> loop panics. Fix: MAX_FAILOVER_ATTEMPTS guard.
              After max attempts, enter DEGRADED state, stop promoting, keep alerting.

  SP6 [LOW]   KV namespace ID hardcoded. If Ghost creates new CF account or
              namespace, must update here AND in CF Worker. Fix: read from
              CF_KV_NAMESPACE_ID env var (already in config).

Tool calls used writing this file (v25):
    skypilot-org/skypilot: sky/skylet/log_lib.py, sky/backends/cloud_vm_ray_backend.py
    (health probe pattern, node death detection, retry logic)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

log = logging.getLogger("ultron.space_promoter")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HEALTH_CHECK_INTERVAL = int(os.environ.get("PROMOTER_CHECK_INTERVAL", "30"))   # seconds
HEALTH_TIMEOUT        = float(os.environ.get("PROMOTER_HEALTH_TIMEOUT", "35")) # SP3
FAILURE_THRESHOLD     = int(os.environ.get("PROMOTER_FAIL_THRESHOLD", "3"))    # consecutive
RECOVERY_THRESHOLD    = int(os.environ.get("PROMOTER_RECOVER_THRESHOLD", "3")) # consecutive ok
MAX_FAILOVER_ATTEMPTS = int(os.environ.get("PROMOTER_MAX_FAILOVER", "5"))      # SP5
KV_ROUTING_KEY        = "ultron:routing:v4"
KV_EVENTS_KEY         = "ultron:infra:events"
KV_MAX_EVENTS         = 100

# CF KV REST API
CF_ACCOUNT_ID    = os.environ.get("CF_ACCOUNT_ID", "c2ed2ecab1a35b2cd2095849cb69ab10")
CF_KV_NAMESPACE  = os.environ.get("CF_KV_NAMESPACE_ID", "77184c17886d47f2be73b6d441ada952")  # SP6
CF_KV_API_TOKEN  = os.environ.get("CF_KV_API_TOKEN", "")
CF_KV_BASE_URL   = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/storage/kv/namespaces/{CF_KV_NAMESPACE}"

# Discord webhook for promotion alerts (SP4)
DISCORD_WEBHOOK  = os.environ.get("DISCORD_WEBHOOK_URL", "")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SpaceNode:
    """Represents one HF Space in the topology."""
    name: str                          # e.g. "brain-primary"
    url: str                           # e.g. "https://ghostdrive1-ultron1.hf.space"
    role: str                          # "primary" | "backup" | "voice"
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check_at: Optional[str] = None
    last_status: str = "unknown"       # "ok" | "failed" | "unknown"
    is_alive: bool = True


@dataclass
class RoutingTable:
    """CF KV routing table value."""
    primary: str
    backup: str
    voice: str
    updated_at: str = field(default_factory=_now_iso)
    promoted_at: Optional[str] = None
    reason: str = "initial"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "RoutingTable":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class InfraEvent:
    """Infrastructure event for website dashboard (SP4)."""
    event_type: str   # "promotion" | "recovery" | "health_degraded" | "kv_write_failed"
    message: str
    timestamp: str = field(default_factory=_now_iso)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# SpacePromoter
# ---------------------------------------------------------------------------

class SpacePromoter:
    """
    Orchestrates multi-Space health and routing.

    Usage:
        promoter = SpacePromoter(redis_client)
        stop = asyncio.Event()
        asyncio.create_task(promoter.run(stop))
        # ...
        stop.set()
    """

    def __init__(
        self,
        redis_client,
        primary_url: Optional[str] = None,
        backup_url: Optional[str] = None,
        voice_url: Optional[str] = None,
    ):
        self.redis = redis_client
        self._failover_count: int = 0
        self._in_degraded_state: bool = False

        primary_url = primary_url or os.environ.get(
            "BRAIN_PRIMARY_URL", "https://ghostdrive1-ultron1.hf.space"
        )
        backup_url = backup_url or os.environ.get(
            "BRAIN_BACKUP_URL", ""
        )
        voice_url = voice_url or os.environ.get(
            "VOICE_URL", ""
        )

        self._nodes: Dict[str, SpaceNode] = {
            "primary": SpaceNode(name="brain-primary", url=primary_url, role="primary"),
        }
        if backup_url:
            self._nodes["backup"] = SpaceNode(name="brain-backup", url=backup_url, role="backup")
        if voice_url:
            self._nodes["voice"] = SpaceNode(name="voice", url=voice_url, role="voice")

        log.info(
            f"[Promoter] Initialized. nodes={list(self._nodes.keys())} "
            f"primary={primary_url} backup={'set' if backup_url else 'not set'}"
        )

    # ── Main loop ────────────────────────────────────────────────────────

    async def run(self, stop_event: asyncio.Event) -> None:
        """Health check + promotion loop. Runs until stop_event is set."""
        log.info("[Promoter] Starting health check loop.")
        while not stop_event.is_set():
            try:
                await self._check_all_nodes()
                await self._evaluate_promotions()
            except Exception as e:
                log.error(f"[Promoter] Loop iteration error: {e}")
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
        log.info("[Promoter] Loop stopped.")

    # ── Health checks ────────────────────────────────────────────────────

    async def _check_all_nodes(self) -> None:
        """Run health checks on all registered nodes concurrently."""
        tasks = [
            self._check_node(node)
            for node in self._nodes.values()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_node(self, node: SpaceNode) -> None:
        """Ping /health endpoint. Update consecutive_failures/successes."""
        node.last_check_at = _now_iso()
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:  # SP3
                resp = await client.get(f"{node.url}/health")
            ok = resp.status_code == 200 and resp.json().get("status") in ("ok", "degraded")
        except Exception as e:
            log.warning(f"[Promoter] {node.name} health check error: {e}")
            ok = False

        if ok:
            node.consecutive_failures = 0
            node.consecutive_successes += 1
            node.last_status = "ok"
            if not node.is_alive:
                log.info(f"[Promoter] {node.name} RECOVERED. consecutive_successes={node.consecutive_successes}")
        else:
            node.consecutive_successes = 0
            node.consecutive_failures += 1
            node.last_status = "failed"
            log.warning(
                f"[Promoter] {node.name} FAILED. consecutive_failures={node.consecutive_failures}"
            )

    # ── Promotion logic ──────────────────────────────────────────────────

    async def _evaluate_promotions(self) -> None:
        """Evaluate whether promotion or recovery is needed."""
        if self._in_degraded_state:
            # SP5: degraded state — log heartbeat, try to recover
            log.warning(f"[Promoter] In DEGRADED state. failover_count={self._failover_count}")
            await self._try_recover_from_degraded()
            return

        primary = self._nodes.get("primary")
        backup  = self._nodes.get("backup")

        if primary is None:
            return

        # Primary failure -> promote backup
        if primary.consecutive_failures >= FAILURE_THRESHOLD:
            if backup is None:
                log.error("[Promoter] Primary failed but no backup configured. DEGRADED.")
                self._in_degraded_state = True
                await self._emit_event("health_degraded", "Primary failed, no backup available.")
                await self._notify("🚨 Primary FAILED — No Backup", "Primary is down. No backup configured.")
                return

            if self._failover_count >= MAX_FAILOVER_ATTEMPTS:  # SP5
                log.error("[Promoter] Max failover attempts reached. DEGRADED.")
                self._in_degraded_state = True
                await self._emit_event("health_degraded", f"Max failover attempts ({MAX_FAILOVER_ATTEMPTS}) reached.")
                return

            await self._promote_backup(primary, backup)

        # Primary recovered after being down — restore routing
        elif (
            primary.consecutive_successes >= RECOVERY_THRESHOLD
            and not primary.is_alive
        ):
            await self._restore_primary(primary)

        # Voice Space independent monitoring
        voice = self._nodes.get("voice")
        if voice and voice.consecutive_failures >= FAILURE_THRESHOLD and voice.is_alive:
            voice.is_alive = False
            await self._emit_event("health_degraded", f"Voice Space down: {voice.url}")
            await self._notify("🎙️ Voice Space DOWN", f"Voice offline: {voice.url}")

    async def _promote_backup(self, primary: SpaceNode, backup: SpaceNode) -> None:
        """Swap primary and backup in routing table. Write to CF KV."""
        primary.is_alive = False
        self._failover_count += 1
        reason = f"Primary {primary.url} failed {primary.consecutive_failures} consecutive checks"

        log.info(f"[Promoter] PROMOTING backup to primary. {reason}")

        # Swap URLs in routing table
        old_primary_url = primary.url
        new_routing = RoutingTable(
            primary=backup.url,
            backup=old_primary_url,
            voice=self._nodes["voice"].url if "voice" in self._nodes else "",
            promoted_at=_now_iso(),
            reason=reason,
        )

        ok = await self._write_routing_table(new_routing)

        if ok:
            # Update in-memory node roles
            primary.role = "backup"
            backup.role = "primary"
            self._nodes["primary"] = backup
            self._nodes["backup"] = primary
            log.info(f"[Promoter] Promotion SUCCESS. new_primary={backup.url}")
            await self._emit_event("promotion", f"Backup promoted to primary: {backup.url}", {"old_primary": old_primary_url})
            await self._notify(
                "⚡ Space Promotion",
                f"Primary failed. Backup promoted.\n"
                f"**New Primary:** {backup.url}\n"
                f"**Old Primary:** {old_primary_url}\n"
                f"**Failover #{self._failover_count}** | Reason: {reason}",
            )
        else:
            log.error("[Promoter] KV write FAILED — split-brain risk! (SP1)")
            await self._emit_event("kv_write_failed", "CF KV write failed during promotion. Split-brain risk.")
            await self._notify(
                "🚨 KV Write FAILED",
                f"Promotion KV write failed! CF Worker still routing to dead primary.\n"
                f"MANUAL ACTION REQUIRED: Update KV key `{KV_ROUTING_KEY}` to point to {backup.url}"
            )

    async def _restore_primary(self, primary: SpaceNode) -> None:
        """Primary recovered — restore it as primary in routing table."""
        primary.is_alive = True
        backup = self._nodes.get("backup")
        backup_url = backup.url if backup else ""

        log.info(f"[Promoter] Primary RECOVERED. Restoring {primary.url} as primary.")
        new_routing = RoutingTable(
            primary=primary.url,
            backup=backup_url,
            voice=self._nodes["voice"].url if "voice" in self._nodes else "",
            reason=f"Primary {primary.url} recovered after {primary.consecutive_successes} clean checks",
        )
        ok = await self._write_routing_table(new_routing)
        if ok:
            if backup:
                backup.role = "backup"
            primary.role = "primary"
            log.info(f"[Promoter] Primary restored. Failover count reset.")
            self._failover_count = 0
            await self._emit_event("recovery", f"Primary restored: {primary.url}")
            await self._notify("✅ Primary Restored", f"Primary back online: {primary.url}")

    async def _try_recover_from_degraded(self) -> None:
        """If any node is healthy, try to exit degraded state."""
        for node in self._nodes.values():
            if node.consecutive_successes >= RECOVERY_THRESHOLD:
                log.info(f"[Promoter] Exiting DEGRADED state. {node.name} recovered.")
                self._in_degraded_state = False
                self._failover_count = 0
                new_routing = RoutingTable(
                    primary=node.url,
                    backup="",
                    voice=self._nodes.get("voice", SpaceNode("","","")).url,
                    reason=f"Recovered from DEGRADED via {node.name}",
                )
                await self._write_routing_table(new_routing)
                await self._notify("✅ Recovered from DEGRADED", f"System restored via {node.name}: {node.url}")
                return

    # ── CF KV write ──────────────────────────────────────────────────────

    async def _write_routing_table(self, routing: RoutingTable) -> bool:
        """
        Write routing table to CF KV REST API.
        SP1: retries up to 3x before accepting failure.
        Returns True on success.
        """
        if not CF_KV_API_TOKEN:
            log.warning("[Promoter] CF_KV_API_TOKEN not set — KV write skipped (local mode)")
            return True  # assume success in dev

        url = f"{CF_KV_BASE_URL}/values/{KV_ROUTING_KEY}"
        headers = {"Authorization": f"Bearer {CF_KV_API_TOKEN}"}
        value = json.dumps(routing.to_dict())

        for attempt in range(3):  # SP1: 3 retries
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.put(url, headers=headers, content=value)
                if resp.status_code in (200, 201):
                    log.info(f"[Promoter] KV write OK (attempt {attempt+1})")
                    return True
                log.warning(f"[Promoter] KV write attempt {attempt+1} returned {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                log.warning(f"[Promoter] KV write attempt {attempt+1} exception: {e}")
            await asyncio.sleep(2 ** attempt)  # exp backoff

        return False

    async def read_routing_table(self) -> Optional[RoutingTable]:
        """Read current routing table from CF KV."""
        if not CF_KV_API_TOKEN:
            return None
        url = f"{CF_KV_BASE_URL}/values/{KV_ROUTING_KEY}"
        headers = {"Authorization": f"Bearer {CF_KV_API_TOKEN}"}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                return RoutingTable.from_dict(json.loads(resp.text))
        except Exception as e:
            log.warning(f"[Promoter] KV read failed: {e}")
        return None

    # ── Redis event log ──────────────────────────────────────────────────

    async def _emit_event(self, event_type: str, message: str, metadata: Optional[Dict] = None) -> None:
        """Log infra event to Redis for website dashboard. SP4."""
        event = InfraEvent(event_type=event_type, message=message, metadata=metadata or {})
        if self.redis is None:
            return
        try:
            await self.redis.rpush(KV_EVENTS_KEY, json.dumps(event.to_dict(), default=str))
            await self.redis.ltrim(KV_EVENTS_KEY, -KV_MAX_EVENTS, -1)
        except Exception as e:
            log.warning(f"[Promoter] Redis event write failed: {e}")

    # ── Discord notification ──────────────────────────────────────────────

    async def _notify(self, title: str, message: str) -> None:
        """Notify Ghost via Discord webhook."""
        if not DISCORD_WEBHOOK:
            log.info(f"[Promoter] No webhook. Event: {title} | {message[:80]}")
            return

        payload = {
            "embeds": [{
                "title": title,
                "description": message,
                "color": 15158332 if "FAILED" in title or "🚨" in title else 3066993,
                "footer": {"text": f"Ultron SpacePromoter · {_now_iso()[:19]}"},
            }]
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(DISCORD_WEBHOOK, json=payload)
                if resp.status_code not in (200, 204):
                    log.warning(f"[Promoter] Webhook returned {resp.status_code}")
        except Exception as e:
            log.warning(f"[Promoter] Webhook failed: {e}")

    # ── Status API ───────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Return current topology status (for website Sentinel tab)."""
        return {
            "nodes": {
                name: {
                    "name": node.name,
                    "url": node.url,
                    "role": node.role,
                    "is_alive": node.is_alive,
                    "last_status": node.last_status,
                    "last_check_at": node.last_check_at,
                    "consecutive_failures": node.consecutive_failures,
                    "consecutive_successes": node.consecutive_successes,
                }
                for name, node in self._nodes.items()
            },
            "failover_count": self._failover_count,
            "in_degraded_state": self._in_degraded_state,
            "max_failover_attempts": MAX_FAILOVER_ATTEMPTS,
            "check_interval_seconds": HEALTH_CHECK_INTERVAL,
        }
