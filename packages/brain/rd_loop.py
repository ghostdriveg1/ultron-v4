"""
packages/brain/rd_loop.py

Autonomous R&D loop for Ultron V4.
Sources: microsoft/RD-Agent (core/proposal.py, evolving_framework.py)
         + EverMemOS Foresight signals
         + AiScientist File-as-Bus pattern (arXiv:2604.13018)

Behavior:
  1. Task completes (deadline reached, output delivered)
  2. RDLoop.start(project_context) called
  3. Foresight signals from LifecycleEngine → candidate improvements
  4. Sentinel ranks by estimated impact
  5. Council debates top-3 (via llm_fn)
  6. Winning improvement → notify Ghost (Discord webhook + Redis event)
  7. Loop sleeps, then proposes next round
  8. Stops when stop_event is set OR resource_exhausted

This is what separates Ultron from every other AI product:
it finishes tasks and then CHOOSES to improve them.

Pre-registered bugs:
  RD1 [HIGH]  Sentinel /sentinel/event call fails mid-loop → catch, log, degrade gracefully
  RD2 [HIGH]  LLM propose call returns malformed JSON → parse with fallback, never crash loop
  RD3 [MED]   Council gather() can return exception objects → filter with isinstance check
  RD4 [MED]   stop_event not set on HF Space restart → add Redis-backed stop flag as fallback
  RD5 [LOW]   Notification webhook call fails → log, don't retry more than 3 times
  RD6 [LOW]   Loop runs forever if Foresight never generates improvements → max_rounds guard
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import httpx

log = logging.getLogger("ultron.rd_loop")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

RD_SLEEP_SECONDS = 600          # 10min between R&D cycles
MAX_ROUNDS = 20                 # RD6: max improvement cycles per project
COUNCIL_EXPERTS = 3             # experts per Council debate
NOTIFY_MAX_RETRIES = 3          # RD5
SENTINEL_TIMEOUT_SECONDS = 30.0 # RD1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class Improvement:
    """A proposed improvement to a completed project."""
    imp_id:           str = field(default_factory=lambda: str(uuid.uuid4()))
    description:      str = ""
    domain:           str = ""          # e.g. "ChemE", "UI", "API"
    estimated_impact: float = 0.0       # Sentinel-ranked 0.0-1.0
    rationale:        str = ""
    status:           str = "proposed"  # proposed | debating | accepted | rejected | implemented
    created_at:       str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Improvement":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RDState:
    """Persisted R&D loop state (Redis key: rd:state:{user_id})."""
    user_id:          str = ""
    project_id:       str = ""
    project_summary:  str = ""
    round:            int = 0
    implemented:      List[str] = field(default_factory=list)   # imp_ids
    rejected:         List[str] = field(default_factory=list)
    started_at:       str = field(default_factory=_now_iso)
    last_round_at:    str = field(default_factory=_now_iso)
    active:           bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "RDState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────
# RDLoop
# ─────────────────────────────────────────────

class RDLoop:
    """
    Autonomous post-deadline R&D engine.

    Usage:
        loop = RDLoop(redis, lifecycle_engine, brain_url, auth_token, discord_webhook)
        stop = asyncio.Event()
        asyncio.create_task(
            loop.run(
                user_id="ghost",
                project_summary="Built calculator website",
                llm_fn=make_provider_llm_fn(pool),
                stop_event=stop,
            )
        )
        # ... later ...
        stop.set()  # stops the loop
    """

    def __init__(
        self,
        redis_client,
        lifecycle,          # LifecycleEngine
        brain_url: str,
        auth_token: str,
        discord_webhook: Optional[str] = None,
    ):
        self.redis = redis_client
        self.lifecycle = lifecycle
        self.brain_url = brain_url
        self.auth_token = auth_token
        self.discord_webhook = discord_webhook

    # ── Main loop ────────────────────────────

    async def run(
        self,
        user_id: str,
        project_summary: str,
        llm_fn: Callable,
        stop_event: asyncio.Event,
        sleep_seconds: int = RD_SLEEP_SECONDS,
    ) -> None:
        """Main R&D loop. Runs until stop_event set or MAX_ROUNDS reached."""
        project_id = str(uuid.uuid4())
        state = RDState(
            user_id=user_id,
            project_id=project_id,
            project_summary=project_summary,
        )
        await self._save_state(user_id, state)

        log.info(f"[RDLoop] Started for user={user_id} project={project_id}")
        await self._notify(
            title="⚡ R&D Loop Started",
            message=f"Ultron completed task. Starting autonomous R&D improvements.\n**Project:** {project_summary[:200]}",
        )

        for round_n in range(MAX_ROUNDS):  # RD6
            if stop_event.is_set():
                log.info(f"[RDLoop] Stop event. Exiting after round {round_n}.")
                break
            if await self._redis_stop_flag(user_id):  # RD4
                log.info(f"[RDLoop] Redis stop flag. Exiting.")
                break

            state.round = round_n + 1
            state.last_round_at = _now_iso()
            await self._save_state(user_id, state)

            log.info(f"[RDLoop] Round {state.round} begin")

            # 1. Get Foresight signals
            foresight = await self.lifecycle.get_foresight(user_id)
            foresight_signals: List[str] = []
            if foresight and foresight.is_valid():
                foresight_signals = foresight.predictions

            # 2. Propose improvements
            improvements = await self.propose_improvements(
                project_summary, foresight_signals, state.implemented, llm_fn
            )
            if not improvements:
                log.info("[RDLoop] No improvements proposed. Sleeping.")
                await asyncio.sleep(sleep_seconds)
                continue

            # 3. Sentinel rank
            ranked = await self.sentinel_rank(improvements)
            top3 = ranked[:3]

            # 4. Council debate
            winner = await self.council_debate(top3, project_summary, llm_fn)
            if not winner:
                log.info("[RDLoop] Council produced no winner. Sleeping.")
                await asyncio.sleep(sleep_seconds)
                continue

            winner.status = "accepted"
            state.implemented.append(winner.imp_id)

            # 5. Notify Ghost
            await self._notify(
                title=f"🔬 R&D Round {state.round}: Improvement Selected",
                message=(
                    f"**{winner.description}**\n"
                    f"Domain: {winner.domain} | Impact: {winner.estimated_impact:.2f}\n"
                    f"Rationale: {winner.rationale[:300]}"
                ),
            )

            # 6. Log to Redis
            await self.redis.rpush(
                f"rd:history:{user_id}",
                json.dumps(winner.to_dict(), default=str),
            )

            await self._save_state(user_id, state)
            log.info(f"[RDLoop] Round {state.round} complete. Winner: {winner.description[:60]}")

            await asyncio.sleep(sleep_seconds)

        state.active = False
        await self._save_state(user_id, state)
        log.info(f"[RDLoop] Loop complete. Rounds: {state.round}")

    # ── Propose ──────────────────────────────

    async def propose_improvements(
        self,
        project_summary: str,
        foresight_signals: List[str],
        already_implemented: List[str],
        llm_fn: Callable,
    ) -> List[Improvement]:
        """
        Ask LLM to propose improvements informed by project context + Foresight.
        RD2: parse failures return empty list, never raise.
        """
        foresight_text = (
            "\n".join(f"- {s}" for s in foresight_signals)
            if foresight_signals
            else "No foresight signals available."
        )
        already_text = (
            f"Already implemented {len(already_implemented)} improvements this session."
            if already_implemented
            else "No improvements implemented yet."
        )

        prompt = [
            {"role": "system", "content": (
                "You are Ultron's autonomous R&D engine. "
                "Your job is to propose meaningful improvements to a completed project. "
                "Return a JSON array of improvement objects. Each has: "
                "description (str), domain (str), estimated_impact (float 0-1), rationale (str). "
                "Propose 3-5 improvements. Be specific and actionable. JSON only, no markdown."
            )},
            {"role": "user", "content": (
                f"Completed project:\n{project_summary}\n\n"
                f"Predicted future needs (Foresight):\n{foresight_text}\n\n"
                f"{already_text}\n\n"
                "Propose 3-5 concrete improvements:"
            )}
        ]

        try:
            raw = await llm_fn(prompt)
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            if not isinstance(data, list):
                return []
            improvements = []
            for d in data[:5]:
                try:
                    imp = Improvement(
                        description=str(d.get("description", "")),
                        domain=str(d.get("domain", "")),
                        estimated_impact=float(d.get("estimated_impact", 0.5)),
                        rationale=str(d.get("rationale", "")),
                    )
                    if imp.description:
                        improvements.append(imp)
                except Exception:
                    continue
            return improvements
        except Exception as e:
            log.warning(f"[RDLoop] propose_improvements parse error: {e}")  # RD2
            return []

    # ── Sentinel rank ────────────────────────

    async def sentinel_rank(
        self, improvements: List[Improvement]
    ) -> List[Improvement]:
        """
        POST /sentinel/event {type: rd_rank, improvements: [...]}
        Sentinel returns ranked list. Falls back to estimated_impact sort on failure.
        RD1: catch httpx errors, degrade gracefully.
        """
        try:
            async with httpx.AsyncClient(timeout=SENTINEL_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    f"{self.brain_url}/sentinel/event",
                    json={
                        "type": "rd_rank",
                        "improvements": [i.to_dict() for i in improvements],
                    },
                    headers={"Authorization": f"Bearer {self.auth_token}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    ranked_ids = data.get("ranked_imp_ids", [])
                    if ranked_ids:
                        id_map = {i.imp_id: i for i in improvements}
                        ordered = [id_map[rid] for rid in ranked_ids if rid in id_map]
                        remaining = [i for i in improvements if i.imp_id not in set(ranked_ids)]
                        return ordered + remaining
        except Exception as e:
            log.warning(f"[RDLoop] Sentinel rank failed: {e}. Using estimated_impact fallback.")  # RD1

        # fallback: sort by estimated_impact
        return sorted(improvements, key=lambda x: x.estimated_impact, reverse=True)

    # ── Council debate ───────────────────────

    async def council_debate(
        self,
        candidates: List[Improvement],
        project_context: str,
        llm_fn: Callable,
    ) -> Optional[Improvement]:
        """
        Each expert evaluates all candidates and votes.
        Majority vote selects winner.
        RD3: filter exception objects from gather().
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        candidates_text = "\n".join(
            f"{i+1}. [{c.domain}] {c.description} (impact: {c.estimated_impact:.2f})"
            for i, c in enumerate(candidates)
        )

        expert_roles = [
            "pragmatic engineer focused on feasibility and immediate value",
            "user experience specialist focused on Ghost's daily workflow",
            "systems architect focused on long-term maintainability",
        ]

        async def expert_vote(role: str) -> Optional[int]:
            prompt = [
                {"role": "system", "content": f"You are a {role}. Pick the SINGLE best improvement. Return only the number (1, 2, or 3)."},
                {"role": "user", "content": f"Project: {project_context[:300]}\n\nCandidates:\n{candidates_text}\n\nPick the best (return number only):"}
            ]
            try:
                raw = await llm_fn(prompt)
                n = int(raw.strip().split()[0])
                return n if 1 <= n <= len(candidates) else None
            except Exception:
                return None

        results = await asyncio.gather(
            *[expert_vote(r) for r in expert_roles[:COUNCIL_EXPERTS]],
            return_exceptions=True,
        )

        # RD3: filter exceptions
        votes = [v for v in results if isinstance(v, int) and v is not None]

        if not votes:
            return candidates[0]  # fallback: first = highest impact

        # majority vote
        from collections import Counter
        most_common = Counter(votes).most_common(1)
        winner_idx = most_common[0][0] - 1  # 1-indexed → 0-indexed
        return candidates[min(winner_idx, len(candidates) - 1)]

    # ── Notify ───────────────────────────────

    async def _notify(
        self, title: str, message: str
    ) -> None:
        """Notify Ghost via Discord webhook. RD5: max 3 retries."""
        if not self.discord_webhook:
            log.info(f"[RDLoop] No webhook. Notification: {title}")
            return

        payload = {
            "embeds": [{
                "title": title,
                "description": message,
                "color": 3066993,  # green
                "footer": {"text": f"Ultron R&D Loop · {_now_iso()[:19]}"},
            }]
        }

        for attempt in range(NOTIFY_MAX_RETRIES):  # RD5
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(self.discord_webhook, json=payload)
                    if resp.status_code in (200, 204):
                        return
                    log.warning(f"[RDLoop] Webhook attempt {attempt+1} returned {resp.status_code}")
            except Exception as e:
                log.warning(f"[RDLoop] Webhook attempt {attempt+1} failed: {e}")
            await asyncio.sleep(2 ** attempt)  # exp backoff

    # ── State persistence ────────────────────

    async def _save_state(self, user_id: str, state: RDState) -> None:
        await self.redis.set(
            f"rd:state:{user_id}",
            json.dumps(state.to_dict(), default=str),
        )

    async def get_state(self, user_id: str) -> Optional[RDState]:
        raw = await self.redis.get(f"rd:state:{user_id}")
        if not raw:
            return None
        return RDState.from_dict(json.loads(raw))

    async def stop(self, user_id: str) -> None:
        """RD4: Redis-backed stop flag for cross-process stop."""
        await self.redis.set(f"rd:stop:{user_id}", "1", ex=3600)

    async def _redis_stop_flag(self, user_id: str) -> bool:
        return bool(await self.redis.get(f"rd:stop:{user_id}"))

    async def get_history(
        self, user_id: str, limit: int = 20
    ) -> List[Improvement]:
        """Return implemented improvements for user."""
        raw_items = await self.redis.lrange(f"rd:history:{user_id}", -limit, -1)
        items = []
        for raw in reversed(raw_items):
            try:
                items.append(Improvement.from_dict(json.loads(raw)))
            except Exception:
                continue
        return items
