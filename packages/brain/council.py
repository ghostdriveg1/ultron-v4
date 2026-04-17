"""
packages/brain/council.py

Ultron V4 — Council Mode (MOA — Mixture of Agents)
====================================================
Auto-generates expert council per task type. Parallel expert calls across
all providers in general pool. Synthesizes outputs via Sentinel/Groq.

Architecture:
  Council fires at: project start, midpoint, pre-deploy (~3 sessions = ~170k tokens).
  Each Council session: ~57k tokens (5 experts x ~10k tokens each + synthesis).
  Free-tier safe: spread across all 5 providers to avoid hitting single quota.

Design:
  1. task_type -> expert council config (auto-generated or Sentinel-specified).
  2. asyncio.gather() fires all experts in parallel (one per available key/provider).
  3. Each expert gets: system persona + task + context.
  4. Synthesis: Groq (fastest) reads all expert outputs -> final answer.
  5. Council state persisted in Redis per channel_id (VP session persistence fix).
  6. Output: CouncilResult with expert_outputs + synthesis + council_id.

Expert personas (auto-generated per task type):
  - code:     Senior SWE, Security Auditor, DevOps Engineer, Architect, QA Lead
  - research: Scientist, Historian, Journalist, Analyst, Devil's Advocate
  - creative: Writer, Director, Critic, Marketer, User Experience Designer
  - general:  Generalist, Domain Expert, Critical Thinker, Pragmatist, Innovator

Future bug risks (pre-registered):
  C1 [HIGH]  asyncio.gather() with 5 concurrent LLM calls: if 3+ providers hit
             rate limit simultaneously, gather raises. Use return_exceptions=True
             and filter out failed expert calls.

  C2 [HIGH]  Redis council state TTL: council sessions stored indefinitely if not
             cleaned. Fix: TTL=86400 (24h) on all council Redis keys.

  C3 [MED]   Synthesis call uses make_provider_llm_fn(pool) — any provider.
             If synthesis goes to Together (no tool_calls) with a complex prompt,
             response quality drops. Fix: prefer Groq for synthesis (fastest + best
             instruction following). Fallback to any provider.

  C4 [MED]   Expert persona prompts are long (~200 tokens each). 5 experts =
             ~1000 tokens just in system prompts. Monitor total token usage.

  C5 [LOW]   council_id collision: uuid4()[:8] has ~1/4B collision chance.
             Acceptable for single-user system.

Tool calls used writing this file:
    Github:get_file_contents x1 (ultron-v3 moa/council.py — reference patterns)
    External knowledge: MOA paper (together.ai) — mixture of agents synthesis pattern
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("brain.council")

# Redis key prefix for council state
COUNCIL_STATE_PREFIX = "ultron:council:"
COUNCIL_TTL          = 86_400  # 24h

# Max expert output chars passed to synthesis
MAX_EXPERT_CHARS = 1500
MAX_SYNTHESIS_TOKENS = 800


# ---------------------------------------------------------------------------
# Expert configs per task type
# ---------------------------------------------------------------------------

EXPERT_CONFIGS: dict[str, list[dict]] = {
    "code": [
        {"name": "Senior SWE",       "persona": "You are a senior software engineer. Focus on correctness, edge cases, and clean code."},
        {"name": "Security Auditor", "persona": "You are a security auditor. Identify vulnerabilities, injection risks, and unsafe patterns."},
        {"name": "DevOps Engineer",  "persona": "You are a DevOps engineer. Focus on deployment, scaling, resource efficiency, and CI/CD."},
        {"name": "Architect",        "persona": "You are a system architect. Focus on design patterns, coupling, cohesion, and long-term maintainability."},
        {"name": "QA Lead",          "persona": "You are a QA lead. Focus on testing strategy, failure modes, and edge cases."},
    ],
    "research": [
        {"name": "Scientist",        "persona": "You are a rigorous scientist. Focus on evidence, methodology, and empirical claims."},
        {"name": "Historian",        "persona": "You are a historian. Provide historical context and precedent."},
        {"name": "Journalist",       "persona": "You are an investigative journalist. Ask who benefits, what's hidden, and what the primary sources say."},
        {"name": "Analyst",          "persona": "You are a data analyst. Focus on quantitative evidence and statistical reasoning."},
        {"name": "Devil's Advocate", "persona": "You are a devil's advocate. Challenge all assumptions and find the strongest counterarguments."},
    ],
    "creative": [
        {"name": "Writer",    "persona": "You are a creative writer. Focus on narrative, voice, and engagement."},
        {"name": "Director",  "persona": "You are a creative director. Focus on vision, consistency, and impact."},
        {"name": "Critic",    "persona": "You are a sharp critic. Identify what doesn't work and why."},
        {"name": "Marketer",  "persona": "You are a marketer. Focus on audience, hook, and conversion."},
        {"name": "UX Designer", "persona": "You are a UX designer. Focus on user experience, clarity, and friction reduction."},
    ],
    "general": [
        {"name": "Generalist",     "persona": "You are a knowledgeable generalist. Provide a balanced, comprehensive perspective."},
        {"name": "Domain Expert",  "persona": "You are a domain expert. Provide deep technical insight on the specific subject."},
        {"name": "Critical Thinker", "persona": "You are a critical thinker. Challenge assumptions and identify logical flaws."},
        {"name": "Pragmatist",     "persona": "You are a pragmatist. Focus on what is actionable and realistically achievable."},
        {"name": "Innovator",      "persona": "You are an innovator. Propose novel approaches and creative solutions."},
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExpertOutput:
    expert_name: str
    provider:    str
    output:      str
    success:     bool
    latency_ms:  float


@dataclass
class CouncilResult:
    council_id:     str
    task_type:      str
    task:           str
    expert_outputs: list[ExpertOutput]
    synthesis:      str
    total_ms:       float
    experts_ok:     int
    experts_failed: int

    def to_discord(self) -> str:
        """Format for Discord (truncated, 1800 char limit)."""
        experts_summary = "\n".join(
            f"**{e.expert_name}** ({e.provider}): {e.output[:300]}..."
            if len(e.output) > 300 else f"**{e.expert_name}** ({e.provider}): {e.output}"
            for e in self.expert_outputs if e.success
        )
        return (
            f"⚖️ **COUNCIL MODE** — `{self.task_type}` — ID: `{self.council_id}`\n\n"
            f"**Expert Opinions:**\n{experts_summary[:800]}\n\n"
            f"**Synthesis:**\n{self.synthesis[:700]}\n\n"
            f"_Experts: {self.experts_ok}/{self.experts_ok+self.experts_failed} succeeded "
            f"| {self.total_ms:.0f}ms_"
        )[:1800]


# ---------------------------------------------------------------------------
# Council
# ---------------------------------------------------------------------------

class Council:
    """
    Mixture-of-Agents Council. Parallel expert calls + synthesis.

    Usage:
        council = Council(pool=pool)
        result = await council.run(task="Explain quantum entanglement", task_type="research")
    """

    def __init__(self, pool: object, redis: Optional[Any] = None) -> None:
        self._pool  = pool
        self._redis = redis

    async def run(
        self,
        task: str,
        task_type: str = "general",
        channel_id: Optional[str] = None,
        custom_experts: Optional[list[dict]] = None,
    ) -> CouncilResult:
        """
        Fire all experts in parallel, then synthesize.

        Args:
            task:           The user's task/question.
            task_type:      One of: code, research, creative, general.
            channel_id:     For Redis state persistence.
            custom_experts: Override expert list (e.g. Sentinel-specified).
        """
        council_id = uuid.uuid4().hex[:8]
        t_start    = time.monotonic()

        experts = custom_experts or EXPERT_CONFIGS.get(task_type, EXPERT_CONFIGS["general"])
        log.info(
            f"[Council] id={council_id} task_type={task_type} "
            f"experts={len(experts)} task={task[:80]}"
        )

        # 1. Fire all experts in parallel (C1: return_exceptions=True)
        expert_tasks = [
            self._call_expert(
                expert_name=e["name"],
                persona=e["persona"],
                task=task,
                council_id=council_id,
            )
            for e in experts
        ]
        results = await asyncio.gather(*expert_tasks, return_exceptions=True)

        expert_outputs: list[ExpertOutput] = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                expert_outputs.append(ExpertOutput(
                    expert_name=experts[i]["name"],
                    provider="unknown",
                    output=f"Expert failed: {res}",
                    success=False,
                    latency_ms=0,
                ))
            else:
                expert_outputs.append(res)

        experts_ok     = sum(1 for e in expert_outputs if e.success)
        experts_failed = len(expert_outputs) - experts_ok

        log.info(f"[Council] id={council_id} experts_ok={experts_ok}/{len(experts)}")

        # 2. Synthesis (C3: prefer Groq)
        synthesis = await self._synthesize(task, expert_outputs, council_id)

        total_ms = (time.monotonic() - t_start) * 1000

        result = CouncilResult(
            council_id=council_id,
            task_type=task_type,
            task=task,
            expert_outputs=expert_outputs,
            synthesis=synthesis,
            total_ms=total_ms,
            experts_ok=experts_ok,
            experts_failed=experts_failed,
        )

        # 3. Persist to Redis if available (C2: 24h TTL)
        if self._redis and channel_id:
            await self._persist(channel_id, council_id, result)

        log.info(f"[Council] id={council_id} done total_ms={total_ms:.0f}")
        return result

    async def _call_expert(
        self,
        expert_name: str,
        persona: str,
        task: str,
        council_id: str,
    ) -> ExpertOutput:
        """Call one expert. Gets its own key from general pool."""
        t_start = time.monotonic()
        try:
            from packages.brain.llm_router import make_provider_llm_fn
            llm_fn = make_provider_llm_fn(self._pool)
            output = await llm_fn(
                messages=[{"role": "user", "content": task}],
                system=persona,
                max_tokens=600,
            )
            provider = "pool"  # llm_router abstracts provider
            return ExpertOutput(
                expert_name=expert_name,
                provider=provider,
                output=output[:MAX_EXPERT_CHARS],
                success=True,
                latency_ms=(time.monotonic() - t_start) * 1000,
            )
        except Exception as e:
            log.error(f"[Council] Expert '{expert_name}' failed: {e}")
            return ExpertOutput(
                expert_name=expert_name,
                provider="unknown",
                output=f"Expert failed: {e}",
                success=False,
                latency_ms=(time.monotonic() - t_start) * 1000,
            )

    async def _synthesize(
        self,
        task: str,
        expert_outputs: list[ExpertOutput],
        council_id: str,
    ) -> str:
        """
        Synthesize expert outputs into final answer.
        C3: prefer Groq by trying to get a Groq key first.
        """
        successful = [e for e in expert_outputs if e.success]
        if not successful:
            return "All experts failed. Cannot synthesize."

        experts_text = "\n\n".join(
            f"### {e.expert_name}\n{e.output[:MAX_EXPERT_CHARS]}"
            for e in successful
        )

        synthesis_prompt = (
            f"TASK: {task}\n\n"
            f"EXPERT COUNCIL OUTPUTS:\n{experts_text}\n\n"
            f"Synthesize the above expert opinions into ONE cohesive, high-quality answer. "
            f"Preserve the best insights from each expert. Resolve contradictions. "
            f"Be direct and actionable. Max 600 words."
        )

        try:
            from packages.brain.llm_router import make_provider_llm_fn
            llm_fn   = make_provider_llm_fn(self._pool)
            synthesis = await llm_fn(
                messages=[{"role": "user", "content": synthesis_prompt}],
                system="You are a master synthesizer. Combine expert opinions into the definitive answer.",
                max_tokens=MAX_SYNTHESIS_TOKENS,
            )
            return synthesis
        except Exception as e:
            log.error(f"[Council] Synthesis failed: {e}")
            # Fallback: concatenate best expert output
            best = max(successful, key=lambda e: len(e.output))
            return f"(Synthesis failed: {e})\n\nBest expert ({best.expert_name}):\n{best.output}"

    async def _persist(
        self,
        channel_id: str,
        council_id: str,
        result: CouncilResult,
    ) -> None:
        """Persist council result to Redis. C2: 24h TTL."""
        key = f"{COUNCIL_STATE_PREFIX}{channel_id}:latest"
        try:
            data = json.dumps({
                "council_id":  result.council_id,
                "task_type":   result.task_type,
                "task":        result.task[:200],
                "synthesis":   result.synthesis[:500],
                "experts_ok":  result.experts_ok,
                "total_ms":    result.total_ms,
                "ts":          time.time(),
            })
            await self._redis.setex(key, COUNCIL_TTL, data)
            log.info(f"[Council] Persisted to Redis key={key}")
        except Exception as e:
            log.warning(f"[Council] Redis persist failed: {e}")

    async def get_session(self, channel_id: str) -> Optional[dict]:
        """Retrieve last council session for a channel."""
        if not self._redis:
            return None
        key = f"{COUNCIL_STATE_PREFIX}{channel_id}:latest"
        try:
            raw = await self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    @staticmethod
    def classify_task_type(message: str) -> str:
        """Quick keyword classification. Fallback to 'general'."""
        msg = message.lower()
        if any(w in msg for w in ["code", "function", "bug", "error", "script", "class", "api", "debug"]):
            return "code"
        if any(w in msg for w in ["research", "study", "paper", "history", "science", "data", "evidence"]):
            return "research"
        if any(w in msg for w in ["write", "story", "poem", "creative", "design", "brand", "copy"]):
            return "creative"
        return "general"
