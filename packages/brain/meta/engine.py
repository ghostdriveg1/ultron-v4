"""
packages/brain/meta/engine.py

Ultron V4 — Metacognition Engine
=================================
Cherry-picked from V5 meta/engine.py (GLM-4 session v30 audit — genuine new capability).
Adapted to V4 pool interface: no V5 ProviderType enum, uses V4 pool.get_key() directly.

Capabilities:
  1. pre_action_assessment() — called BEFORE every dispatch: mode, confidence, knowledge
  2. post_action_reflection() — called AFTER every dispatch: quality score, lessons, adapt
  3. executive_prioritize()  — Eisenhower matrix task ranking
  4. generate_creative_ideas() — divergent thinking via LLM
  5. Strategy adaptation      — switches to cautious approach if quality avg < 0.5

Integration points:
  task_dispatcher.py: pre_action_assessment() before dispatch, post_action_reflection() after
  main.py lifespan:   get_metacognition() init (Step 5e)
  /metacog/state:     GET endpoint returns get_state() for dashboard

Future bug risks (pre-registered):
  MC1 [HIGH]  _llm_callback set via set_llm_callback() must be awaited inside engine.
              If pool is exhausted when metacog tries to call LLM for reflection,
              _llm_callback raises AllKeysExhaustedError → reflection silently skipped.
              Fine for now (non-critical path), but log the failure explicitly.

  MC2 [MED]   post_action_reflection() updates _confidence_calibrations keyed by
              action[:50]. If two simultaneous dispatches (M2 risk) call this concurrently,
              dict mutation is not thread-safe (Python GIL protects list.append but not
              dict growth in complex concurrent scenarios). Add asyncio.Lock around writes.

  MC3 [MED]   _reflection_history capped at 100. After 100 reflections, oldest lessons
              pruned. If Ghost runs V4 24/7 with Discord active, 100 reflections hit
              within hours. Consider Redis persistence (key: ultron:metacog:reflections)
              with same TTL=7200 as planner plans.

  MC4 [LOW]   generate_creative_ideas() fires LLM call without pool reporting
              (uses _llm_callback which already does report internally via make_provider_llm_fn).
              Verify _llm_callback is always the pool-backed function, never a raw httpx call.

  MC5 [LOW]   executive_prioritize() mutates task dicts in-place. Callers that pass the
              same task dicts to other components will see injected priority/priority_score
              fields. Document this side effect or deepcopy before mutating.

  MC6 [LOW]   cognitive mode detection is keyword-only (action_lower string scan).
              Multi-intent messages ("analyze and create a plan") may set wrong mode.
              Future: use Groq classify call (1 token classification) like planner.py does.

Tool calls used writing this file:
  Github:get_file_contents x3 (pool.py, task_dispatcher.py, main.py)
  Github:push_files x1
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ultron.metacognition")


# =============================================================================
# Cognitive Dimensions
# =============================================================================

class CognitiveMode(str, Enum):
    FOCUSED     = "focused"      # Deep concentration on single task
    EXPLORATORY = "exploratory"  # Broad exploration, brainstorming
    ANALYTICAL  = "analytical"   # Logical reasoning, data-driven
    CREATIVE    = "creative"     # Divergent thinking, innovation
    REFLECTIVE  = "reflective"   # Self-review, metacognition
    EXECUTIVE   = "executive"    # Planning, prioritizing, delegating


class ConfidenceLevel(str, Enum):
    VERY_HIGH = "very_high"  # >90%
    HIGH      = "high"       # 70-90%
    MEDIUM    = "medium"     # 40-70%
    LOW       = "low"        # 10-40%
    VERY_LOW  = "very_low"   # <10%


class KnowledgeState(str, Enum):
    KNOWN        = "known"
    LIKELY_KNOWN = "likely_known"
    UNCERTAIN    = "uncertain"
    UNKNOWN      = "unknown"
    CONFLICTING  = "conflicting"


@dataclass
class CognitiveState:
    mode:               CognitiveMode    = CognitiveMode.FOCUSED
    confidence:         ConfidenceLevel  = ConfidenceLevel.MEDIUM
    knowledge_state:    KnowledgeState   = KnowledgeState.UNCERTAIN
    attention_focus:    str              = ""
    task_stack:         List[str]        = field(default_factory=list)
    recent_reflections: List[Dict]       = field(default_factory=list)
    creative_ideas:     List[Dict]       = field(default_factory=list)
    learning_history:   List[Dict]       = field(default_factory=list)
    emotional_valence:  float            = 0.5   # 0=negative, 1=positive
    urgency_level:      float            = 0.5   # 0=relaxed, 1=critical


@dataclass
class Reflection:
    id:           str
    action:       str
    outcome:      str
    quality_score: float            # 0-1
    lessons:      List[str]
    improvements: List[str]
    timestamp:    str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class CreativeIdea:
    id:               str
    idea:             str
    novelty_score:    float
    feasibility_score: float
    domain:           str
    connections:      List[str]
    timestamp:        str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# =============================================================================
# MetacognitionEngine
# =============================================================================

class MetacognitionEngine:
    """
    V4 Metacognition Engine — Ultron's inner intelligence layer.

    Runs alongside all other systems:
      Before action  → pre_action_assessment()
      After action   → post_action_reflection()
      On demand      → executive_prioritize(), generate_creative_ideas()

    Set _llm_callback via set_llm_callback(fn) where fn is
    make_provider_llm_fn(pool) result — pool-backed async callable.
    MC4: always use pool-backed fn, never raw httpx.
    """

    def __init__(self) -> None:
        self._state                 = CognitiveState()
        self._llm_callback          = None
        self._reflection_history:   List[Reflection]      = []
        self._idea_bank:            List[CreativeIdea]    = []
        self._strategy_adaptations: List[Dict]            = []
        self._confidence_calibrations: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()  # MC2: guard concurrent dict writes

    def set_llm_callback(self, callback) -> None:
        self._llm_callback = callback

    # -------------------------------------------------------------------------
    # pre_action_assessment — called by task_dispatcher BEFORE dispatch
    # -------------------------------------------------------------------------

    async def pre_action_assessment(
        self, action: str, context: Dict = None
    ) -> Dict[str, Any]:
        """
        Assess before acting:
          1. Cognitive mode from action keywords
          2. Knowledge state from reflection history
          3. Confidence level mapped from knowledge state
          4. Recommendation (execute direct / research first / etc.)
        """
        action_lower = action.lower()

        # Cognitive mode from keywords
        if any(kw in action_lower for kw in ["create", "design", "imagine", "brainstorm"]):
            self._state.mode = CognitiveMode.CREATIVE
        elif any(kw in action_lower for kw in ["analyze", "evaluate", "compare"]):
            self._state.mode = CognitiveMode.ANALYTICAL
        elif any(kw in action_lower for kw in ["explore", "search", "find"]):
            self._state.mode = CognitiveMode.EXPLORATORY
        elif any(kw in action_lower for kw in ["plan", "prioritize", "decide"]):
            self._state.mode = CognitiveMode.EXECUTIVE
        elif any(kw in action_lower for kw in ["review", "reflect", "assess"]):
            self._state.mode = CognitiveMode.REFLECTIVE
        else:
            self._state.mode = CognitiveMode.FOCUSED

        knowledge   = await self._assess_knowledge(action, context)
        confidence  = self._assess_confidence(knowledge)

        self._state.confidence     = confidence
        self._state.knowledge_state = knowledge
        self._state.attention_focus = action[:100]

        return {
            "mode":               self._state.mode.value,
            "confidence":         self._state.confidence.value,
            "knowledge_state":    self._state.knowledge_state.value,
            "should_gather_info": knowledge in (KnowledgeState.UNCERTAIN, KnowledgeState.UNKNOWN),
            "recommended_approach": self._recommend_approach(),
        }

    # -------------------------------------------------------------------------
    # post_action_reflection — called by task_dispatcher AFTER dispatch
    # -------------------------------------------------------------------------

    async def post_action_reflection(
        self, action: str, outcome: str, success: bool, context: Dict = None
    ) -> Reflection:
        """
        Reflect after acting:
          1. Quality score (0.9 success / 0.3 failure baseline)
          2. LLM deep reflection if outcome is substantial (>50 chars)
          3. Strategy adaptation if rolling avg < 0.5
          4. Confidence calibration update
        MC1: AllKeysExhaustedError during LLM reflection → skip gracefully, log.
        """
        quality_score = 0.9 if success else 0.3

        if self._llm_callback and len(outcome) > 50:
            try:
                reflection_prompt = (
                    f"Reflect on this action and its outcome.\n"
                    f"Action: {action[:500]}\n"
                    f"Outcome: {outcome[:1000]}\n"
                    f"Success: {success}\n\n"
                    f"Respond ONLY with JSON (no markdown):\n"
                    f'{{"quality_score": 0.0, "lessons": [], "improvements": [], "strategy_adaptation": ""}}'
                )
                msgs = [{"role": "user", "content": reflection_prompt}]
                resp = await self._llm_callback(msgs, [])
                if resp and resp.get("content"):
                    raw = resp["content"]
                    s, e = raw.find("{"), raw.rfind("}") + 1
                    if s >= 0 and e > s:
                        data = json.loads(raw[s:e])
                        quality_score = float(data.get("quality_score", quality_score))
            except Exception as exc:
                logger.warning(f"[Metacog] LLM reflection skipped: {exc}")  # MC1

        reflection = Reflection(
            id=hashlib.md5(f"reflect:{action}:{time.time()}".encode()).hexdigest()[:10],
            action=action[:500],
            outcome=outcome[:500],
            quality_score=quality_score,
            lessons=["Action completed" if success else "Action failed — retry with different strategy"],
            improvements=["Continue" if success else "Consider alternative approach"],
        )

        async with self._lock:  # MC2: concurrent dict/list protection
            self._reflection_history.append(reflection)
            self._state.recent_reflections.append({
                "action":    action[:100],
                "quality":   quality_score,
                "timestamp": reflection.timestamp,
            })

            # Cap at 100 (MC3: add Redis persistence later)
            if len(self._reflection_history) > 100:
                self._reflection_history = self._reflection_history[-100:]
            if len(self._state.recent_reflections) > 20:
                self._state.recent_reflections = self._state.recent_reflections[-20:]

            # Confidence calibration
            self._confidence_calibrations[action[:50]].append(quality_score)

            # Strategy adaptation — rolling avg of last 10
            recent_q = [r.quality_score for r in self._reflection_history[-10:]]
            if recent_q and sum(recent_q) / len(recent_q) < 0.5:
                self._strategy_adaptations.append({
                    "reason":     "Low rolling avg quality (<0.5) over last 10 actions",
                    "adaptation": "Switch to cautious, research-heavy approach",
                    "timestamp":  datetime.now(timezone.utc).isoformat(),
                })
                logger.warning(
                    "[Metacog] Strategy adaptation triggered — quality degraded"
                )

        return reflection

    # -------------------------------------------------------------------------
    # executive_prioritize — Eisenhower matrix
    # -------------------------------------------------------------------------

    async def executive_prioritize(self, tasks: List[Dict]) -> List[Dict]:
        """
        Prioritize tasks by urgency × importance (Eisenhower).
        MC5: mutates task dicts in-place — callers should deepcopy if needed.
        """
        for task in tasks:
            urgency    = task.get("urgency", 0.5)
            importance = task.get("importance", 0.5)
            deadline   = task.get("deadline_hours", 48)

            if urgency > 0.7 and importance > 0.7:
                priority, score = "critical", 1.0
            elif importance > 0.7:
                priority, score = "high", 0.8
            elif urgency > 0.7:
                priority, score = "urgent", 0.6
            else:
                priority, score = "normal", 0.4

            if deadline < 4:
                score += 0.3
            elif deadline < 12:
                score += 0.15

            task["priority"]       = priority
            task["priority_score"] = min(score, 1.0)

        return sorted(tasks, key=lambda t: t.get("priority_score", 0), reverse=True)

    # -------------------------------------------------------------------------
    # generate_creative_ideas — divergent thinking
    # -------------------------------------------------------------------------

    async def generate_creative_ideas(
        self, domain: str, count: int = 5
    ) -> List[CreativeIdea]:
        """MC4: uses _llm_callback which must be pool-backed."""
        if not self._llm_callback:
            return []

        prompt = (
            f"Generate {count} creative and novel ideas for: {domain}\n"
            f"Think divergently. Connect unrelated concepts.\n"
            f"Respond ONLY with JSON array (no markdown):\n"
            f'[{{"idea": "", "novelty": 0.0, "feasibility": 0.0, "domain_connections": []}}]'
        )
        ideas: List[CreativeIdea] = []
        try:
            resp = await self._llm_callback([{"role": "user", "content": prompt}], [])
            if resp and resp.get("content"):
                raw = resp["content"]
                s, e = raw.find("["), raw.rfind("]") + 1
                if s >= 0 and e > s:
                    data = json.loads(raw[s:e])
                    for item in data[:count]:
                        idea = CreativeIdea(
                            id=hashlib.md5(
                                f"idea:{item.get('idea','')}:{time.time()}".encode()
                            ).hexdigest()[:10],
                            idea=item.get("idea", ""),
                            novelty_score=float(item.get("novelty", 0.5)),
                            feasibility_score=float(item.get("feasibility", 0.5)),
                            domain=domain,
                            connections=item.get("domain_connections", []),
                        )
                        ideas.append(idea)
                        self._idea_bank.append(idea)
        except Exception as exc:
            logger.error(f"[Metacog] creative idea generation failed: {exc}")
        return ideas

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _assess_knowledge(self, action: str, context: Dict = None) -> KnowledgeState:
        """Check reflection history for similar past actions."""
        tokens = action.lower().split()[:5]
        similar = [
            r for r in self._reflection_history
            if any(t in r.action.lower() for t in tokens)
        ]
        if similar:
            avg = sum(r.quality_score for r in similar) / len(similar)
            if avg > 0.8:
                return KnowledgeState.KNOWN
            if avg > 0.5:
                return KnowledgeState.LIKELY_KNOWN
        return KnowledgeState.UNCERTAIN

    def _assess_confidence(self, knowledge: KnowledgeState) -> ConfidenceLevel:
        return {
            KnowledgeState.KNOWN:        ConfidenceLevel.HIGH,
            KnowledgeState.LIKELY_KNOWN: ConfidenceLevel.MEDIUM,
            KnowledgeState.UNCERTAIN:    ConfidenceLevel.LOW,
            KnowledgeState.UNKNOWN:      ConfidenceLevel.VERY_LOW,
            KnowledgeState.CONFLICTING:  ConfidenceLevel.LOW,
        }.get(knowledge, ConfidenceLevel.MEDIUM)

    def _recommend_approach(self) -> str:
        table = {
            (CognitiveMode.FOCUSED,     ConfidenceLevel.HIGH):   "Execute directly",
            (CognitiveMode.FOCUSED,     ConfidenceLevel.MEDIUM): "Execute with verification",
            (CognitiveMode.FOCUSED,     ConfidenceLevel.LOW):    "Research first, then execute",
            (CognitiveMode.CREATIVE,    ConfidenceLevel.HIGH):   "Explore bold directions",
            (CognitiveMode.CREATIVE,    ConfidenceLevel.MEDIUM): "Brainstorm with evaluation",
            (CognitiveMode.ANALYTICAL,  ConfidenceLevel.HIGH):   "Provide definitive analysis",
            (CognitiveMode.ANALYTICAL,  ConfidenceLevel.LOW):    "Gather data first",
            (CognitiveMode.EXECUTIVE,   ConfidenceLevel.HIGH):   "Decisively prioritize",
            (CognitiveMode.EXECUTIVE,   ConfidenceLevel.LOW):    "Consult council before deciding",
            (CognitiveMode.REFLECTIVE,  ConfidenceLevel.HIGH):   "Deep self-review",
            (CognitiveMode.REFLECTIVE,  ConfidenceLevel.LOW):    "Identify knowledge gaps",
            (CognitiveMode.EXPLORATORY, ConfidenceLevel.MEDIUM): "Search broadly, synthesize",
        }
        return table.get(
            (self._state.mode, self._state.confidence),
            "Proceed with caution and verification",
        )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def get_state(self) -> Dict:
        return {
            "mode":                 self._state.mode.value,
            "confidence":           self._state.confidence.value,
            "knowledge_state":      self._state.knowledge_state.value,
            "attention_focus":      self._state.attention_focus,
            "recent_reflections":   len(self._state.recent_reflections),
            "total_reflections":    len(self._reflection_history),
            "creative_ideas":       len(self._idea_bank),
            "strategy_adaptations": len(self._strategy_adaptations),
            "emotional_valence":    self._state.emotional_valence,
            "urgency_level":        self._state.urgency_level,
            "recommendation":       self._recommend_approach(),
        }


# Singleton
_metacog: Optional[MetacognitionEngine] = None


def get_metacognition() -> MetacognitionEngine:
    global _metacog
    if _metacog is None:
        _metacog = MetacognitionEngine()
    return _metacog
