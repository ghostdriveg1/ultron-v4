"""
packages/brain/planner.py

Ultron V4 — Hierarchical Task Planner
======================================
Sits ABOVE TaskDispatcher. Decomposes complex goals into an ordered subtask
DAG, drives TaskDispatcher per subtask, aggregates results into a coherent
final response.

Inspired by:
  - OpenHands PlannerAgent      : goal → subtask list, re-plan on failure
  - HuggingGPT / TaskMatrix     : task dependency graph, parallel-safe ordering
  - AutoGPT task_storage        : append-only task queue, status tracking
  - browser-use multi-tab mode  : parallel independent subtask execution

Design rules:
  - Decompose via single Groq classify call (≤300 tokens) — NOT a full ReAct loop
  - Max 6 subtasks per plan (free-tier token budget)
  - Subtasks marked 'parallel' run via asyncio.gather; 'sequential' run in order
  - Each subtask dispatched through existing TaskDispatcher (single LLM call layer)
  - On subtask failure: inject error as context for next subtask, continue (degrade)
  - Plan stored in Redis (key: ultron:plan:{channel_id}) for mid-session resume
  - PlannerAgent.run() returns Discord-safe string (≤1900 chars)
  - No new HTTP calls — all LLM via llm_router through TaskDispatcher

Future bug risks (pre-registered):
  PL1 [HIGH]  Decomposer LLM returns invalid JSON → fallback to single-task mode.
              Risk: complex tasks silently downgrade without user notice.
  PL2 [HIGH]  Parallel subtasks share channel_id Redis state → state collision.
              Fix: suffix channel_id with subtask index for parallel runs.
  PL3 [MED]   Subtask dependency declared but circular → infinite wait.
              Fix: topological sort with cycle detection before execution.
  PL4 [MED]   Aggregate response exceeds 1900 chars → truncation cuts mid-sentence.
              Fix: per-subtask 300-char summary before aggregation.
  PL5 [LOW]   Redis plan key not set with TTL → accumulates. Fix: TTL=7200.
  PL6 [LOW]   If TaskDispatcher singleton pool=None (no credentials yet) → all
              subtasks return stub responses silently. Acceptable for dev mode.

Tool calls used this session:
  Github:get_file_contents x3 (react_loop.py, task_dispatcher.py, packages tree)
  Github:push_files x1
  Notion:notion-fetch x1
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from packages.brain.task_dispatcher import TaskDispatcher, get_dispatcher
from packages.brain.llm_router import make_provider_llm_fn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SUBTASKS = 6                   # hard cap — free-tier token budget
REDIS_PLAN_PREFIX = "ultron:plan:" # key: ultron:plan:{channel_id}
REDIS_PLAN_TTL = 7200              # 2h — bug PL5 mitigation
SUBTASK_SUMMARY_MAX = 300         # chars per subtask result in aggregate — bug PL4
DECOMPOSE_TIMEOUT = 20.0          # seconds max for decompose LLM call


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SubtaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"  # order-dependent subtasks
    PARALLEL = "parallel"      # independent subtasks safe to gather()


@dataclass
class Subtask:
    """Single unit of a decomposed plan."""
    index: int
    description: str                        # natural language task for TaskDispatcher
    mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    depends_on: list[int] = field(default_factory=list)  # subtask indexes
    status: SubtaskStatus = SubtaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "description": self.description,
            "mode": self.mode.value,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class Plan:
    """Ordered collection of subtasks for a goal."""
    goal: str
    subtasks: list[Subtask] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    channel_id: str = ""
    user_id: str = ""

    @property
    def is_complete(self) -> bool:
        return all(
            st.status in (SubtaskStatus.DONE, SubtaskStatus.FAILED, SubtaskStatus.SKIPPED)
            for st in self.subtasks
        )

    @property
    def failed_count(self) -> int:
        return sum(1 for st in self.subtasks if st.status == SubtaskStatus.FAILED)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "created_at": self.created_at,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
        }


# ---------------------------------------------------------------------------
# Decomposer — single Groq classify call, NOT a ReAct loop
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM_PROMPT = """You are a task planner. Given a complex user goal, decompose it into
at most {max_subtasks} concrete subtasks that Ultron can execute sequentially.

Rules:
- Each subtask must be a complete, self-contained instruction.
- Mark subtasks as 'parallel' ONLY if they are truly independent (e.g., search A and search B).
- Return ONLY valid JSON. No prose. No markdown.
- If goal is simple (single action), return exactly 1 subtask.

JSON schema:
{{
  "subtasks": [
    {{"index": 0, "description": "...", "mode": "sequential", "depends_on": []}},
    {{"index": 1, "description": "...", "mode": "sequential", "depends_on": [0]}},
    ...
  ]
}}"""


async def decompose_goal(
    goal: str,
    llm_call_fn: Any,
    max_subtasks: int = MAX_SUBTASKS,
) -> list[Subtask]:
    """Call LLM once to decompose goal → list[Subtask].

    Falls back to single-task plan on any failure (bug PL1 mitigation).
    """
    system_prompt = DECOMPOSE_SYSTEM_PROMPT.format(max_subtasks=max_subtasks)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Goal: {goal}"},
    ]

    raw: Optional[Any] = None
    try:
        async with asyncio.timeout(DECOMPOSE_TIMEOUT):
            raw = await llm_call_fn(messages=messages, tools=[])
    except asyncio.TimeoutError:
        logger.warning("[Planner] decompose LLM timed out — fallback single task")
        return [Subtask(index=0, description=goal)]
    except Exception as exc:
        logger.error(f"[Planner] decompose LLM failed: {exc} — fallback single task")
        return [Subtask(index=0, description=goal)]

    # Parse response
    try:
        if isinstance(raw, dict):
            content = raw.get("content", "{}")
        else:
            content = str(raw)

        if isinstance(content, str):
            content = content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

        parsed = json.loads(content) if isinstance(content, str) else content
        raw_subtasks = parsed.get("subtasks", [])

        if not raw_subtasks:
            logger.warning("[Planner] decompose returned empty subtasks — fallback")
            return [Subtask(index=0, description=goal)]

        subtasks = []
        for i, st in enumerate(raw_subtasks[:max_subtasks]):
            subtasks.append(Subtask(
                index=i,
                description=st.get("description", goal),
                mode=ExecutionMode(st.get("mode", "sequential")),
                depends_on=st.get("depends_on", []),
            ))

        logger.info(f"[Planner] decomposed into {len(subtasks)} subtasks")
        return subtasks

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(f"[Planner] decompose parse failed: {exc} — fallback single task")
        return [Subtask(index=0, description=goal)]


# ---------------------------------------------------------------------------
# Cycle detection for dependency DAG
# ---------------------------------------------------------------------------

def _has_cycle(subtasks: list[Subtask]) -> bool:
    """Topological sort cycle detection — bug PL3 mitigation."""
    graph: dict[int, list[int]] = {st.index: st.depends_on for st in subtasks}
    visited: set[int] = set()
    in_stack: set[int] = set()

    def dfs(node: int) -> bool:
        visited.add(node)
        in_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in in_stack:
                return True
        in_stack.discard(node)
        return False

    for st in subtasks:
        if st.index not in visited:
            if dfs(st.index):
                return True
    return False


# ---------------------------------------------------------------------------
# Executor — drives TaskDispatcher per subtask
# ---------------------------------------------------------------------------

async def _execute_subtask(
    subtask: Subtask,
    plan: Plan,
    dispatcher: TaskDispatcher,
    prior_context: str,
) -> None:
    """Execute a single subtask via TaskDispatcher. Mutates subtask in-place."""
    subtask.status = SubtaskStatus.RUNNING
    subtask.started_at = time.time()

    # For parallel subtasks: suffix channel_id to avoid Redis state collision (bug PL2)
    channel_id = (
        f"{plan.channel_id}_pl{subtask.index}"
        if subtask.mode == ExecutionMode.PARALLEL
        else plan.channel_id
    )

    context = prior_context
    if subtask.depends_on:
        dep_results = [
            plan.subtasks[i].result or ""
            for i in subtask.depends_on
            if i < len(plan.subtasks)
        ]
        if dep_results:
            context = "\n".join(dep_results) + "\n" + context

    try:
        result_str = await dispatcher.dispatch(
            message=subtask.description,
            channel_id=channel_id,
            user_id=plan.user_id,
            context=context,
        )
        subtask.result = result_str[:SUBTASK_SUMMARY_MAX]
        subtask.status = SubtaskStatus.DONE
        logger.info(f"[Planner] subtask {subtask.index} DONE")
    except Exception as exc:
        logger.error(f"[Planner] subtask {subtask.index} FAILED: {exc}")
        subtask.error = str(exc)[:200]
        subtask.status = SubtaskStatus.FAILED
    finally:
        subtask.finished_at = time.time()


async def execute_plan(
    plan: Plan,
    dispatcher: TaskDispatcher,
    initial_context: str = "",
) -> None:
    """Execute all subtasks in plan. Sequential by default; gather parallel groups."""
    # Cycle guard — bug PL3
    if _has_cycle(plan.subtasks):
        logger.error("[Planner] dependency cycle detected — falling back to sequential")
        for st in plan.subtasks:
            st.depends_on = []

    executed: set[int] = set()
    prior_context = initial_context

    # Group subtasks by execution wave: all subtasks whose deps are already executed
    while True:
        ready = [
            st for st in plan.subtasks
            if st.status == SubtaskStatus.PENDING
            and all(dep in executed for dep in st.depends_on)
        ]
        if not ready:
            # Check if any pending remain (blocked deps)
            still_pending = [st for st in plan.subtasks if st.status == SubtaskStatus.PENDING]
            if still_pending:
                logger.warning(
                    f"[Planner] {len(still_pending)} subtasks blocked — marking skipped"
                )
                for st in still_pending:
                    st.status = SubtaskStatus.SKIPPED
            break

        parallel_batch = [st for st in ready if st.mode == ExecutionMode.PARALLEL]
        sequential_batch = [st for st in ready if st.mode == ExecutionMode.SEQUENTIAL]

        # Run parallel batch concurrently
        if parallel_batch:
            await asyncio.gather(*[
                _execute_subtask(st, plan, dispatcher, prior_context)
                for st in parallel_batch
            ])
            for st in parallel_batch:
                executed.add(st.index)

        # Run sequential batch one by one
        for st in sequential_batch:
            await _execute_subtask(st, plan, dispatcher, prior_context)
            executed.add(st.index)
            # Feed result as context for next sequential step
            if st.result:
                prior_context = st.result + "\n" + prior_context


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def aggregate_results(plan: Plan) -> str:
    """Build Discord-safe response from completed plan."""
    parts: list[str] = []

    done_tasks = [st for st in plan.subtasks if st.status == SubtaskStatus.DONE and st.result]
    failed_tasks = [st for st in plan.subtasks if st.status == SubtaskStatus.FAILED]

    if len(plan.subtasks) == 1:
        # Simple pass-through — no multi-task header
        return done_tasks[0].result if done_tasks else "Task failed: " + (failed_tasks[0].error or "unknown error")

    for st in plan.subtasks:
        if st.status == SubtaskStatus.DONE and st.result:
            label = f"**[{st.index + 1}]** {st.description[:60]}"
            result_snippet = st.result[:SUBTASK_SUMMARY_MAX]
            parts.append(f"{label}\n{result_snippet}")
        elif st.status == SubtaskStatus.FAILED:
            parts.append(f"**[{st.index + 1}] FAILED** {st.description[:50]}: {st.error or 'unknown'}")
        elif st.status == SubtaskStatus.SKIPPED:
            parts.append(f"**[{st.index + 1}] SKIPPED** (blocked dependency)")

    if failed_tasks:
        parts.append(f"\n⚠ {len(failed_tasks)}/{len(plan.subtasks)} subtasks failed.")

    raw = "\n\n".join(parts)
    # Bug PL4 mitigation — hard truncate
    return raw[:1900] if len(raw) > 1900 else raw


# ---------------------------------------------------------------------------
# Redis plan persistence
# ---------------------------------------------------------------------------

async def _save_plan(redis: Any, plan: Plan) -> None:
    if redis is None:
        return
    try:
        key = f"{REDIS_PLAN_PREFIX}{plan.channel_id}"
        await redis.set(key, json.dumps(plan.to_dict()), ex=REDIS_PLAN_TTL)
    except Exception as exc:
        logger.warning(f"[Planner] Redis plan save failed: {exc}")


async def _load_plan(redis: Any, channel_id: str) -> Optional[dict]:
    if redis is None:
        return None
    try:
        key = f"{REDIS_PLAN_PREFIX}{channel_id}"
        raw = await redis.get(key)
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.warning(f"[Planner] Redis plan load failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# PlannerAgent — public interface
# ---------------------------------------------------------------------------

class PlannerAgent:
    """Top-level entry point for complex multi-step goals.

    Simple tasks (1 step) pass straight through to TaskDispatcher.
    Complex tasks get decomposed into a Plan → executed → aggregated.

    Usage::
        agent = PlannerAgent(pool=key_pool, redis=redis_client)
        response = await agent.run(
            goal="Research quantum computing breakthroughs this week and summarize top 3",
            channel_id="12345",
            user_id="ghost_uid",
        )
    """

    # Goals with ≤ this many tokens skip decompose call (saves one LLM call)
    SIMPLE_GOAL_WORD_THRESHOLD = 12

    def __init__(
        self,
        pool: Any = None,
        redis: Any = None,
        max_subtasks: int = MAX_SUBTASKS,
    ) -> None:
        self.pool = pool
        self.redis = redis
        self.max_subtasks = max_subtasks
        self._dispatcher = get_dispatcher(pool=pool, redis=redis)

    async def run(
        self,
        goal: str,
        channel_id: str,
        user_id: str,
        initial_context: str = "",
    ) -> str:
        """Decompose goal → execute plan → return Discord-safe string."""
        # Short-circuit: simple goals skip decompose overhead
        if len(goal.split()) <= self.SIMPLE_GOAL_WORD_THRESHOLD:
            logger.info("[Planner] simple goal — bypassing decompose")
            return await self._dispatcher.dispatch(
                message=goal,
                channel_id=channel_id,
                user_id=user_id,
                context=initial_context,
            )

        # Decompose
        llm_call_fn = await make_provider_llm_fn(self.pool)
        subtasks = await decompose_goal(
            goal=goal,
            llm_call_fn=llm_call_fn,
            max_subtasks=self.max_subtasks,
        )

        # Single subtask after decompose = still simple
        if len(subtasks) == 1:
            return await self._dispatcher.dispatch(
                message=subtasks[0].description,
                channel_id=channel_id,
                user_id=user_id,
                context=initial_context,
            )

        plan = Plan(
            goal=goal,
            subtasks=subtasks,
            channel_id=channel_id,
            user_id=user_id,
        )

        await _save_plan(self.redis, plan)
        logger.info(f"[Planner] executing plan: {len(subtasks)} subtasks for goal='{goal[:60]}'")

        await execute_plan(plan, self._dispatcher, initial_context)
        await _save_plan(self.redis, plan)  # persist final state

        return aggregate_results(plan)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_planner_instance: Optional[PlannerAgent] = None


def get_planner(pool: Any = None, redis: Any = None) -> PlannerAgent:
    """Return singleton PlannerAgent. Call from main.py lifespan."""
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = PlannerAgent(pool=pool, redis=redis)
        logger.info("[Planner] Singleton created")
    return _planner_instance
