"""
packages/brain/react_loop.py

Ultron V4 — ReAct Agentic Loop Engine
======================================
THINK → ACT → OBSERVE → REFLECT → repeat (max N iterations)

Informed by:
  - browser-use/browser-use  : ActionLoopDetector, MessageCompaction, flash_mode, AgentOutput schema
  - OpenHands/codeact_agent  : pending_actions deque, condenser pattern, tool dispatch via function_calling

Design rules:
  - flash_mode = True by default for Groq  (memory + action fields only, strips thinking/eval/next_goal)
  - ActionLoopDetector: rolling hash window 20 steps, SOFT nudge at 5/8/12 repeats (never blocks)
  - MessageCompaction: compact every 25 steps, keep_last_items=6, summary_max_chars=6000
  - Hard ceiling: max_iterations=5 default, absolute max=10 — prevents runaway on free tier
  - Tool registry: async callables registered by name; tool_result fed into next LLM call
  - AgentState fully tracked in-memory; Redis persistence wired externally (task_dispatcher.py)
  - No vision for Groq (DOM/text only, 10x token saving per browser-use pattern)

Future bug risks (pre-registered):
  B1 [HIGH]  task_dispatcher.py resets n_steps per Discord msg → loop detector window clears → infinite loops undetected
  B2 [HIGH]  discord_bot.py splits long msg into chunks → consecutive_failures resets mid-task
  B3 [MED]   key_rotation/pool.py switches provider mid-loop → new provider rejects flash_mode schema → ValidationError
  B4 [MED]   memory write fires AFTER loop → crash at iter 3/5 = no memory write ever triggers
  B5 [LOW]   MessageCompaction fires during Zilliz batch embed → compacted summary != embedded text → retrieval mismatch
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_ITERATIONS: int = 5
ABSOLUTE_MAX_ITERATIONS: int = 10      # hard ceiling, never exceed on free tier
DEFAULT_MAX_FAILURES: int = 3          # consecutive LLM failures before abort
LOOP_DETECTION_WINDOW: int = 20        # rolling action hash window (browser-use pattern)
LOOP_NUDGE_THRESHOLDS: tuple = (5, 8, 12)  # soft nudge counts (NEVER hard block)
COMPACT_EVERY_N_STEPS: int = 25       # MessageCompaction: compact frequency
COMPACT_KEEP_LAST: int = 6            # MessageCompaction: keep last N messages verbatim
COMPACT_SUMMARY_MAX_CHARS: int = 6000  # MessageCompaction: summary char cap
GROQ_CONTEXT_LIMIT: int = 7500        # safety margin below 8k Groq ctx limit


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LoopStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    MAX_ITERATIONS = "max_iterations"
    MAX_FAILURES = "max_failures"
    PAUSED = "paused"
    STOPPED = "stopped"


class ActionType(str, Enum):
    SEARCH = "search"
    CODE_EXEC = "code_exec"
    BROWSER_FETCH = "browser_fetch"
    FILE_READ = "file_read"
    COMPUTER_USE = "computer_use"
    DONE = "done"
    THINK = "think"      # pure reasoning, no tool call (OpenHands ThinkTool pattern)


# ---------------------------------------------------------------------------
# Data models  (ported + simplified from browser-use/agent/views.py)
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """Structured result from a tool execution.

    Directly mirrors browser-use ActionResult.
    Rule: always check results[-1] for is_done — NEVER results[0].
    """
    is_done: bool = False
    success: bool = True
    error: Optional[str] = None
    extracted_content: Optional[str] = None
    long_term_memory: Optional[str] = None  # snippet worth persisting to Tier2
    tool_name: Optional[str] = None
    raw_output: Optional[Any] = None

    def to_prompt_str(self) -> str:
        """Compact string representation injected into next LLM observation."""
        if self.error:
            return f"[TOOL ERROR] {self.tool_name}: {self.error}"
        if self.extracted_content:
            return f"[TOOL RESULT] {self.tool_name}: {self.extracted_content[:2000]}"
        return f"[TOOL OK] {self.tool_name}: completed"


@dataclass
class AgentOutput:
    """LLM response schema for full mode (non-Groq or non-flash).

    All four fields REQUIRED. Missing any = validation failure.
    """
    thinking: str = ""           # chain-of-thought (stripped in flash_mode)
    eval_prev_goal: str = ""     # reflection on previous step
    memory: str = ""             # running summary of important findings
    next_goal: str = ""          # explicit goal for next action
    action_type: str = ""        # one of ActionType values
    action_params: dict = field(default_factory=dict)

    @classmethod
    def from_groq_flash(cls, raw: dict) -> "AgentOutput":
        """Parse flash_mode response: {memory, action_type, action_params} only.

        Groq returns only memory + action fields. Full AgentOutput parser
        would fail on missing thinking/eval/next_goal. Lock to flash schema.
        Pre-registered bug B3: if provider switches mid-loop, schema mismatch here.
        """
        return cls(
            thinking="",
            eval_prev_goal="",
            memory=raw.get("memory", ""),
            next_goal="",
            action_type=raw.get("action_type", ActionType.DONE.value),
            action_params=raw.get("action_params", {}),
        )

    @classmethod
    def from_full_response(cls, raw: dict) -> "AgentOutput":
        """Parse full (non-flash) response schema."""
        return cls(
            thinking=raw.get("thinking", ""),
            eval_prev_goal=raw.get("eval_prev_goal", ""),
            memory=raw.get("memory", ""),
            next_goal=raw.get("next_goal", ""),
            action_type=raw.get("action_type", ActionType.DONE.value),
            action_params=raw.get("action_params", {}),
        )

    def is_done(self) -> bool:
        return self.action_type == ActionType.DONE.value


@dataclass
class AgentState:
    """Mutable loop state. Tracks all iteration bookkeeping.

    Mirrored from browser-use AgentState.
    Stored in-memory; task_dispatcher.py is responsible for Redis persistence.
    Pre-registered bug B1: if task_dispatcher resets n_steps on each Discord
    message, loop detector window clears → infinite loops undetected.
    """
    task: str = ""
    n_steps: int = 0
    consecutive_failures: int = 0
    paused: bool = False
    stopped: bool = False
    results: list[ActionResult] = field(default_factory=list)
    message_history: list[dict] = field(default_factory=list)  # LLM message dicts
    running_memory: str = ""     # accumulated memory string from AgentOutput.memory
    status: LoopStatus = LoopStatus.RUNNING
    started_at: float = field(default_factory=time.time)

    def last_result(self) -> Optional[ActionResult]:
        """Always return results[-1], never results[0]. Bug B_done_detection."""
        return self.results[-1] if self.results else None

    def is_complete(self) -> bool:
        last = self.last_result()
        return last.is_done if last else False


# ---------------------------------------------------------------------------
# ActionLoopDetector  (ported from browser-use/agent/views.py)
# ---------------------------------------------------------------------------

class ActionLoopDetector:
    """Detects repeated action patterns using rolling hash window.

    Soft nudge only — NEVER blocks execution.
    Nudge thresholds: 5, 8, 12 repeats (matches browser-use defaults).
    Window size: 20 actions.
    """

    def __init__(
        self,
        window: int = LOOP_DETECTION_WINDOW,
        nudge_thresholds: tuple = LOOP_NUDGE_THRESHOLDS,
    ) -> None:
        self.window = window
        self.nudge_thresholds = nudge_thresholds
        self._hashes: deque[str] = deque(maxlen=window)
        self._nudge_counts: dict[str, int] = {}

    def _hash_action(self, action_type: str, action_params: dict) -> str:
        """SHA256 of action_type + sorted params JSON."""
        payload = json.dumps(
            {"t": action_type, "p": action_params}, sort_keys=True
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def check(self, action_type: str, action_params: dict) -> Optional[str]:
        """Record action. Returns nudge message string if threshold hit, else None.

        Call BEFORE executing the action. Inject returned nudge into next LLM prompt.
        """
        h = self._hash_action(action_type, action_params)
        self._hashes.append(h)
        self._nudge_counts[h] = self._nudge_counts.get(h, 0) + 1
        count = self._nudge_counts[h]

        for threshold in self.nudge_thresholds:
            if count == threshold:
                logger.warning(
                    f"[LoopDetector] Action '{action_type}' repeated {count}x — injecting nudge"
                )
                return (
                    f"[LOOP WARNING] You have performed '{action_type}' {count} times "
                    f"with similar parameters. Consider a different approach or "
                    f"call DONE if the task cannot be completed."
                )
        return None

    def reset(self) -> None:
        self._hashes.clear()
        self._nudge_counts.clear()


# ---------------------------------------------------------------------------
# MessageCompaction  (ported from browser-use MessageCompactionSettings)
# ---------------------------------------------------------------------------

class MessageCompactor:
    """Compacts message history every N steps to stay within Groq 8k context.

    Strategy: summarize messages[:-keep_last] to a single summary string,
    keep last K messages verbatim, rebuild history as [system, summary, *last_K].
    """

    def __init__(
        self,
        compact_every: int = COMPACT_EVERY_N_STEPS,
        keep_last: int = COMPACT_KEEP_LAST,
        summary_max_chars: int = COMPACT_SUMMARY_MAX_CHARS,
    ) -> None:
        self.compact_every = compact_every
        self.keep_last = keep_last
        self.summary_max_chars = summary_max_chars

    def should_compact(self, n_steps: int) -> bool:
        return n_steps > 0 and n_steps % self.compact_every == 0

    async def compact(
        self,
        messages: list[dict],
        summarizer_fn: Optional[Callable] = None,
    ) -> list[dict]:
        """Compact messages. summarizer_fn = async (text) -> str.

        If no summarizer provided, truncate older messages to summary_max_chars.
        Keep system message + last keep_last messages verbatim.
        Pre-registered bug B5: if Zilliz embed fires during compact, summary
        content diverges from what was embedded → retrieval mismatch.
        """
        if len(messages) <= self.keep_last + 1:  # +1 for system msg
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        older = non_system[: -self.keep_last] if self.keep_last else non_system
        recent = non_system[-self.keep_last :] if self.keep_last else []

        # Build text blob for older messages
        older_text = "\n".join(
            f"[{m.get('role','?')}]: {str(m.get('content',''))[:500]}"
            for m in older
        )

        if summarizer_fn:
            try:
                summary_text = await summarizer_fn(older_text)
                summary_text = summary_text[: self.summary_max_chars]
            except Exception as exc:
                logger.warning(f"[Compactor] summarizer failed: {exc} — truncating")
                summary_text = older_text[: self.summary_max_chars]
        else:
            summary_text = older_text[: self.summary_max_chars]

        summary_msg = {
            "role": "user",
            "content": f"[COMPACTED HISTORY SUMMARY]\n{summary_text}",
        }

        compacted = system_msgs + [summary_msg] + recent
        logger.info(
            f"[Compactor] {len(messages)} → {len(compacted)} messages after compaction"
        )
        return compacted


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Pluggable async tool registry.

    Register tools by name. ReActLoop calls execute(name, params) each iteration.
    Tool functions must be async and return ActionResult.

    Groq function_calling: tool schemas registered here are also exposed as
    Groq tool definitions in the LLM call. See build_groq_tool_schemas().
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}  # Groq-compatible JSON schema per tool

    def register(
        self,
        name: str,
        fn: Callable,
        schema: Optional[dict] = None,
    ) -> None:
        """Register a tool. fn must be async (params: dict) -> ActionResult."""
        self._tools[name] = fn
        if schema:
            self._schemas[name] = schema
        logger.debug(f"[ToolRegistry] registered: {name}")

    async def execute(self, name: str, params: dict) -> ActionResult:
        """Execute a registered tool. Returns error ActionResult if not found."""
        if name not in self._tools:
            logger.error(f"[ToolRegistry] unknown tool: {name}")
            return ActionResult(
                success=False,
                error=f"Tool '{name}' not registered. Available: {list(self._tools.keys())}",
                tool_name=name,
            )
        try:
            result = await self._tools[name](params)
            result.tool_name = name
            return result
        except Exception as exc:
            logger.exception(f"[ToolRegistry] tool '{name}' raised: {exc}")
            return ActionResult(
                success=False,
                error=str(exc),
                tool_name=name,
            )

    def build_groq_tool_schemas(self) -> list[dict]:
        """Return list of Groq-compatible tool schema dicts for function_calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema.get("description", name),
                    "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for name, schema in self._schemas.items()
        ]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


# ---------------------------------------------------------------------------
# Flash-mode Groq system prompt builder
# ---------------------------------------------------------------------------

FLASH_SYSTEM_PROMPT_TEMPLATE = """You are Ultron, a powerful AI agent. Complete the task step by step.

You MUST respond ONLY with a valid JSON object — no prose, no markdown fences.

Response schema (flash_mode=True for Groq):
{{
  "memory": "<running summary of findings so far, max 500 chars>",
  "action_type": "<one of: {tool_names}, done>",
  "action_params": {{<tool-specific params>}}
}}

Available tools: {tool_names}
Task: {task}
Iteration: {iteration}/{max_iterations}

{nudge_message}
{observation}
"""


def _build_flash_prompt(
    task: str,
    tool_names: list[str],
    iteration: int,
    max_iterations: int,
    observation: str = "",
    nudge_message: str = "",
) -> str:
    return FLASH_SYSTEM_PROMPT_TEMPLATE.format(
        task=task,
        tool_names=", ".join(tool_names),
        iteration=iteration,
        max_iterations=max_iterations,
        nudge_message=nudge_message,
        observation=observation,
    )


# ---------------------------------------------------------------------------
# ReActLoop — main engine
# ---------------------------------------------------------------------------

class ReActLoop:
    """ReAct agentic loop engine for Ultron V4.

    Usage::
        registry = ToolRegistry()
        registry.register("search", my_search_fn, schema={...})

        loop = ReActLoop(
            llm_call_fn=groq_call,   # async (messages, tools) -> dict
            tool_registry=registry,
            flash_mode=True,         # Groq path
            max_iterations=5,
        )
        result = await loop.run(task="Find the price of NVDA stock")
        print(result.extracted_content)
    """

    def __init__(
        self,
        llm_call_fn: Callable,           # async (messages: list[dict], tools: list[dict]) -> dict
        tool_registry: ToolRegistry,
        flash_mode: bool = True,          # True = Groq fast path (memory+action only)
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_failures: int = DEFAULT_MAX_FAILURES,
        summarizer_fn: Optional[Callable] = None,  # for MessageCompactor
    ) -> None:
        if max_iterations > ABSOLUTE_MAX_ITERATIONS:
            logger.warning(
                f"[ReActLoop] max_iterations {max_iterations} > absolute max "
                f"{ABSOLUTE_MAX_ITERATIONS}. Clamping."
            )
            max_iterations = ABSOLUTE_MAX_ITERATIONS

        self.llm_call_fn = llm_call_fn
        self.tools = tool_registry
        self.flash_mode = flash_mode
        self.max_iterations = max_iterations
        self.max_failures = max_failures
        self.summarizer_fn = summarizer_fn

        self.loop_detector = ActionLoopDetector()
        self.compactor = MessageCompactor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, task: str, initial_context: str = "") -> ActionResult:
        """Run the ReAct loop for a given task. Returns final ActionResult.

        Pre-registered bug B4: memory write fires AFTER this returns.
        If loop raises before returning, caller must catch + trigger memory write.
        """
        state = AgentState(task=task)
        self.loop_detector.reset()

        # Build initial messages
        state.message_history = self._build_initial_messages(
            task=task,
            context=initial_context,
        )

        while state.n_steps < self.max_iterations:
            if state.paused:
                logger.info("[ReActLoop] paused — waiting")
                await asyncio.sleep(0.5)
                continue
            if state.stopped:
                state.status = LoopStatus.STOPPED
                break

            # Compact if needed
            if self.compactor.should_compact(state.n_steps):
                state.message_history = await self.compactor.compact(
                    state.message_history,
                    summarizer_fn=self.summarizer_fn,
                )

            # LLM call
            agent_output = await self._llm_step(state)
            if agent_output is None:
                state.consecutive_failures += 1
                logger.warning(
                    f"[ReActLoop] LLM step failed ({state.consecutive_failures}/{self.max_failures})"
                )
                if state.consecutive_failures >= self.max_failures:
                    state.status = LoopStatus.MAX_FAILURES
                    break
                continue

            state.consecutive_failures = 0
            state.n_steps += 1
            state.running_memory = agent_output.memory or state.running_memory

            # Check if done
            if agent_output.is_done():
                final_result = ActionResult(
                    is_done=True,
                    success=True,
                    extracted_content=agent_output.memory,
                    tool_name="done",
                )
                state.results.append(final_result)
                state.status = LoopStatus.DONE
                logger.info(
                    f"[ReActLoop] DONE after {state.n_steps} steps: "
                    f"{agent_output.memory[:100]}"
                )
                break

            # Loop detection — inject nudge if threshold hit
            nudge = self.loop_detector.check(
                agent_output.action_type, agent_output.action_params
            )

            # Execute tool
            tool_result = await self.tools.execute(
                agent_output.action_type, agent_output.action_params
            )
            state.results.append(tool_result)

            # Build observation for next LLM call
            observation = tool_result.to_prompt_str()
            if nudge:
                observation = nudge + "\n" + observation

            self._append_observation(state, observation)

        else:
            # Loop exhausted max_iterations without done
            state.status = LoopStatus.MAX_ITERATIONS
            logger.warning(
                f"[ReActLoop] max_iterations ({self.max_iterations}) reached — forcing done"
            )

        # Return last result or synthetic done
        final = state.last_result()
        if final is None:
            final = ActionResult(
                is_done=True,
                success=False,
                error=f"Loop ended with status={state.status.value}, no results",
            )
        return final

    def pause(self, state: AgentState) -> None:
        state.paused = True
        state.status = LoopStatus.PAUSED

    def resume(self, state: AgentState) -> None:
        state.paused = False
        state.status = LoopStatus.RUNNING

    def stop(self, state: AgentState) -> None:
        state.stopped = True
        state.status = LoopStatus.STOPPED

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_initial_messages(self, task: str, context: str) -> list[dict]:
        """Build initial message list for LLM.

        flash_mode: system prompt encodes flash JSON schema.
        full mode: system prompt encodes full AgentOutput schema.
        """
        if self.flash_mode:
            system_content = _build_flash_prompt(
                task=task,
                tool_names=self.tools.tool_names,
                iteration=0,
                max_iterations=self.max_iterations,
            )
        else:
            system_content = (
                f"You are Ultron, a powerful AI agent. Task: {task}\n"
                f"Tools available: {', '.join(self.tools.tool_names)}\n"
                f"Respond with JSON: {{thinking, eval_prev_goal, memory, next_goal, "
                f"action_type, action_params}}"
            )

        messages: list[dict] = [{"role": "system", "content": system_content}]
        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})
        messages.append({"role": "user", "content": f"Begin. Task: {task}"})
        return messages

    async def _llm_step(self, state: AgentState) -> Optional[AgentOutput]:
        """Single LLM call. Returns parsed AgentOutput or None on failure.

        Pre-registered bug B3: if key_rotation switches provider mid-loop,
        flash_mode schema may not match new provider's expectations.
        """
        tools_schema = self.tools.build_groq_tool_schemas()
        try:
            raw_response = await self.llm_call_fn(
                messages=state.message_history,
                tools=tools_schema,
            )
        except Exception as exc:
            logger.error(f"[ReActLoop] LLM call failed: {exc}")
            return None

        try:
            # raw_response expected: {"content": "{...json...}"} or parsed dict
            if isinstance(raw_response, str):
                parsed = json.loads(raw_response)
            elif isinstance(raw_response, dict):
                content = raw_response.get("content", "{}")
                if isinstance(content, str):
                    # Strip markdown fences if present
                    content = content.strip().lstrip("```json").lstrip("```").rstrip("```")
                    parsed = json.loads(content)
                else:
                    parsed = content
            else:
                logger.error(f"[ReActLoop] unexpected LLM response type: {type(raw_response)}")
                return None

            if self.flash_mode:
                return AgentOutput.from_groq_flash(parsed)
            else:
                return AgentOutput.from_full_response(parsed)

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error(f"[ReActLoop] response parse failed: {exc} | raw={str(raw_response)[:300]}")
            return None

    def _append_observation(self, state: AgentState, observation: str) -> None:
        """Append tool observation to message history as user turn."""
        state.message_history.append({
            "role": "user",
            "content": f"[OBSERVATION] {observation}\n[Step {state.n_steps}/{self.max_iterations}]",
        })


# ---------------------------------------------------------------------------
# Module-level default registry (for simple usage without DI)
# ---------------------------------------------------------------------------

_default_registry = ToolRegistry()


def get_default_registry() -> ToolRegistry:
    """Return the shared default tool registry.

    Intended for simple use: import + register tools at module load.
    task_dispatcher.py should call this to wire search/code_exec/browser tools.
    """
    return _default_registry


def register_tool(
    name: str,
    fn: Callable,
    schema: Optional[dict] = None,
) -> None:
    """Convenience: register a tool in the default registry."""
    _default_registry.register(name, fn, schema)
