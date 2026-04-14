"""
packages/brain/key_rotation/pool.py

Ultron V4 — KeyPool: Dual Sub-Pool Key Rotation with Circuit Breaker
======================================================================
Architecture (v17 LOCKED — never re-debate):

  GENERAL POOL  → self.general[]
    Providers: Groq, Cerebras, Together, OpenRouter, Gemini (5 regular keys each)
    Used by:   main task, MOA, Council, coding agent, browser agent, voice, embeddings
    Selection: weighted round-robin + circuit breaker
    Weights:   Groq=3, Cerebras=3, Together=2, OpenRouter=2, Gemini=2
               (equal priority — goal is token EXHAUSTION across all providers)

  SENTINEL POOL → self.sentinel[]
    Provider:  Gemini 2.5 Pro ONLY (1 dedicated key, never shared)
    Used by:   Sentinel layer ONLY — routing decisions, incident detection, weekly audit
    Selection: returns the one key. Raises if tripped.

  KEY STRUCT:
    {
      "key_id":    str,        # unique, e.g. "groq_0", "gemini_sentinel"
      "key":       str,        # raw API key
      "provider":  str,        # "groq"|"cerebras"|"together"|"openrouter"|"gemini"
      "model":     str,        # model string for this key
      "pool_type": str,        # "general" or "sentinel"
      "failures":  int,        # current consecutive failure count
      "reset_at":  float,      # UNIX timestamp: key blocked until this time (0 = not tripped)
      "weight":    int,        # weighted RR weight (see PROVIDER_WEIGHTS)
    }

Circuit breaker thresholds (failures → cooldown duration):
  groq:                  3 failures → 1 hour cooldown
  gemini:                3 failures → 24 hour cooldown
  cerebras:              3 failures → 30 min cooldown
  together:              3 failures → 30 min cooldown
  openrouter:            3 failures → 30 min cooldown

Thread safety:
  asyncio.Lock on get_key(), get_sentinel_key(), report_success(), report_failure().
  No cross-coroutine race on failure counter.

State:
  Pure in-memory. Redis persistence is a future task (noted in gut-feel bugs).
  On process restart, all keys reset to 0 failures.

Future bug risks (pre-registered):
  P1 [HIGH]   No Redis persistence. If HF Space restarts mid-cooldown, tripped keys
              are instantly reset → burst retries → re-trip → exponential 429 storm.
              Fix (future): on report_failure, also write {key_id: reset_at} to Redis.
              On init, load reset_at from Redis to restore cooldown state.

  P2 [HIGH]   Weighted RR uses a simple modular index (self._rr_index % total_weight).
              If all high-weight keys (Groq x3, Cerebras x3) trip simultaneously,
              the remaining pool is Together/OpenRouter/Gemini (weight 2 each).
              Index may skip available keys if calculated position lands on a tripped slot.
              Fix: after weighted slot selection, linear scan forward to first available.
              (Already implemented in _select_weighted — but verify scan wraps correctly.)

  P3 [MED]    Sentinel key has no retry logic. Single failure → 24hr locked.
              If Ghost adds a second Gemini sentinel key later, the pool must support
              sentinel[] as a list and rotate within it. Current impl supports list
              but treats all as equally sentinel. Add priority ordering later.

  P4 [MED]    get_key() called without await in task_dispatcher or llm_router
              (copy-paste error) → TypeError: object dict can't be used in await.
              Fix: explicit asyncio.iscoroutinefunction checks or mypy annotations.
              Already annotated async def — still watch for it.

  P5 [LOW]    If config passes same key string under two different key_ids,
              both can trip independently but share the same API quota.
              Reset on one doesn't help the other — both still make calls
              until both trip. No dedup guard. Low priority — Ghost controls config.

  P6 [LOW]    _is_available() auto-resets reset_at to 0 when cooldown expires.
              If two coroutines call _is_available() simultaneously on the same key
              right as cooldown expires, both see it available and both call it,
              doubling the call rate briefly. asyncio.Lock on get_key() prevents this
              since _is_available() is only called inside the locked block.
              Confirmed safe — document it here for future maintainers.

Tool calls used writing this file:
  Github:get_file_contents x2 (repo root, packages/shared check),
  Github:push_files x1 (batch: exceptions.py + __init__.py + pool.py)
"""

from __future__ import annotations

import asyncio
import logging
import time
from copy import deepcopy
from typing import Any, Optional

from packages.shared.exceptions import (
    AllKeysExhaustedError,
    KeyPoolConfigError,
    SentinelKeyUnavailableError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider-level config: failure threshold + cooldown seconds + RR weight
# ---------------------------------------------------------------------------

PROVIDER_FAILURE_THRESHOLD: dict[str, int] = {
    "groq":       3,
    "cerebras":   3,
    "together":   3,
    "openrouter": 3,
    "gemini":     3,
}

PROVIDER_COOLDOWN_SECONDS: dict[str, int] = {
    "groq":       3600,       # 1 hour
    "cerebras":   1800,       # 30 min
    "together":   1800,       # 30 min
    "openrouter": 1800,       # 30 min
    "gemini":     86400,      # 24 hours
}

PROVIDER_WEIGHTS: dict[str, int] = {
    "groq":       3,
    "cerebras":   3,
    "together":   2,
    "openrouter": 2,
    "gemini":     2,
}

REQUIRED_KEY_FIELDS = {"key_id", "key", "provider", "model", "pool_type"}


# ---------------------------------------------------------------------------
# KeyPool
# ---------------------------------------------------------------------------

class KeyPool:
    """Dual sub-pool key rotation with per-provider circuit breakers.

    Instantiate once at startup and pass to TaskDispatcher + Sentinel:

        pool = KeyPool(config)

        # General usage (task_dispatcher, llm_router):
        key_obj = await pool.get_key()            # any provider, weighted RR
        key_obj = await pool.get_key("general")   # explicit

        # Sentinel usage (sentinel.py only):
        key_obj = await pool.get_sentinel_key()

        # After call completes:
        await pool.report_success(key_obj["key_id"])
        # On 429 / 5xx:
        await pool.report_failure(key_obj["key_id"])
    """

    def __init__(self, config: dict) -> None:
        """Load and validate all keys from config. Split into general + sentinel sub-pools.

        config format:
            {
              "keys": [
                {"key_id": "groq_0", "key": "gsk_...", "provider": "groq",
                 "model": "llama-3.3-70b-versatile", "pool_type": "general", "weight": 3},
                {"key_id": "gemini_sentinel", "key": "AIza...", "provider": "gemini",
                 "model": "gemini-2.5-pro-preview-03-25", "pool_type": "sentinel", "weight": 1},
                ...
              ]
            }
        """
        raw_keys: list[dict] = config.get("keys", [])
        if not raw_keys:
            raise KeyPoolConfigError("KeyPool config has no keys defined.")

        self.general: list[dict] = []
        self.sentinel: list[dict] = []
        self._key_index: dict[str, dict] = {}  # key_id → key_obj for O(1) lookup

        for raw in raw_keys:
            # Validate required fields
            missing = REQUIRED_KEY_FIELDS - set(raw.keys())
            if missing:
                raise KeyPoolConfigError(
                    f"Key '{raw.get('key_id', '?')}' missing fields: {missing}"
                )

            key_obj = {
                "key_id":   str(raw["key_id"]),
                "key":      str(raw["key"]),
                "provider": str(raw["provider"]).lower(),
                "model":    str(raw["model"]),
                "pool_type": str(raw["pool_type"]).lower(),
                "failures": 0,
                "reset_at": 0.0,
                "weight":   int(
                    raw.get("weight")
                    or PROVIDER_WEIGHTS.get(str(raw["provider"]).lower(), 2)
                ),
            }

            if key_obj["pool_type"] == "sentinel":
                self.sentinel.append(key_obj)
            elif key_obj["pool_type"] == "general":
                self.general.append(key_obj)
            else:
                raise KeyPoolConfigError(
                    f"Key '{key_obj['key_id']}' has unknown pool_type '{key_obj['pool_type']}'. "
                    f"Must be 'general' or 'sentinel'."
                )

            self._key_index[key_obj["key_id"]] = key_obj

        # Weighted RR state (general pool only; sentinel pool has no RR)
        self._rr_index: int = 0

        # Single lock for all mutation (asyncio-safe)
        self._lock = asyncio.Lock()

        logger.info(
            f"[KeyPool] Initialized: general={len(self.general)} keys, "
            f"sentinel={len(self.sentinel)} keys."
        )
        # Log provider distribution (no key values)
        provider_counts: dict[str, int] = {}
        for k in self.general:
            provider_counts[k["provider"]] = provider_counts.get(k["provider"], 0) + 1
        logger.info(f"[KeyPool] General pool provider distribution: {provider_counts}")
        if self.sentinel:
            logger.info(
                f"[KeyPool] Sentinel key: provider={self.sentinel[0]['provider']} "
                f"model={self.sentinel[0]['model']}"
            )
        else:
            logger.warning("[KeyPool] No sentinel key configured — Sentinel layer will be inactive.")

    # -----------------------------------------------------------------------
    # Availability check (called inside lock only — see P6)
    # -----------------------------------------------------------------------

    def _is_available(self, key_obj: dict) -> bool:
        """Return True if key is not tripped, or if its cooldown has expired (auto-reset).

        Auto-reset: when cooldown expires, clear failures + reset_at so the key
        returns to full quota immediately (not gradual warm-up needed at free tier).

        Bug P6: only called inside asyncio.Lock block — no race condition.
        """
        if key_obj["reset_at"] == 0.0:
            return True  # never tripped or already reset

        if time.monotonic() >= key_obj["reset_at"]:
            # Cooldown expired — auto-reset
            key_obj["failures"] = 0
            key_obj["reset_at"] = 0.0
            logger.info(
                f"[KeyPool] Key '{key_obj['key_id']}' cooldown expired — auto-reset."
            )
            return True

        return False  # still in cooldown

    # -----------------------------------------------------------------------
    # Weighted round-robin selection (general pool)
    # -----------------------------------------------------------------------

    def _select_weighted(self) -> Optional[dict]:
        """Select next available key from general pool using weighted RR.

        Algorithm:
          1. Build expanded list: each key appears weight times.
          2. Starting from rr_index, scan forward (wrapping) for first available key.
          3. Advance rr_index past selected key's weight slot.

        Bug P2: if all high-weight keys trip, scan must still reach low-weight keys.
        Scan is linear up to total_weight slots — guaranteed to find any available key.

        Returns None if all keys unavailable (caller raises AllKeysExhaustedError).
        """
        if not self.general:
            return None

        # Build expanded weighted list (indices into self.general)
        weighted: list[int] = []
        for i, k in enumerate(self.general):
            weighted.extend([i] * k["weight"])

        total = len(weighted)
        if total == 0:
            return None

        # Scan from rr_index forward, wrapping
        for offset in range(total):
            slot = (self._rr_index + offset) % total
            key_idx = weighted[slot]
            candidate = self.general[key_idx]
            if self._is_available(candidate):
                # Advance rr_index to next slot after this one
                self._rr_index = (slot + 1) % total
                return candidate

        # All slots exhausted
        return None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def get_key(self, pool_type: str = "general") -> dict:
        """Return best available key from the specified sub-pool.

        Args:
            pool_type: "general" (default) or "sentinel".
                       For sentinel calls, prefer get_sentinel_key() directly.

        Returns:
            key_obj dict. Caller MUST call report_success or report_failure after use.

        Raises:
            AllKeysExhaustedError: all general keys are in cooldown.
            SentinelKeyUnavailableError: sentinel key is tripped (if pool_type="sentinel").
        """
        async with self._lock:
            if pool_type == "sentinel":
                return await self._get_sentinel_locked()

            key_obj = self._select_weighted()
            if key_obj is None:
                states = {
                    k["key_id"]: {"failures": k["failures"], "reset_at": k["reset_at"]}
                    for k in self.general
                }
                raise AllKeysExhaustedError(provider_states=states)

            logger.debug(
                f"[KeyPool] get_key → key_id={key_obj['key_id']} "
                f"provider={key_obj['provider']}"
            )
            return key_obj

    async def get_sentinel_key(self) -> dict:
        """Return the dedicated Sentinel Gemini key.

        Raises:
            SentinelKeyUnavailableError: if the sentinel key is in cooldown.
        """
        async with self._lock:
            return await self._get_sentinel_locked()

    async def _get_sentinel_locked(self) -> dict:
        """Internal sentinel key retrieval (must be called inside self._lock)."""
        if not self.sentinel:
            raise SentinelKeyUnavailableError(reset_at=None)

        # Current design: use first available sentinel key
        # Bug P3: when multiple sentinel keys added, add priority ordering here.
        for key_obj in self.sentinel:
            if self._is_available(key_obj):
                logger.debug(
                    f"[KeyPool] get_sentinel_key → key_id={key_obj['key_id']}"
                )
                return key_obj

        # All sentinel keys tripped
        earliest_reset = min(k["reset_at"] for k in self.sentinel)
        raise SentinelKeyUnavailableError(reset_at=earliest_reset)

    async def report_success(self, key_id: str) -> None:
        """Reset failure count and reset_at for the given key after a successful call.

        Idempotent — safe to call even if key was never tripped.
        """
        async with self._lock:
            key_obj = self._key_index.get(key_id)
            if key_obj is None:
                logger.warning(f"[KeyPool] report_success: unknown key_id '{key_id}'")
                return
            if key_obj["failures"] > 0 or key_obj["reset_at"] > 0:
                logger.debug(
                    f"[KeyPool] report_success: resetting key '{key_id}' "
                    f"(was failures={key_obj['failures']})"
                )
            key_obj["failures"] = 0
            key_obj["reset_at"] = 0.0

    async def report_failure(self, key_id: str) -> None:
        """Increment failure count. Trip circuit breaker if threshold reached.

        Threshold (per provider): 3 failures → cooldown.
        Groq: 1hr. Gemini: 24hr. All others: 30min.

        Bug P1: no Redis write here yet. Add Redis persistence in future phase.
        """
        async with self._lock:
            key_obj = self._key_index.get(key_id)
            if key_obj is None:
                logger.warning(f"[KeyPool] report_failure: unknown key_id '{key_id}'")
                return

            key_obj["failures"] += 1
            provider = key_obj["provider"]
            threshold = PROVIDER_FAILURE_THRESHOLD.get(provider, 3)
            cooldown = PROVIDER_COOLDOWN_SECONDS.get(provider, 1800)

            if key_obj["failures"] >= threshold and key_obj["reset_at"] == 0.0:
                # Trip the breaker
                key_obj["reset_at"] = time.monotonic() + cooldown
                logger.warning(
                    f"[KeyPool] Circuit breaker TRIPPED: key_id='{key_id}' "
                    f"provider={provider} failures={key_obj['failures']} "
                    f"cooldown={cooldown}s reset_at={key_obj['reset_at']:.0f}"
                )
            else:
                logger.debug(
                    f"[KeyPool] report_failure: key_id='{key_id}' "
                    f"failures={key_obj['failures']}/{threshold}"
                )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    async def status(self) -> dict:
        """Return current pool status dict. Useful for /health endpoint + Sentinel audit.

        Returns:
            {
              "general": [
                {"key_id": str, "provider": str, "failures": int,
                 "tripped": bool, "reset_in_seconds": float | None},
                ...
              ],
              "sentinel": [...same shape...],
              "general_available": int,   # count of non-tripped general keys
              "sentinel_available": int,  # count of non-tripped sentinel keys
            }
        """
        now = time.monotonic()
        async with self._lock:
            def key_status(k: dict) -> dict:
                tripped = k["reset_at"] > 0 and now < k["reset_at"]
                return {
                    "key_id":   k["key_id"],
                    "provider": k["provider"],
                    "model":    k["model"],
                    "failures": k["failures"],
                    "tripped":  tripped,
                    "reset_in_seconds": max(0.0, k["reset_at"] - now) if tripped else None,
                }

            general_statuses = [key_status(k) for k in self.general]
            sentinel_statuses = [key_status(k) for k in self.sentinel]

            return {
                "general":           general_statuses,
                "sentinel":          sentinel_statuses,
                "general_available": sum(1 for s in general_statuses if not s["tripped"]),
                "sentinel_available": sum(1 for s in sentinel_statuses if not s["tripped"]),
            }
