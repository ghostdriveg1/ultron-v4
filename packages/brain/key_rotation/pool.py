"""
packages/brain/key_rotation/pool.py

Ultron V4 — KeyPool: Dual Sub-Pool Key Rotation with Circuit Breaker
======================================================================
Architecture (v17 LOCKED — never re-debate):

  GENERAL POOL  → self.general[]
    Providers: Groq, Cerebras, Together, OpenRouter, Gemini (5 regular keys each)
    + NEW (v31): SambaNova (weight 2), Fireworks (weight 2), HuggingFace (weight 1)
    Used by:   main task, MOA, Council, coding agent, browser agent, voice, embeddings
    Selection: weighted round-robin + circuit breaker
    Weights:   Groq=3, Cerebras=3, Together=2, OpenRouter=2, Gemini=2
               SambaNova=2, Fireworks=2, HuggingFace=1
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
                               # |"sambanova"|"fireworks"|"hf"
      "model":     str,        # model string for this key
      "pool_type": str,        # "general" or "sentinel"
      "failures":  int,        # current consecutive failure count
      "reset_at":  float,      # UNIX timestamp: key blocked until this time (0 = not tripped)
      "weight":    int,        # weighted RR weight (see PROVIDER_WEIGHTS)
    }

Circuit breaker thresholds (failures → cooldown duration):
  groq:        3 failures → 1 hour cooldown
  gemini:      3 failures → 24 hour cooldown
  cerebras:    3 failures → 30 min cooldown
  together:    3 failures → 30 min cooldown
  openrouter:  3 failures → 30 min cooldown
  sambanova:   3 failures → 30 min cooldown
  fireworks:   3 failures → 30 min cooldown
  hf:          3 failures → 30 min cooldown

Env vars for new providers (indexed key rotation pattern — matches existing):
  SAMBANOVA_KEY_0..N  — SambaNova Cloud keys
  FIREWORKS_KEY_0..N  — Fireworks AI keys
  HF_KEY_0..N         — HuggingFace Inference API tokens

Thread safety:
  asyncio.Lock on get_key(), get_sentinel_key(), report_success(), report_failure().
  No cross-coroutine race on failure counter.

State:
  Pure in-memory. Redis persistence is a future task (noted in gut-feel bugs).
  On process restart, all keys reset to 0 failures.

Future bug risks (pre-registered):
  P1 [HIGH]   No Redis persistence. If HF Space restarts mid-cooldown, tripped keys
              are instantly reset → burst retries → re-trip → exponential 429 storm.
  P2 [HIGH]   Weighted RR: if all high-weight keys trip, scan must reach low-weight keys.
              (Already handled by linear scan in _select_weighted — verified.)
  P3 [MED]    Sentinel key: single failure → 24hr locked. Add priority ordering when
              multiple sentinel keys added.
  P4 [MED]    get_key() called without await → TypeError. Annotated async def — watch.
  P5 [LOW]    Same key string under two key_ids shares quota but trips independently.
  P6 [LOW]    _is_available() auto-reset race — safe inside asyncio.Lock.

Tool calls used writing this file (v31):
  Github:get_file_contents x1 (pool.py)
  Github:push_files x1 (batch commit)
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
# Provider-level config
# ---------------------------------------------------------------------------

PROVIDER_FAILURE_THRESHOLD: dict[str, int] = {
    "groq":       3,
    "cerebras":   3,
    "together":   3,
    "openrouter": 3,
    "gemini":     3,
    "sambanova":  3,
    "fireworks":  3,
    "hf":         3,
}

PROVIDER_COOLDOWN_SECONDS: dict[str, int] = {
    "groq":       3600,   # 1 hour
    "cerebras":   1800,   # 30 min
    "together":   1800,
    "openrouter": 1800,
    "gemini":     86400,  # 24 hours
    "sambanova":  1800,
    "fireworks":  1800,
    "hf":         1800,
}

PROVIDER_WEIGHTS: dict[str, int] = {
    "groq":       3,
    "cerebras":   3,
    "together":   2,
    "openrouter": 2,
    "gemini":     2,
    "sambanova":  2,
    "fireworks":  2,
    "hf":         1,   # slower inference, lower weight
}

# Rate/token limits per provider (informational — used by future quota tracker)
PROVIDER_LIMITS: dict[str, dict] = {
    "groq":       {"rpm": 30,  "tpm": 14_400,    "tpd": 1_000_000,  "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]},
    "gemini":     {"rpm": 15,  "tpm": 1_000_000, "tpd": 1_000_000,  "models": ["gemini-2.0-flash", "gemini-1.5-pro"]},
    "openrouter": {"rpm": 20,  "tpm": 40_000,    "tpd": 500_000,    "models": ["meta-llama/llama-3.3-70b-instruct:free"]},
    "together":   {"rpm": 20,  "tpm": 40_000,    "tpd": 500_000,    "models": ["meta-llama/Llama-3-70b-chat-hf"]},
    "cerebras":   {"rpm": 60,  "tpm": 80_000,    "tpd": 1_000_000,  "models": ["llama-3.3-70b", "llama-3.1-8b"]},
    "sambanova":  {"rpm": 10,  "tpm": 20_000,    "tpd": 200_000,    "models": ["Meta-Llama-3.1-405B-Instruct", "Meta-Llama-3.1-70B-Instruct"]},
    "fireworks":  {"rpm": 30,  "tpm": 60_000,    "tpd": 500_000,    "models": ["accounts/fireworks/models/llama-v3p3-70b-instruct"]},
    "hf":         {"rpm": 10,  "tpm": 20_000,    "tpd": 100_000,    "models": ["meta-llama/Llama-3.3-70B-Instruct"]},
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
        raw_keys: list[dict] = config.get("keys", [])
        if not raw_keys:
            raise KeyPoolConfigError("KeyPool config has no keys defined.")

        self.general:  list[dict] = []
        self.sentinel: list[dict] = []
        self._key_index: dict[str, dict] = {}

        for raw in raw_keys:
            missing = REQUIRED_KEY_FIELDS - set(raw.keys())
            if missing:
                raise KeyPoolConfigError(
                    f"Key '{raw.get('key_id', '?')}' missing fields: {missing}"
                )

            key_obj = {
                "key_id":    str(raw["key_id"]),
                "key":       str(raw["key"]),
                "provider":  str(raw["provider"]).lower(),
                "model":     str(raw["model"]),
                "pool_type": str(raw["pool_type"]).lower(),
                "failures":  0,
                "reset_at":  0.0,
                "weight":    int(
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

        self._rr_index: int = 0
        self._lock = asyncio.Lock()

        logger.info(
            f"[KeyPool] Initialized: general={len(self.general)} keys, "
            f"sentinel={len(self.sentinel)} keys."
        )
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
    # Availability check
    # -----------------------------------------------------------------------

    def _is_available(self, key_obj: dict) -> bool:
        if key_obj["reset_at"] == 0.0:
            return True
        if time.monotonic() >= key_obj["reset_at"]:
            key_obj["failures"] = 0
            key_obj["reset_at"] = 0.0
            logger.info(f"[KeyPool] Key '{key_obj['key_id']}' cooldown expired — auto-reset.")
            return True
        return False

    # -----------------------------------------------------------------------
    # Weighted round-robin
    # -----------------------------------------------------------------------

    def _select_weighted(self) -> Optional[dict]:
        if not self.general:
            return None

        weighted: list[int] = []
        for i, k in enumerate(self.general):
            weighted.extend([i] * k["weight"])

        total = len(weighted)
        if total == 0:
            return None

        for offset in range(total):
            slot = (self._rr_index + offset) % total
            key_idx = weighted[slot]
            candidate = self.general[key_idx]
            if self._is_available(candidate):
                self._rr_index = (slot + 1) % total
                return candidate

        return None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def get_key(self, pool_type: str = "general") -> dict:
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
        async with self._lock:
            return await self._get_sentinel_locked()

    async def _get_sentinel_locked(self) -> dict:
        if not self.sentinel:
            raise SentinelKeyUnavailableError(reset_at=None)
        for key_obj in self.sentinel:
            if self._is_available(key_obj):
                return key_obj
        earliest_reset = min(k["reset_at"] for k in self.sentinel)
        raise SentinelKeyUnavailableError(reset_at=earliest_reset)

    async def report_success(self, key_id: str) -> None:
        async with self._lock:
            key_obj = self._key_index.get(key_id)
            if key_obj is None:
                logger.warning(f"[KeyPool] report_success: unknown key_id '{key_id}'")
                return
            key_obj["failures"] = 0
            key_obj["reset_at"] = 0.0

    async def report_failure(self, key_id: str) -> None:
        async with self._lock:
            key_obj = self._key_index.get(key_id)
            if key_obj is None:
                logger.warning(f"[KeyPool] report_failure: unknown key_id '{key_id}'")
                return

            key_obj["failures"] += 1
            provider = key_obj["provider"]
            threshold = PROVIDER_FAILURE_THRESHOLD.get(provider, 3)
            cooldown  = PROVIDER_COOLDOWN_SECONDS.get(provider, 1800)

            if key_obj["failures"] >= threshold and key_obj["reset_at"] == 0.0:
                key_obj["reset_at"] = time.monotonic() + cooldown
                logger.warning(
                    f"[KeyPool] Circuit breaker TRIPPED: key_id='{key_id}' "
                    f"provider={provider} failures={key_obj['failures']} "
                    f"cooldown={cooldown}s"
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
        now = time.monotonic()
        async with self._lock:
            def key_status(k: dict) -> dict:
                tripped = k["reset_at"] > 0 and now < k["reset_at"]
                return {
                    "key_id":           k["key_id"],
                    "provider":         k["provider"],
                    "model":            k["model"],
                    "failures":         k["failures"],
                    "tripped":          tripped,
                    "reset_in_seconds": max(0.0, k["reset_at"] - now) if tripped else None,
                }

            general_statuses  = [key_status(k) for k in self.general]
            sentinel_statuses = [key_status(k) for k in self.sentinel]

            return {
                "general":            general_statuses,
                "sentinel":           sentinel_statuses,
                "general_available":  sum(1 for s in general_statuses  if not s["tripped"]),
                "sentinel_available": sum(1 for s in sentinel_statuses if not s["tripped"]),
            }
