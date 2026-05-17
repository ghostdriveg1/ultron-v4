"""
packages/brain/config_loader.py

Ultron V4 — Config → KeyPool Bridge
=====================================
Single glue layer between packages.shared.config.Settings and
packages.brain.key_rotation.pool.KeyPool.

Responsibilities:
  1. Read Settings singleton (get_settings())
  2. Build KeyPool-compatible config dict from all provider key lists
  3. Instantiate KeyPool singleton (get_pool())
  4. Provide get_sentinel_llm_fn() for Sentinel-only direct calls

This file is the ONLY place that maps Settings.{provider}_keys → KeyPool entries.
No other file should construct KeyPool directly.

Usage:
    from packages.brain.config_loader import get_pool, get_sentinel_llm_fn

    pool = get_pool()                         # KeyPool singleton
    sentinel_fn = get_sentinel_llm_fn()       # Sentinel-only LLM fn, or None

Pool build rules (locked):
  General pool: groq(w3) cerebras(w3) together(w2) openrouter(w2) gemini(w2)
                sambanova(w2) fireworks(w2) hf(w1)
  Sentinel pool: gemini_sentinel_key only. Absent → sentinel inactive.
  Key ID format: "{provider}_{index}" e.g. "groq_0", "sambanova_1"
  Model: from PROVIDER_DEFAULT_MODELS mapping.

Future bug risks (pre-registered):
  CL7 [HIGH]   get_pool() uses lru_cache — frozen after first call.
               If Settings changes (cache_clear + reinit), pool is NOT updated.
               Pattern: always call get_pool() fresh in tests after cache clear.

  CL8 [MED]    KeyPool.__init__ raises KeyPoolConfigError if zero keys.
               get_pool() will propagate this — main.py lifespan must catch it
               and log FATAL before raising (already done in main.py lifespan).

  CL9 [MED]    HF keys (hf_keys) give lower weight=1. If HF is only provider
               set, pool works but may be slow. Warn in logs if hf is sole provider.

  CL10 [LOW]   get_sentinel_llm_fn returns None if no sentinel key. Callers
               (sentinel.py, eternal_loop_router.py) must guard: if fn is None → skip.

Tool calls used writing this file (v33):
    Github:get_file_contents x1 (packages/shared/config.py)
    Github:get_file_contents x1 (packages/brain/key_rotation/pool.py)
    Github:get_file_contents x1 (packages/brain/llm_router.py)
    Github:push_files x1 (batch commit)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable, Optional

from packages.shared.config import PROVIDER_DEFAULT_MODELS, get_settings

logger = logging.getLogger(__name__)

# Provider weight table — mirrors pool.py PROVIDER_WEIGHTS (single source = pool.py)
_PROVIDER_WEIGHTS: dict[str, int] = {
    "groq":       3,
    "cerebras":   3,
    "together":   2,
    "openrouter": 2,
    "gemini":     2,
    "sambanova":  2,
    "fireworks":  2,
    "hf":         1,
}


# ---------------------------------------------------------------------------
# Internal: build KeyPool config dict from Settings
# ---------------------------------------------------------------------------

def _build_pool_config() -> dict:
    """
    Reads Settings, constructs the list[dict] that KeyPool.__init__ expects.

    General pool entries: one entry per key per provider.
    Sentinel pool entry: one entry (or zero if key absent — CL10).

    Returns:
        {"keys": list[dict]}  — KeyPool-compatible config.
    """
    s = get_settings()
    keys: list[dict] = []

    # ── General pool ────────────────────────────────────────────────────────
    _general_providers: list[tuple[str, list[str]]] = [
        ("groq",       s.groq_keys),
        ("cerebras",   s.cerebras_keys),
        ("together",   s.together_keys),
        ("openrouter", s.openrouter_keys),
        ("gemini",     s.gemini_keys),
        ("sambanova",  s.sambanova_keys),
        ("fireworks",  s.fireworks_keys),
        ("hf",         s.hf_keys),
    ]

    total_general = 0
    hf_only = False

    for provider, key_list in _general_providers:
        model   = PROVIDER_DEFAULT_MODELS.get(provider, "")
        weight  = _PROVIDER_WEIGHTS.get(provider, 2)
        for idx, raw_key in enumerate(key_list):
            keys.append({
                "key_id":    f"{provider}_{idx}",
                "key":       raw_key,
                "provider":  provider,
                "model":     model,
                "pool_type": "general",
                "weight":    weight,
            })
            total_general += 1

    # CL9: warn if HF is the only provider (slow inference)
    non_hf = sum(
        len(kl) for p, kl in _general_providers if p != "hf"
    )
    if total_general > 0 and non_hf == 0:
        logger.warning(
            "[ConfigLoader] HF Inference API is the only general pool provider. "
            "Expect cold-start latency ~20s. Add at least GROQ_KEY_0 for better perf."
        )

    # ── Sentinel pool ────────────────────────────────────────────────────────
    if s.gemini_sentinel_key:
        keys.append({
            "key_id":    "gemini_sentinel",
            "key":       s.gemini_sentinel_key,
            "provider":  "gemini",
            "model":     PROVIDER_DEFAULT_MODELS["gemini_sentinel"],
            "pool_type": "sentinel",
            "weight":    1,
        })
        logger.info("[ConfigLoader] Sentinel key loaded.")
    else:
        logger.warning(
            "[ConfigLoader] No GEMINI_SENTINEL_KEY — sentinel pool empty. "
            "Sentinel layer inactive (CL10)."
        )

    logger.info(
        f"[ConfigLoader] Pool config built: {total_general} general keys, "
        f"{'1' if s.gemini_sentinel_key else '0'} sentinel keys."
    )
    return {"keys": keys}


# ---------------------------------------------------------------------------
# Public: KeyPool singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_pool():
    """
    Build and cache KeyPool singleton.

    CL7: frozen after first call — cache_clear() in tests only.
    CL8: KeyPoolConfigError propagates if zero keys — catch in main.py lifespan.

    Returns:
        KeyPool instance
    Raises:
        KeyPoolConfigError: if no keys at all (Settings validation passed but pool
                            rejects empty list — shouldn't happen in normal flow).
    """
    from packages.brain.key_rotation.pool import KeyPool

    config = _build_pool_config()
    pool   = KeyPool(config)
    return pool


# ---------------------------------------------------------------------------
# Public: Sentinel LLM fn
# ---------------------------------------------------------------------------

def get_sentinel_llm_fn() -> Optional[Callable]:
    """
    Returns an async fn(messages, tools) -> Optional[dict] bound to sentinel key.
    Returns None if sentinel key absent (CL10) — callers must guard.

    Usage in sentinel.py:
        from packages.brain.config_loader import get_sentinel_llm_fn
        _sentinel_fn = get_sentinel_llm_fn()
        if _sentinel_fn is None:
            logger.warning('Sentinel inactive — no key')
            return
        result = await _sentinel_fn(messages, tools)
    """
    s = get_settings()
    if not s.gemini_sentinel_key:
        return None

    from packages.brain.llm_router import call_provider

    sentinel_key   = s.gemini_sentinel_key
    sentinel_model = PROVIDER_DEFAULT_MODELS["gemini_sentinel"]

    async def _sentinel_call(
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Optional[dict]:
        return await call_provider(
            provider="gemini",
            api_key=sentinel_key,
            model=sentinel_model,
            messages=messages,
            tools=tools or [],
        )

    return _sentinel_call


# ---------------------------------------------------------------------------
# Convenience: parallel AI key round-robin (for search.py)
# ---------------------------------------------------------------------------

_parallel_ai_rr: int = 0


def get_parallel_ai_key() -> Optional[str]:
    """
    Returns next Parallel AI key via round-robin, or None if none configured.
    Thread-safe for single-worker uvicorn (no asyncio.Lock needed — GIL-protected int).

    S6: keys parsed from PARALLEL_AI_KEY_0..N in Settings.
    """
    global _parallel_ai_rr
    s = get_settings()
    if not s.parallel_ai_keys:
        return None
    key = s.parallel_ai_keys[_parallel_ai_rr % len(s.parallel_ai_keys)]
    _parallel_ai_rr += 1
    return key
