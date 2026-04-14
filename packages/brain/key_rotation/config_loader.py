"""
packages/brain/key_rotation/config_loader.py

Ultron V4 — KeyPool Config Loader
===================================
Single function: build_pool_config(settings) → dict

Translates the flat Settings object into the nested dict that KeyPool.__init__
expects. This is the ONLY place where key_id strings are generated and provider
routings are decided.

Output format (matches KeyPool.__init__ exactly):
    {
      "keys": [
        {
          "key_id":    "groq_0",
          "key":       "gsk_...",
          "provider":  "groq",
          "model":     "llama-3.3-70b-versatile",
          "pool_type": "general",
          "weight":    3
        },
        ...
        {
          "key_id":    "gemini_sentinel",
          "key":       "AIza...",
          "provider":  "gemini",
          "model":     "gemini-2.5-pro-preview-03-25",
          "pool_type": "sentinel",
          "weight":    1
        }
      ]
    }

Provider → weight (from LOCKED DECISIONS v17):
    groq=3, cerebras=3, together=2, openrouter=2, gemini=2

Provider → model:
    Pulled from PROVIDER_DEFAULT_MODELS in config.py.
    Can be overridden by env var GROQ_MODEL / CEREBRAS_MODEL / etc. (optional).

Future bug risks (pre-registered):
  CL1 [HIGH]  If Settings.groq_keys is empty BUT gemini_keys has entries, general
              pool still works but Groq weight=0 keys appear nowhere. Silent — correct
              behavior. BUT if someone checks `pool.general[0]['provider'] == 'groq'`
              they'll crash on empty list. Never assume provider ordering in pool.

  CL2 [MED]   GROQ_MODEL override via env var is optional. If Ghost sets GROQ_MODEL=
              'some-retired-model', all groq keys will 401. No validation here.
              Fix: add model validity check in future (litellm.get_model_info() call).

  CL3 [MED]   key_id = f"{provider}_{index}" — if Ghost passes same key string
              under different providers (copy-paste), both will be in pool and
              both will trip. No dedup guard (same as pool P5). Low priority.

  CL4 [LOW]   If gemini_sentinel_key is None (not set in HF Space), no sentinel
              entry is added to config. KeyPool logs a WARNING. Downstream:
              get_sentinel_key() raises SentinelKeyUnavailableError — EXPECTED.
              Every sentinel caller already handles this. Not a bug, document it.

Tool calls used writing this file:
    Github:get_file_contents x1 (pool.py — confirmed KeyPool constructor format)
    Github:get_file_contents x1 (litellm proxy_server — ProxyConfig.load_config pattern)
"""

from __future__ import annotations

import logging
import os

from packages.shared.config import PROVIDER_DEFAULT_MODELS, Settings

logger = logging.getLogger(__name__)

# Per-provider weights — mirrors PROVIDER_WEIGHTS in pool.py (locked v17)
_WEIGHTS: dict[str, int] = {
    "groq":       3,
    "cerebras":   3,
    "together":   2,
    "openrouter": 2,
    "gemini":     2,
}


def _resolve_model(provider: str) -> str:
    """Return model string for provider. Env override takes priority.

    Env var: GROQ_MODEL, CEREBRAS_MODEL, TOGETHER_MODEL, OPENROUTER_MODEL, GEMINI_MODEL.
    Falls back to PROVIDER_DEFAULT_MODELS if override not set.
    Bug CL2: no validity check on override value.
    """
    env_key = f"{provider.upper()}_MODEL"
    override = os.environ.get(env_key, "").strip()
    if override:
        logger.info(f"[ConfigLoader] Model override: {provider} → {override}")
        return override
    return PROVIDER_DEFAULT_MODELS.get(provider, "unknown-model")


def build_pool_config(settings: Settings) -> dict:
    """Build the KeyPool config dict from parsed Settings.

    Called once during FastAPI lifespan startup. Result passed directly to
    KeyPool.__init__. No external calls — pure transformation.

    Returns:
        {"keys": [key_obj, ...]}
    """
    keys: list[dict] = []

    # ── General pool ──────────────────────────────────────────────────────
    # Provider order here is cosmetic — KeyPool uses weighted RR anyway.
    provider_key_map: list[tuple[str, list[str]]] = [
        ("groq",       settings.groq_keys),
        ("cerebras",   settings.cerebras_keys),
        ("together",   settings.together_keys),
        ("openrouter", settings.openrouter_keys),
        ("gemini",     settings.gemini_keys),
    ]

    for provider, raw_keys in provider_key_map:
        if not raw_keys:
            logger.debug(f"[ConfigLoader] No keys for provider '{provider}' — skipping.")
            continue

        model  = _resolve_model(provider)
        weight = _WEIGHTS.get(provider, 2)

        for i, raw_key in enumerate(raw_keys):
            key_obj = {
                "key_id":    f"{provider}_{i}",
                "key":       raw_key,
                "provider":  provider,
                "model":     model,
                "pool_type": "general",
                "weight":    weight,
            }
            keys.append(key_obj)
            logger.debug(f"[ConfigLoader] Registered general key: {provider}_{i} model={model}")

    # ── Sentinel pool ─────────────────────────────────────────────────────
    # Bug CL4: if sentinel key absent, no entry added. KeyPool handles gracefully.
    if settings.gemini_sentinel_key:
        sentinel_model = os.environ.get("GEMINI_SENTINEL_MODEL", "").strip() \
                         or PROVIDER_DEFAULT_MODELS["gemini_sentinel"]
        keys.append({
            "key_id":    "gemini_sentinel",
            "key":       settings.gemini_sentinel_key,
            "provider":  "gemini",
            "model":     sentinel_model,
            "pool_type": "sentinel",
            "weight":    1,
        })
        logger.info(
            f"[ConfigLoader] Sentinel key registered: model={sentinel_model}"
        )
    else:
        logger.warning(
            "[ConfigLoader] No sentinel key — Sentinel layer inactive. "
            "Set GEMINI_SENTINEL_KEY in HF Space secrets."
        )

    total = len(keys)
    general_count = sum(1 for k in keys if k["pool_type"] == "general")
    sentinel_count = total - general_count

    logger.info(
        f"[ConfigLoader] Pool config built: "
        f"general={general_count} sentinel={sentinel_count} total={total}"
    )

    if general_count == 0:
        # Settings.__init__ already raised — this is a double-guard
        raise RuntimeError(
            "[ConfigLoader] FATAL: general pool is empty after config build. "
            "This should have been caught by Settings validation."
        )

    return {"keys": keys}
