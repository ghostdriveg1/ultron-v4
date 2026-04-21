"""
packages/shared/config.py

Ultron V4 — Centralised Settings (Pydantic v2 BaseSettings)
=============================================================
Single source of truth for ALL environment variables across every V4 module.

Usage (anywhere in codebase):
    from packages.shared.config import get_settings
    s = get_settings()          # cached singleton, zero re-parse cost
    s.groq_keys                 # list[str]
    s.redis_url                 # str

Environment variable naming convention:
    GROQ_KEY_0, GROQ_KEY_1, ...   (indexed, parsed into list)
    CEREBRAS_KEY_0, ...
    TOGETHER_KEY_0, ...
    OPENROUTER_KEY_0, ...
    GEMINI_KEY_0, ...             (general pool)
    GEMINI_SENTINEL_KEY            (sentinel pool — dedicated, never shared)
    REDIS_URL
    ZILLIZ_URI, ZILLIZ_TOKEN
    SUPABASE_URL, SUPABASE_KEY
    DISCORD_TOKEN                  (optional — bot inactive if unset)
    CF_KV_API_TOKEN, CF_ACCOUNT_ID, CF_KV_NAMESPACE_ID
    ULTRON_AUTH_TOKEN              (brain ↔ CF Worker shared secret)
    TAVILY_API_KEY                 (search tool — free tier)
    BRAIN_PORT                     (default 7860 for HF Spaces)

Startup validation:
    get_settings() raises on first call if any REQUIRED var is missing.
    Required = REDIS_URL + at least one LLM key in general pool.
    DISCORD_TOKEN is optional — bot skipped if unset.
    All others are optional-with-warnings.

Future bug risks (pre-registered):
  S1 [HIGH]   HF Space secrets sometimes inject vars with trailing whitespace
              → key.strip() called on all key strings at parse time.
              If strip() removed, Groq/Gemini API will 401 silently.

  S2 [HIGH]   If GEMINI_SENTINEL_KEY is absent, Sentinel layer fails at import.
              Config logs a WARNING but does NOT raise — Sentinel is optional.
              Any file that calls get_sentinel_key() MUST handle SentinelKeyUnavailableError.

  S3 [MED]    Pydantic v1 vs v2: model_config vs class Config. This file uses v2.
              If HF Space has pydantic<2 pinned, all BaseSettings calls break silently.
              Fix: pin pydantic>=2.0 in requirements.txt.

  S4 [MED]    _parse_indexed_keys reads MAX_KEYS_PER_PROVIDER=20 slots.
              If Ghost adds key_21, it will be silently ignored.
              Fix: increase MAX or use dynamic discovery.

  S5 [LOW]    lru_cache on get_settings() means settings are frozen after first call.
              If Ghost rotates a key in HF Space secrets mid-run, change won't be seen
              until process restart. This is intentional — document it clearly.

Tool calls used writing this file:
    Github:get_file_contents x1 (pool.py — to confirm KeyPool config format)
    Github:get_file_contents (litellm proxy_server.py — startup/config patterns)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Max indexed keys to scan per provider (GROQ_KEY_0 … GROQ_KEY_19)
MAX_KEYS_PER_PROVIDER = 20

# ---------------------------------------------------------------------------
# Provider → default model mapping
# ---------------------------------------------------------------------------

PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "groq":       "llama-3.3-70b-versatile",
    "cerebras":   "llama3.1-70b",
    "together":   "meta-llama/Llama-3-70b-chat-hf",
    "openrouter": "mistralai/mistral-7b-instruct",
    "gemini":     "gemini-2.0-flash",
    "gemini_sentinel": "gemini-2.5-pro-preview-03-25",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_indexed_keys(prefix: str) -> list[str]:
    """Scan env vars GROQ_KEY_0 … GROQ_KEY_N. Return stripped, non-empty values.

    Bug S1: strip() on every key — HF Space sometimes injects trailing whitespace.
    Bug S4: stops at MAX_KEYS_PER_PROVIDER — raise if you need more.
    """
    keys: list[str] = []
    for i in range(MAX_KEYS_PER_PROVIDER):
        val = os.environ.get(f"{prefix}_{i}", "").strip()
        if val:
            keys.append(val)
    return keys


def _require(name: str, context: str = "") -> str:
    """Return env var value or raise RuntimeError with actionable message."""
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(
            f"[Config] REQUIRED env var '{name}' is missing or empty. "
            f"{context}Set it in HF Space → Settings → Repository Secrets."
        )
    return val


def _optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# ---------------------------------------------------------------------------
# Settings dataclass (no Pydantic dependency — plain Python for portability)
# ---------------------------------------------------------------------------

class Settings:
    """
    Parsed + validated settings. Instantiated once via get_settings().
    All fields are read-only after construction (no setattr enforcement,
    but treat them as immutable — bug S5).
    """

    # LLM keys — general pool
    groq_keys:       list[str]
    cerebras_keys:   list[str]
    together_keys:   list[str]
    openrouter_keys: list[str]
    gemini_keys:     list[str]   # general-pool Gemini keys (NOT sentinel)

    # LLM keys — sentinel pool
    gemini_sentinel_key: Optional[str]   # None if not configured (S2)

    # Infrastructure
    redis_url:          str
    zilliz_uri:         str
    zilliz_token:       str
    supabase_url:       str
    supabase_key:       str

    # Discord — optional, bot skipped if unset
    discord_token: Optional[str]

    # Cloudflare
    cf_kv_api_token:    str
    cf_account_id:      str
    cf_kv_namespace_id: str

    # Auth
    ultron_auth_token: str

    # Search
    tavily_api_key: str

    # Server
    brain_port: int

    def __init__(self) -> None:
        # ── LLM keys ──────────────────────────────────────────────────────
        self.groq_keys       = _parse_indexed_keys("GROQ_KEY")
        self.cerebras_keys   = _parse_indexed_keys("CEREBRAS_KEY")
        self.together_keys   = _parse_indexed_keys("TOGETHER_KEY")
        self.openrouter_keys = _parse_indexed_keys("OPENROUTER_KEY")
        self.gemini_keys     = _parse_indexed_keys("GEMINI_KEY")

        # Sentinel key — optional, warn if missing
        _sentinel_raw = _optional("GEMINI_SENTINEL_KEY")
        self.gemini_sentinel_key = _sentinel_raw if _sentinel_raw else None
        if not self.gemini_sentinel_key:
            logger.warning(
                "[Config] GEMINI_SENTINEL_KEY not set. "
                "Sentinel layer will be INACTIVE. "
                "Set it in HF Space secrets when ready."
            )

        # ── Validate: at least one general pool key must exist ─────────────
        total_general = (
            len(self.groq_keys) + len(self.cerebras_keys)
            + len(self.together_keys) + len(self.openrouter_keys)
            + len(self.gemini_keys)
        )
        if total_general == 0:
            raise RuntimeError(
                "[Config] FATAL: No LLM keys found in any general pool provider. "
                "Set at least GROQ_KEY_0 in HF Space secrets."
            )

        # ── Infrastructure ─────────────────────────────────────────────────
        self.redis_url = _require(
            "REDIS_URL",
            "Used for per-channel context windows and AgentState persistence. "
        )
        self.zilliz_uri   = _optional("ZILLIZ_URI")
        self.zilliz_token = _optional("ZILLIZ_TOKEN")
        if not self.zilliz_uri:
            logger.warning("[Config] ZILLIZ_URI not set — Tier 2 vector memory inactive.")

        self.supabase_url = _optional("SUPABASE_URL")
        self.supabase_key = _optional("SUPABASE_KEY")
        if not self.supabase_url:
            logger.warning("[Config] SUPABASE_URL not set — Supabase structured memory inactive.")

        # ── Discord — OPTIONAL, bot skipped if unset ───────────────────────
        _discord_raw = _optional("DISCORD_TOKEN")
        self.discord_token = _discord_raw if _discord_raw else None
        if not self.discord_token:
            logger.warning(
                "[Config] DISCORD_TOKEN not set — Discord bot will be INACTIVE. "
                "Website-only mode active."
            )

        # ── Cloudflare ─────────────────────────────────────────────────────
        self.cf_kv_api_token    = _optional("CF_KV_API_TOKEN")
        self.cf_account_id      = _optional("CF_ACCOUNT_ID")
        self.cf_kv_namespace_id = _optional("CF_KV_NAMESPACE_ID", "77184c17886d47f2be73b6d441ada952")
        if not self.cf_kv_api_token:
            logger.warning("[Config] CF_KV_API_TOKEN not set — Sentinel KV routing writes disabled.")

        # ── Auth ───────────────────────────────────────────────────────────
        self.ultron_auth_token = _optional("ULTRON_AUTH_TOKEN")
        if not self.ultron_auth_token:
            logger.warning(
                "[Config] ULTRON_AUTH_TOKEN not set — /infer endpoint has no auth. "
                "OK for dev, NOT for prod."
            )

        # ── Search ─────────────────────────────────────────────────────────
        self.tavily_api_key = _optional("TAVILY_API_KEY")
        if not self.tavily_api_key:
            logger.warning("[Config] TAVILY_API_KEY not set — search tool will be inactive.")

        # ── Server ─────────────────────────────────────────────────────────
        port_str = _optional("BRAIN_PORT", "7860")
        try:
            self.brain_port = int(port_str)
        except ValueError:
            logger.warning(f"[Config] BRAIN_PORT='{port_str}' is not an integer. Using 7860.")
            self.brain_port = 7860

        # ── Summary log (no key values ever printed) ───────────────────────
        logger.info(
            f"[Config] Loaded. General pool: groq={len(self.groq_keys)} "
            f"cerebras={len(self.cerebras_keys)} together={len(self.together_keys)} "
            f"openrouter={len(self.openrouter_keys)} gemini={len(self.gemini_keys)} "
            f"total={total_general}. Sentinel={'YES' if self.gemini_sentinel_key else 'NO'}. "
            f"Redis={'SET' if self.redis_url else 'MISSING'}. "
            f"Discord={'SET' if self.discord_token else 'INACTIVE'}."
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the global Settings singleton. Parsed + validated on first call.

    Bug S5: lru_cache means settings are frozen after first import.
    Env var changes mid-run are NOT picked up — intentional.
    To force reload (testing only): get_settings.cache_clear() then call again.

    Raises:
        RuntimeError: if required env vars are missing.
    """
    return Settings()
