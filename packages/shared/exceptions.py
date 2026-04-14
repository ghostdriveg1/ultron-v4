"""
packages/shared/exceptions.py

Ultron V4 — Shared Custom Exceptions
======================================
All custom exception classes used across V4 modules live here.
Import pattern:
  from packages.shared.exceptions import AllKeysExhaustedError, SentinelKeyUnavailableError

Future bug risks (pre-registered):
  EX1 [MED] If pool.py is imported before shared/ is on PYTHONPATH, ImportError.
             Fix: ensure packages/ is in sys.path at startup (main.py responsibility).
  EX2 [LOW] New modules may define local exceptions with same name → shadowing.
             Rule: always import from shared.exceptions, never redefine locally.

Tool calls used writing this file:
  Github:push_files x1 (batch with pool.py)
"""


class UltronBaseError(Exception):
    """Base class for all Ultron V4 exceptions."""


class AllKeysExhaustedError(UltronBaseError):
    """Raised by KeyPool.get_key() when all keys in the general pool are tripped.

    Callers (task_dispatcher, llm_router) must catch this and return a 503-style
    response to Discord. Never swallow silently.

    Attributes:
        provider_states: dict of {provider: failure_count} for diagnostics.
    """

    def __init__(self, provider_states: dict | None = None):
        self.provider_states = provider_states or {}
        super().__init__(
            f"All keys in general pool exhausted. States: {self.provider_states}"
        )


class SentinelKeyUnavailableError(UltronBaseError):
    """Raised by KeyPool.get_sentinel_key() when the Sentinel Gemini key is tripped.

    Sentinel unavailability is a degraded-mode event. Ultron should:
      1. Log as CRITICAL
      2. Write incident to Notion (when Notion MCP available in Sentinel code)
      3. Continue serving user requests from general pool (Sentinel is non-user-facing)
      4. Retry Sentinel key after reset_at cooldown (24hr for Gemini).

    Attributes:
        reset_at: float UNIX timestamp when the key comes out of cooldown.
    """

    def __init__(self, reset_at: float | None = None):
        self.reset_at = reset_at
        super().__init__(
            f"Sentinel key unavailable. Cooldown until: {reset_at}"
        )


class KeyPoolConfigError(UltronBaseError):
    """Raised during KeyPool.__init__ if config is malformed.

    Fail loud at startup. Never silently skip bad config.
    """


class ProviderNotSupportedError(UltronBaseError):
    """Raised when an unrecognized provider string is encountered in llm_router."""
