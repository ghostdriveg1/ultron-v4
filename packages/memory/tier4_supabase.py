"""
packages/memory/tier4_supabase.py

Ultron V4 — Tier 4: Structured Store (Supabase)
================================================
Cherry-picked from V5 memory/engine.py StructuredStore (session v30 verdict: real+thin).
Gracefully degrades if SUPABASE_URL / SUPABASE_KEY absent.

Tables assumed in Supabase:
  preferences     (user_id TEXT, key TEXT, value JSONB, updated TIMESTAMPTZ)
  projects        (id TEXT PK, name TEXT, description TEXT, metadata JSONB, updated TIMESTAMPTZ)
  rd_improvements (id BIGSERIAL PK, project_id TEXT, cycle INT, improvement TEXT,
                   verdict TEXT, metadata JSONB, created TIMESTAMPTZ)

Integration:
  main.py lifespan Step 5f: StructuredStore init, app.state.tier4
  rd_loop.py (future): call store_rd_improvement() on each accepted improvement

Future bug risks (pre-registered):
  SB1 [HIGH]  supabase-py is synchronous. All calls block the event loop.
              Fix: run in asyncio.get_event_loop().run_in_executor(None, fn) or
              switch to postgrest-py async client when mature.
              For now, all methods wrap in asyncio.to_thread() (Python 3.9+).

  SB2 [MED]   Supabase free tier has row-level security disabled by default.
              If Ghost enables RLS without configuring policies, all writes silently
              fail with 403. Symptom: store_preference() returns None without error.
              Fix: test with explicit anon key + check RLS policies on each table.

  SB3 [MED]   preferences table has no unique constraint on (user_id, key) in
              default Supabase schema. upsert() requires a conflict column declared
              or it inserts duplicates. Fix: add UNIQUE(user_id, key) constraint or
              use .on_conflict("user_id, key") in supabase-py v2.

  SB4 [LOW]   create_client() is called at initialize() time. If SUPABASE_URL is
              malformed (e.g., missing https://) supabase-py raises ValueError, not
              caught by the ImportError guard. Add explicit URL validation before init.

Tool calls used writing this file:
  Github:get_file_contents x1 (packages/memory/ listing)
  Github:push_files x1 (batch commit)
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("ultron.tier4_supabase")


class StructuredStore:
    """
    Tier 4: Relational structured memory via Supabase.
    User preferences, project metadata, R&D improvement records.

    All public methods are async; synchronous supabase-py calls wrapped in
    asyncio.to_thread() to avoid blocking the event loop (SB1).
    """

    def __init__(
        self,
        supabase_url: str = "",
        supabase_key: str = "",
    ) -> None:
        self._url    = supabase_url.strip()
        self._key    = supabase_key.strip()
        self._client = None  # set in initialize()

    async def initialize(self) -> None:
        """Connect to Supabase. Graceful degrade if env vars absent or import fails."""
        if not self._url or not self._key:
            logger.warning(
                "[Tier4] SUPABASE_URL or SUPABASE_KEY not set — StructuredStore disabled"
            )
            return
        try:
            # SB4: basic URL validation
            if not self._url.startswith("https://"):
                raise ValueError(f"SUPABASE_URL must start with https://, got: {self._url[:30]}")
            from supabase import create_client  # type: ignore
            self._client = await asyncio.to_thread(create_client, self._url, self._key)
            logger.info("[Tier4] Supabase StructuredStore connected")
        except ImportError:
            logger.warning(
                "[Tier4] supabase package not installed — pip install supabase. StructuredStore disabled."
            )
        except Exception as exc:
            logger.warning(f"[Tier4] Supabase init failed (non-fatal): {exc}")

    # -------------------------------------------------------------------------
    # Preferences
    # -------------------------------------------------------------------------

    async def store_preference(self, user_id: str, key: str, value: Any) -> None:
        """Upsert a user preference. SB3: requires UNIQUE(user_id, key) in Supabase."""
        if not self._client:
            return
        try:
            payload = {
                "user_id": user_id,
                "key":     key,
                "value":   json.dumps(value),
                "updated": datetime.now(timezone.utc).isoformat(),
            }
            await asyncio.to_thread(
                lambda: self._client.table("preferences").upsert(payload).execute()
            )
        except Exception as exc:
            logger.error(f"[Tier4] store_preference failed: {exc}")

    async def get_preferences(self, user_id: str) -> Dict[str, Any]:
        if not self._client:
            return {}
        try:
            result = await asyncio.to_thread(
                lambda: self._client.table("preferences")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )
            return {row["key"]: json.loads(row["value"]) for row in result.data}
        except Exception as exc:
            logger.error(f"[Tier4] get_preferences failed: {exc}")
            return {}

    # -------------------------------------------------------------------------
    # Projects
    # -------------------------------------------------------------------------

    async def store_project(
        self,
        project_id: str,
        name: str,
        description: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        if not self._client:
            return
        try:
            payload = {
                "id":          project_id,
                "name":        name,
                "description": description,
                "metadata":    json.dumps(metadata or {}),
                "updated":     datetime.now(timezone.utc).isoformat(),
            }
            await asyncio.to_thread(
                lambda: self._client.table("projects").upsert(payload).execute()
            )
        except Exception as exc:
            logger.error(f"[Tier4] store_project failed: {exc}")

    # -------------------------------------------------------------------------
    # R&D Improvements
    # -------------------------------------------------------------------------

    async def store_rd_improvement(
        self,
        project_id: str,
        cycle: int,
        improvement: str,
        verdict: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Persist accepted R&D loop improvements for long-term audit."""
        if not self._client:
            return
        try:
            payload = {
                "project_id":  project_id,
                "cycle":       cycle,
                "improvement": improvement,
                "verdict":     verdict,
                "metadata":    json.dumps(metadata or {}),
                "created":     datetime.now(timezone.utc).isoformat(),
            }
            await asyncio.to_thread(
                lambda: self._client.table("rd_improvements").insert(payload).execute()
            )
        except Exception as exc:
            logger.error(f"[Tier4] store_rd_improvement failed: {exc}")

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict:
        return {
            "connected": self._client is not None,
            "url":       self._url[:30] + "..." if self._url else "",
        }


# Singleton
_store: Optional[StructuredStore] = None


def get_structured_store(
    supabase_url: str = "",
    supabase_key: str = "",
) -> StructuredStore:
    global _store
    if _store is None:
        _store = StructuredStore(supabase_url=supabase_url, supabase_key=supabase_key)
    return _store
