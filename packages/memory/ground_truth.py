"""
packages/memory/ground_truth.py

Raw episode store for Ultron V4 — MemMachine pattern (arXiv:2604.04853).

Principle:
  - WRITE raw, EXTRACT on read. Never LLM-paraphrase at write time.
  - Raw episodes are bit-perfect records. No drift across long projects.
  - Structured extraction (entities, actions, intents) happens on READ,
    keyed to the query — so the same episode yields different extractions
    depending on what's being asked.

Design:
  - RawEpisode: list of raw message dicts stored as Redis Hash
  - Per-user episode index: Redis list 'gt:index:{user_id}'
  - Max 500 episodes per user (oldest evicted). ~3-day project coverage.
  - Extraction is cached per (episode_id, query_hash) for 1h to avoid
    repeated LLM calls on the same data.

Pre-registered bugs:
  GT1 [HIGH]  episode_id key collides across users → always prefix 'gt:ep:{user_id}:{episode_id}'
  GT2 [HIGH]  Large episode (500 msgs) exceeds LLM context → slice last 50 messages for extraction
  GT3 [MED]   Extraction cache key hash collision (SHA256 truncated) → use full hex digest
  GT4 [MED]   Episode index grows unbounded if eviction fails → enforce at write, not lazily
  GT5 [LOW]   Raw message dict may have non-JSON-serializable values → json.dumps with default=str
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

MAX_EPISODES_PER_USER = 500     # GT4
EXTRACTION_TTL_SECONDS = 3600  # 1h extraction cache
EXTRACTION_CONTEXT_MSGS = 50   # GT2: slice to last N messages for LLM


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class RawEpisode:
    episode_id:  str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id:     str = ""
    channel_id:  str = ""
    messages:    List[Dict[str, Any]] = field(default_factory=list)  # raw [{role, content, ts}]
    started_at:  str = field(default_factory=_now_iso)
    ended_at:    str = field(default_factory=_now_iso)
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "RawEpisode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def message_count(self) -> int:
        return len(self.messages)


@dataclass
class ExtractedFacts:
    episode_id: str
    query:      str
    facts:      List[str]         # extracted factual statements
    entities:   List[str]         # named entities
    actions:    List[str]         # actions taken / decisions made
    extracted_at: str = field(default_factory=_now_iso)


# ─────────────────────────────────────────────
# GroundTruthStore
# ─────────────────────────────────────────────

class GroundTruthStore:
    """
    Write raw. Extract on read. Never summarize at write.

    Usage:
        store = GroundTruthStore(redis_client)
        eid = await store.write(user_id, channel_id, messages)
        facts = await store.extract(eid, "what tools did Ghost use?", llm_fn)
        episodes = await store.list_episodes(user_id)
    """

    def __init__(self, redis_client):
        self.redis = redis_client
        self._lock = asyncio.Lock()

    # ── Write ─────────────────────────────────

    async def write(
        self,
        user_id: str,
        channel_id: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict] = None,
        started_at: Optional[str] = None,
    ) -> str:
        """
        Persist raw messages as a RawEpisode. Returns episode_id.
        GT5: serializes with default=str for safety.
        """
        episode = RawEpisode(
            user_id=user_id,
            channel_id=channel_id,
            messages=messages,
            started_at=started_at or _now_iso(),
            ended_at=_now_iso(),
            metadata=metadata or {},
        )

        ep_key = f"gt:ep:{user_id}:{episode.episode_id}"  # GT1
        raw_json = json.dumps(episode.to_dict(), default=str)  # GT5

        async with self._lock:
            await self.redis.set(ep_key, raw_json)
            await self._append_index(user_id, episode.episode_id)
            await self._enforce_limit(user_id)  # GT4

        return episode.episode_id

    # ── Read ──────────────────────────────────

    async def read(self, user_id: str, episode_id: str) -> Optional[RawEpisode]:
        """Retrieve raw episode by ID."""
        raw = await self.redis.get(f"gt:ep:{user_id}:{episode_id}")
        if not raw:
            return None
        return RawEpisode.from_dict(json.loads(raw))

    async def list_episodes(
        self, user_id: str, limit: int = 50
    ) -> List[str]:
        """Return episode_ids for user, newest first."""
        index = await self._get_index(user_id)
        return list(reversed(index))[:limit]

    # ── Extract on read (MemMachine pattern) ──

    async def extract(
        self,
        user_id: str,
        episode_id: str,
        query: str,
        llm_fn: Callable,  # async fn(messages_list) → str
    ) -> Optional[ExtractedFacts]:
        """
        Extract structured facts from raw episode, scoped to query.
        Cached per (episode_id, query_hash) for 1h.
        GT2: slices to last EXTRACTION_CONTEXT_MSGS messages.
        GT3: uses full SHA256 hex digest as cache key.
        """
        # cache check
        cache_key = self._cache_key(user_id, episode_id, query)
        cached = await self.redis.get(cache_key)
        if cached:
            try:
                data = json.loads(cached)
                return ExtractedFacts(**data)
            except Exception:
                pass

        episode = await self.read(user_id, episode_id)
        if not episode:
            return None

        # GT2: slice context
        context_msgs = episode.messages[-EXTRACTION_CONTEXT_MSGS:]
        context_text = "\n".join(
            f"[{m.get('role', 'user')}] {m.get('content', '')}"
            for m in context_msgs
        )

        prompt = [
            {"role": "system", "content": (
                "You are an information extractor. Given a conversation log and a query, "
                "extract structured facts. Return JSON with keys: "
                "facts (list of factual statements), entities (list of named entities), "
                "actions (list of decisions/actions taken). Be concise."
            )},
            {"role": "user", "content": (
                f"Query: {query}\n\nConversation:\n{context_text}\n\n"
                "Return JSON only, no markdown fences:"
            )}
        ]

        try:
            raw_response = await llm_fn(prompt)
            clean = raw_response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(clean)
            facts = ExtractedFacts(
                episode_id=episode_id,
                query=query,
                facts=parsed.get("facts", []),
                entities=parsed.get("entities", []),
                actions=parsed.get("actions", []),
            )
            # cache it
            await self.redis.set(
                cache_key,
                json.dumps(asdict(facts) if hasattr(facts, '__dataclass_fields__')
                           else facts.__dict__),
                ex=EXTRACTION_TTL_SECONDS,
            )
            return facts
        except Exception:
            return None

    async def extract_batch(
        self,
        user_id: str,
        episode_ids: List[str],
        query: str,
        llm_fn: Callable,
        concurrency: int = 3,
    ) -> List[ExtractedFacts]:
        """Extract from multiple episodes concurrently."""
        sem = asyncio.Semaphore(concurrency)

        async def _extract_one(eid: str) -> Optional[ExtractedFacts]:
            async with sem:
                return await self.extract(user_id, eid, query, llm_fn)

        results = await asyncio.gather(
            *[_extract_one(eid) for eid in episode_ids],
            return_exceptions=True,
        )
        return [r for r in results if isinstance(r, ExtractedFacts)]

    # ── Internal ─────────────────────────────

    def _cache_key(self, user_id: str, episode_id: str, query: str) -> str:
        h = hashlib.sha256(f"{user_id}:{episode_id}:{query}".encode()).hexdigest()  # GT3
        return f"gt:extract_cache:{h}"

    async def _get_index(self, user_id: str) -> List[str]:
        raw = await self.redis.get(f"gt:index:{user_id}")
        return json.loads(raw) if raw else []

    async def _append_index(self, user_id: str, episode_id: str) -> None:
        index = await self._get_index(user_id)
        if episode_id not in index:
            index.append(episode_id)
        await self.redis.set(f"gt:index:{user_id}", json.dumps(index))

    async def _enforce_limit(self, user_id: str) -> None:
        """GT4: evict oldest episodes when at capacity."""
        index = await self._get_index(user_id)
        if len(index) <= MAX_EPISODES_PER_USER:
            return
        to_evict = index[:len(index) - MAX_EPISODES_PER_USER]
        for eid in to_evict:
            await self.redis.delete(f"gt:ep:{user_id}:{eid}")
        kept = index[len(index) - MAX_EPISODES_PER_USER:]
        await self.redis.set(f"gt:index:{user_id}", json.dumps(kept))
