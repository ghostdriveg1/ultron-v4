"""
packages/memory/worker.py

Memory flush worker — drains Redis mem_buffer → embeds → RAPTOR tree build → Zilliz.

Redis key: ultron:mem_buffer:{user_id}  (LIST, right-push on write, left-pop on flush)
Flush trigger: FLUSH_THRESHOLD items OR FLUSH_INTERVAL_SECS elapsed since last flush.
Run as background asyncio task started in main.py lifespan.

Flow per user_id:
    1. LRANGE ultron:mem_buffer:{uid} 0 FLUSH_THRESHOLD-1
    2. Decode JSON: [{"role": ..., "content": ..., "ts": ...}, ...]
    3. Chunk content into ~512-token windows (50-token overlap)
    4. Embedder.encode(chunks)
    5. RaptorTree.build_tree(uid, chunks)  -- upserts leaf + summary nodes
    6. LTRIM ultron:mem_buffer:{uid} FLUSH_THRESHOLD -1  (remove flushed items)
    7. Set ultron:mem_last_flush:{uid} = now (Redis SET EX 86400)

Error handling: Redis unavailable → log + skip (never crash worker loop).
                Zilliz unavailable → log + skip, retry next interval.

Pre-registered gut-feel bugs (other files that could break this one):
    MW1 [HIGH]  main.py lifespan cancels the worker task on shutdown. If flush is
                mid-write (after embed, before upsert), Zilliz gets orphan vectors
                without tree structure. Fix: asyncio.shield() the upsert call.
    MW2 [HIGH]  Redis LRANGE returns bytes, not str. JSON decode fails on raw bytes.
                Fix: decode bytes → str before json.loads().
    MW3 [MED]   Two worker iterations overlap if flush_interval_secs is shorter than
                build_tree() runtime. Fix: per-user_id asyncio.Lock in worker loop.
    MW4 [MED]   chunk_text() splits on whitespace — if a message is one long URL
                (2000 chars), it becomes a single chunk > 512 tokens. Groq context
                may handle it but RAPTOR BIC clustering will treat it as noise.
                Fix: hard-cap chunk size at 512 tokens with forced split.
    MW5 [LOW]   mem_buffer list unbounded if flush never fires (e.g. Zilliz always
                fails). Fix: LTRIM to 200 items max as safety in ensure_buffer_size().
"""

import asyncio
import json
import logging
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

FLUSH_THRESHOLD = 10          # flush after this many messages buffered
FLUSH_INTERVAL_SECS = 300     # flush every 5 min regardless of count
CHUNK_SIZE_CHARS = 2000       # ~500 tokens at 4 chars/token
CHUNK_OVERLAP_CHARS = 200     # ~50 tokens overlap
MEM_BUFFER_PREFIX = "ultron:mem_buffer"
MEM_LAST_FLUSH_PREFIX = "ultron:mem_last_flush"
MAX_BUFFER_SIZE = 200         # MW5: safety cap
FLUSH_SLEEP_SECS = 60         # worker poll interval


def _chunk_text(text: str) -> List[str]:
    """
    Split text into overlapping chunks of ~CHUNK_SIZE_CHARS characters.
    MW4: hard-cap each chunk to CHUNK_SIZE_CHARS.
    """
    if len(text) <= CHUNK_SIZE_CHARS:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE_CHARS
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP_CHARS  # overlap
        if start >= len(text) - CHUNK_OVERLAP_CHARS:
            break
    return chunks


def _messages_to_chunks(messages: List[dict]) -> List[str]:
    """Convert role-content message dicts to text chunks."""
    chunks: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content or not content.strip():
            continue
        text = f"{role}: {content}"
        chunks.extend(_chunk_text(text))
    return chunks


class MemoryWorker:
    """
    Background memory flush worker.

    Usage (in main.py lifespan):
        worker = MemoryWorker(redis_client, embedder, raptor_tree)
        task = asyncio.create_task(worker.run())
        # on shutdown:
        task.cancel()
    """

    def __init__(self, redis, embedder, raptor_tree):
        """
        Args:
            redis:       aioredis / redis.asyncio client (already connected).
            embedder:    Embedder instance.
            raptor_tree: RaptorTree instance.
        """
        self._redis = redis
        self._embedder = embedder
        self._raptor = raptor_tree
        self._user_locks: dict[str, asyncio.Lock] = {}  # MW3
        self._lock_mutex = asyncio.Lock()

    async def _get_lock(self, user_id: str) -> asyncio.Lock:
        async with self._lock_mutex:
            if user_id not in self._user_locks:
                self._user_locks[user_id] = asyncio.Lock()
            return self._user_locks[user_id]

    async def run(self) -> None:
        """Main worker loop. Runs indefinitely until cancelled."""
        logger.info("MemoryWorker started")
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                logger.info("MemoryWorker cancelled")
                raise
            except Exception as exc:
                logger.error(f"MemoryWorker tick error: {exc}")
            await asyncio.sleep(FLUSH_SLEEP_SECS)

    async def _tick(self) -> None:
        """Single flush cycle: scan all active buffers."""
        try:
            # Find all mem_buffer keys
            keys = await self._redis.keys(f"{MEM_BUFFER_PREFIX}:*")
        except Exception as exc:
            logger.warning(f"Redis keys scan failed: {exc}")
            return

        if not keys:
            return

        # MW2: keys may be bytes
        user_ids = []
        for k in keys:
            key_str = k.decode() if isinstance(k, bytes) else k
            uid = key_str.replace(f"{MEM_BUFFER_PREFIX}:", "")
            user_ids.append(uid)

        # Flush each user concurrently (bounded to 5 at a time)
        sem = asyncio.Semaphore(5)
        tasks = [self._flush_user(uid, sem) for uid in user_ids]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _flush_user(self, user_id: str, sem: asyncio.Semaphore) -> None:
        """Flush one user's mem_buffer if threshold or interval reached."""
        async with sem:
            lock = await self._get_lock(user_id)  # MW3
            async with lock:
                await self._do_flush(user_id)

    async def _do_flush(self, user_id: str) -> None:
        """Core flush logic for a single user_id."""
        buf_key = f"{MEM_BUFFER_PREFIX}:{user_id}"
        last_key = f"{MEM_LAST_FLUSH_PREFIX}:{user_id}"

        try:
            buf_len = await self._redis.llen(buf_key)
        except Exception as exc:
            logger.warning(f"Redis llen failed for {user_id}: {exc}")
            return

        if buf_len == 0:
            return

        # Check flush triggers
        should_flush = buf_len >= FLUSH_THRESHOLD
        if not should_flush:
            try:
                last_flush = await self._redis.get(last_key)
                if last_flush:
                    elapsed = time.time() - float(last_flush)
                    should_flush = elapsed >= FLUSH_INTERVAL_SECS
            except Exception:
                should_flush = False

        if not should_flush:
            return

        # LRANGE: get up to FLUSH_THRESHOLD items
        try:
            raw_items = await self._redis.lrange(buf_key, 0, FLUSH_THRESHOLD - 1)
        except Exception as exc:
            logger.warning(f"Redis lrange failed for {user_id}: {exc}")
            return

        messages = []
        for item in raw_items:
            # MW2: decode bytes
            item_str = item.decode() if isinstance(item, bytes) else item
            try:
                messages.append(json.loads(item_str))
            except json.JSONDecodeError:
                messages.append({"role": "user", "content": item_str})

        chunks = _messages_to_chunks(messages)
        if not chunks:
            logger.debug(f"No chunks extracted for {user_id}")
            return

        logger.info(f"MemoryWorker flushing {len(chunks)} chunks for {user_id}")

        try:
            # asyncio.shield: protect upsert from cancellation (MW1)
            await asyncio.shield(
                self._raptor.build_tree(user_id, chunks)
            )
        except Exception as exc:
            logger.error(f"RAPTOR build_tree failed for {user_id}: {exc}")
            return  # don't trim buffer if build failed

        # Trim flushed items from buffer
        try:
            await self._redis.ltrim(buf_key, len(raw_items), -1)
            await self._redis.set(last_key, str(time.time()), ex=86400)
        except Exception as exc:
            logger.warning(f"Redis post-flush trim failed for {user_id}: {exc}")

        # MW5: safety cap
        try:
            await self._redis.ltrim(buf_key, -MAX_BUFFER_SIZE, -1)
        except Exception:
            pass

        logger.info(f"MemoryWorker flush complete for {user_id}")
