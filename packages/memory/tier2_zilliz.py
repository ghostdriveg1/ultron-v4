"""
packages/memory/tier2_zilliz.py

Zilliz (Milvus-as-a-Service) async vector store.

Architecture:
    - AsyncMilvusClient (pymilvus) — async context manager pattern.
    - Collection per shard: ultron_mem_{shard_id}, shard = hash(user_id) % 15.
    - Schema: id (VARCHAR PK, 64 chars), user_id (VARCHAR), chunk_id (VARCHAR),
      text (VARCHAR, 2048), node_type (VARCHAR, 32), layer (INT64),
      vector (FLOAT_VECTOR, 384), ts (INT64 unix ms).
    - Metric: COSINE. Index: AUTOINDEX (Zilliz manages HNSW internally).
    - Upsert (not insert) keyed on chunk_id — dedup guard (W3).
    - search() returns top_k results as list[dict].
    - ensure_collection() is idempotent — safe to call on every startup.

Pre-registered gut-feel bugs (other files that could break this one):
    Z1 [HIGH]   Zilliz free cluster goes cold after inactivity (~10min). First
                upsert/search after cold start → gRPC timeout (default 10s).
                Fix: set timeout=30 on all ops; /health pings each shard daily.
    Z2 [HIGH]   pymilvus AsyncMilvusClient is EXPERIMENTAL (their own warning).
                If Zilliz server ≥ 2.5 returns new protobuf field, client may
                raise decode error. Fix: pin pymilvus==2.4.x in requirements.txt.
    Z3 [MED]    ensure_collection() called concurrently for same shard on startup
                → duplicate create_collection race. Fix: asyncio.Lock per shard.
    Z4 [MED]    FLOAT_VECTOR dim mismatch (embedder returns 384, collection created
                with wrong dim) → MilvusException on upsert. Fix: assert dim==384
                in ensure_collection() describe result.
    Z5 [LOW]    VARCHAR PK max_length=64 but chunk_id is SHA256 hex (64 chars)
                → exact boundary. If pymilvus adds null terminator, will overflow.
                Fix: truncate chunk_id to 60 chars.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NUM_SHARDS = 15
EMBED_DIM = 384
COLLECTION_PREFIX = "ultron_mem"
MAX_TEXT_LEN = 2048
CHUNK_ID_MAX = 60  # Z5: stay under 64 VARCHAR PK limit


def _shard(user_id: str) -> int:
    """Deterministic shard index for a user_id. Matches locked Zilliz sharding rule."""
    return int(hashlib.md5(user_id.encode()).hexdigest(), 16) % NUM_SHARDS


def _collection_name(user_id: str) -> str:
    return f"{COLLECTION_PREFIX}_{_shard(user_id)}"


def _chunk_id(text: str, user_id: str) -> str:
    """Stable dedup key: SHA256(user_id + text), truncated to 60 chars."""
    raw = hashlib.sha256(f"{user_id}:{text}".encode()).hexdigest()
    return raw[:CHUNK_ID_MAX]


class ZillizStore:
    """
    Async Zilliz vector store.

    Usage:
        store = ZillizStore(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        await store.ensure_collection(user_id)
        await store.upsert(user_id, chunks, embeddings, node_type="leaf", layer=0)
        results = await store.search(user_id, query_vec, top_k=5)
        await store.close()
    """

    def __init__(self, uri: str, token: str):
        self._uri = uri
        self._token = token
        self._clients: Dict[int, Any] = {}   # shard_id → AsyncMilvusClient
        self._ensure_locks: Dict[int, asyncio.Lock] = {}  # Z3: per-shard lock
        self._global_lock = asyncio.Lock()

    async def _get_client(self, shard: int) -> Any:
        """Lazy init AsyncMilvusClient per shard."""
        if shard not in self._clients:
            async with self._global_lock:
                if shard not in self._clients:  # double-check after lock
                    from pymilvus import AsyncMilvusClient  # type: ignore
                    client = AsyncMilvusClient(uri=self._uri, token=self._token)
                    await client._connect()
                    self._clients[shard] = client
                    self._ensure_locks[shard] = asyncio.Lock()
        return self._clients[shard]

    async def ensure_collection(self, user_id: str) -> None:
        """
        Idempotent: create collection + index if not exists.
        Safe to call every startup. Z3: serialised per shard.
        """
        shard = _shard(user_id)
        client = await self._get_client(shard)
        col = _collection_name(user_id)

        # Ensure lock exists (may be first call)
        if shard not in self._ensure_locks:
            self._ensure_locks[shard] = asyncio.Lock()

        async with self._ensure_locks[shard]:  # Z3
            exists = await client.has_collection(col, timeout=30)
            if exists:
                logger.debug(f"Collection {col} already exists")
                return

            logger.info(f"Creating collection {col} (shard {shard})")
            from pymilvus import DataType  # type: ignore
            from pymilvus.milvus_client import IndexParams  # type: ignore

            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("user_id", DataType.VARCHAR, max_length=128)
            schema.add_field("chunk_id", DataType.VARCHAR, max_length=64)
            schema.add_field("text", DataType.VARCHAR, max_length=MAX_TEXT_LEN)
            schema.add_field("node_type", DataType.VARCHAR, max_length=32)  # leaf|summary
            schema.add_field("layer", DataType.INT64)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBED_DIM)  # Z4
            schema.add_field("ts", DataType.INT64)

            idx = IndexParams()
            idx.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")

            await client.create_collection(
                col, schema=schema, index_params=idx, timeout=30
            )
            logger.info(f"Collection {col} created")

    async def upsert(
        self,
        user_id: str,
        texts: List[str],
        embeddings: List[List[float]],
        node_type: str = "leaf",
        layer: int = 0,
    ) -> int:
        """
        Upsert text chunks + embeddings into user's shard.

        Args:
            user_id:    Discord user_id string.
            texts:      Raw text chunks (parallel with embeddings).
            embeddings: 384-dim float vectors.
            node_type:  'leaf' or 'summary'.
            layer:      RAPTOR tree layer (0 = leaf).

        Returns:
            Number of rows upserted.
        """
        if len(texts) != len(embeddings):
            raise ValueError(f"texts/embeddings length mismatch: {len(texts)} vs {len(embeddings)}")
        if not texts:
            return 0

        await self.ensure_collection(user_id)
        shard = _shard(user_id)
        client = await self._get_client(shard)
        col = _collection_name(user_id)
        ts = int(time.time() * 1000)

        rows = []
        for text, vec in zip(texts, embeddings):
            cid = _chunk_id(text, user_id)
            rows.append({
                "id": cid,           # PK = chunk_id for upsert dedup
                "user_id": user_id,
                "chunk_id": cid,
                "text": text[:MAX_TEXT_LEN],
                "node_type": node_type,
                "layer": layer,
                "vector": vec,
                "ts": ts,
            })

        res = await client.upsert(col, rows, timeout=30)
        count = res.get("upsert_count", len(rows))
        logger.info(f"Zilliz upsert {count} rows → {col} (layer={layer}, type={node_type})")
        return count

    async def search(
        self,
        user_id: str,
        query_vec: List[float],
        top_k: int = 5,
        node_type_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Cosine ANN search in user's shard.

        Args:
            user_id:           Discord user_id.
            query_vec:         384-dim query embedding.
            top_k:             Number of results.
            node_type_filter:  Optional 'leaf' or 'summary' filter.

        Returns:
            List of dicts with keys: text, node_type, layer, score, ts.
        """
        shard = _shard(user_id)
        col = _collection_name(user_id)
        client = await self._get_client(shard)

        filter_expr = ""
        if node_type_filter:
            filter_expr = f'node_type == "{node_type_filter}"'

        try:
            hits = await client.search(
                collection_name=col,
                data=[query_vec],
                anns_field="vector",
                limit=top_k,
                output_fields=["text", "node_type", "layer", "ts"],
                filter=filter_expr or "",
                timeout=30,
            )
        except Exception as exc:
            logger.warning(f"Zilliz search failed for {user_id}: {exc}")
            return []

        results = []
        for hit in hits[0]:  # hits[0] = results for first query vector
            results.append({
                "text": hit["entity"].get("text", ""),
                "node_type": hit["entity"].get("node_type", "leaf"),
                "layer": hit["entity"].get("layer", 0),
                "score": hit.get("distance", 0.0),
                "ts": hit["entity"].get("ts", 0),
            })
        return results

    async def close(self) -> None:
        """Close all open client connections."""
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
