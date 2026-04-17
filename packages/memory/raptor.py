"""
packages/memory/raptor.py

RAPTOR tree — adapted from parthsarthi03/raptor for Ultron V4.

Key divergences from original:
    - No OpenAI dependency. Summariser = Groq/any provider via llm_router.
    - No FAISS. Retrieval delegates to ZillizStore (async).
    - No tiktoken. Token counting via simple split (good enough for Groq 8k ctx).
    - Clustering: UMAP + GMM from sklearn (same as original).
    - Tree stored in Zilliz: leaf nodes (layer=0) + summary nodes (layer>0).
    - build_tree(): embed chunks → cluster → summarise clusters → recurse.
    - query(): search summary nodes first (collapse_tree mode), fall back raw.
    - Async throughout. Single asyncio.Lock per user_id during build (W1 guard).

Pre-registered gut-feel bugs (other files that could break this one):
    R1 [HIGH]   worker.py calls build_tree() after every flush. If flush interval
                is short (e.g., 10 msgs) and sessions are long, build_tree() runs
                repeatedly on overlapping chunk sets → duplicate summary nodes.
                Fix: track last_build_ts per user_id in Redis; skip if < 5min ago.
    R2 [HIGH]   UMAP requires n_samples > n_neighbors. If chunk batch < 4,
                global_cluster_embeddings() crashes. Fix: skip RAPTOR if len < 4,
                write raw leaf nodes only.
    R3 [MED]    Groq summariser called inside build_tree() → key exhausted →
                AllKeysExhaustedError → tree build aborts, partial nodes in Zilliz.
                Fix: catch AllKeysExhaustedError, mark nodes as orphaned, continue.
    R4 [MED]    RAPTOR max_layers=3 default. For very long sessions (100+ chunks)
                tree depth may be insufficient → top-level summaries too coarse.
                Fix: auto-scale layers = max(2, log2(len(chunks)//4)).
    R5 [LOW]    sklearn GaussianMixture BIC search up to max_clusters=50 is slow
                on CPU for n_chunks > 200. Fix: cap max_clusters = min(50, n//4).
"""

import asyncio
import logging
import math
from typing import List, Optional

logger = logging.getLogger(__name__)

MAX_LAYERS = 3
MIN_CLUSTER_SIZE = 4   # R2: skip RAPTOR if fewer chunks
MAX_TOKENS_PER_CLUSTER = 3500
SUMMARY_MAX_TOKENS = 150

# Per-user locks to prevent concurrent tree builds (W1)
_build_locks: dict[str, asyncio.Lock] = {}
_build_locks_mutex = asyncio.Lock()


async def _get_user_lock(user_id: str) -> asyncio.Lock:
    async with _build_locks_mutex:
        if user_id not in _build_locks:
            _build_locks[user_id] = asyncio.Lock()
        return _build_locks[user_id]


def _count_tokens(text: str) -> int:
    """Approximate token count via whitespace split. ~4 chars/token is common."""
    return max(1, len(text) // 4)


def _cluster_embeddings(embeddings, threshold: float = 0.1):
    """
    RAPTOR clustering: UMAP dim-reduce → GMM soft clustering.
    Returns list of cluster label arrays per node (node can belong to multiple clusters).
    Mirrors cluster_utils.perform_clustering from parthsarthi03/raptor.
    """
    import numpy as np
    from sklearn.mixture import GaussianMixture  # type: ignore
    import umap  # type: ignore

    n = len(embeddings)
    dim = min(10, n - 2)  # UMAP can't reduce to >= n_samples
    if dim < 2:
        # Can't cluster — return all in one cluster
        return [[0] for _ in embeddings]

    arr = np.array(embeddings)
    n_neighbors = max(2, int(n ** 0.5))

    # Global reduction
    reduced = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric="cosine"
    ).fit_transform(arr)

    # GMM — BIC to find optimal clusters
    max_k = min(50, n // 4)  # R5
    if max_k < 2:
        return [[0] for _ in embeddings]

    bics = []
    for k in range(1, max_k + 1):
        gm = GaussianMixture(n_components=k, random_state=42)
        gm.fit(reduced)
        bics.append(gm.bic(reduced))

    best_k = bics.index(min(bics)) + 1
    gm = GaussianMixture(n_components=best_k, random_state=42)
    gm.fit(reduced)
    probs = gm.predict_proba(reduced)
    labels = [list(np.where(p > threshold)[0]) or [int(p.argmax())] for p in probs]
    return labels


class RaptorTree:
    """
    RAPTOR tree builder and retriever.

    Usage:
        raptor = RaptorTree(embedder, zilliz_store, llm_fn)
        await raptor.build_tree(user_id, chunks)
        context = await raptor.query(user_id, query_text, query_vec)
    """

    def __init__(self, embedder, zilliz_store, llm_fn):
        """
        Args:
            embedder:     Embedder instance.
            zilliz_store: ZillizStore instance.
            llm_fn:       Async callable(messages: list) → str.
                          Should be make_provider_llm_fn(pool) from llm_router.
        """
        self._embedder = embedder
        self._store = zilliz_store
        self._llm = llm_fn

    async def build_tree(
        self,
        user_id: str,
        chunks: List[str],
        max_layers: int = MAX_LAYERS,
    ) -> None:
        """
        Build RAPTOR tree from text chunks and upsert all nodes into Zilliz.

        Steps:
            1. Embed all chunks.
            2. Upsert as leaf nodes (layer=0).
            3. Cluster embeddings → summarise each cluster → upsert summary (layer=1).
            4. Recurse on summary texts up to max_layers.

        Skips if len(chunks) < MIN_CLUSTER_SIZE (R2).
        """
        if not chunks:
            logger.warning(f"build_tree called with empty chunks for {user_id}")
            return

        lock = await _get_user_lock(user_id)
        async with lock:  # W1: serialise per user_id
            await self._build_layer(user_id, chunks, layer=0, max_layers=max_layers)

    async def _build_layer(
        self,
        user_id: str,
        texts: List[str],
        layer: int,
        max_layers: int,
    ) -> None:
        """Recursive layer builder."""
        logger.info(f"RAPTOR build_layer user={user_id} layer={layer} n_chunks={len(texts)}")

        # Embed current layer
        embeddings = await self._embedder.encode(texts)

        # Upsert current layer to Zilliz
        node_type = "leaf" if layer == 0 else "summary"
        await self._store.upsert(user_id, texts, embeddings, node_type=node_type, layer=layer)

        # Base cases: max depth reached or too few chunks to cluster
        if layer >= max_layers or len(texts) < MIN_CLUSTER_SIZE:  # R2
            return

        # Cluster
        try:
            labels = _cluster_embeddings(embeddings)
        except Exception as exc:
            logger.warning(f"RAPTOR clustering failed at layer {layer}: {exc}")
            return

        # Group texts by cluster
        clusters: dict[int, List[str]] = {}
        for i, node_labels in enumerate(labels):
            for lbl in node_labels:
                clusters.setdefault(int(lbl), []).append(texts[i])

        # Enforce max token budget per cluster
        summary_texts: List[str] = []
        for cluster_texts in clusters.values():
            # Truncate cluster to MAX_TOKENS_PER_CLUSTER
            budget = 0
            selected = []
            for t in cluster_texts:
                toks = _count_tokens(t)
                if budget + toks > MAX_TOKENS_PER_CLUSTER:
                    break
                selected.append(t)
                budget += toks
            if not selected:
                selected = cluster_texts[:1]

            summary = await self._summarise_cluster(selected)
            if summary:
                summary_texts.append(summary)

        if not summary_texts:
            return

        # Recurse on summary layer
        await self._build_layer(user_id, summary_texts, layer=layer + 1, max_layers=max_layers)

    async def _summarise_cluster(self, texts: List[str]) -> Optional[str]:
        """Call LLM to summarise a cluster of texts. Returns None on failure (R3)."""
        combined = "\n\n".join(texts)
        messages = [
            {
                "role": "system",
                "content": "Summarise the following text concisely in 1-3 sentences. Return only the summary.",
            },
            {"role": "user", "content": combined[:6000]},  # stay under Groq ctx
        ]
        try:
            from packages.shared.exceptions import AllKeysExhaustedError  # type: ignore
            summary = await self._llm(messages)
            return summary.strip() if summary else None
        except Exception as exc:  # R3: AllKeysExhaustedError + any other
            logger.warning(f"RAPTOR summarise failed: {exc}")
            return None

    async def query(
        self,
        user_id: str,
        query_text: str,
        query_vec: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> str:
        """
        Retrieve relevant context for a query.

        Strategy (collapse_tree mode from original RAPTOR):
            1. Search summary nodes first (higher-level context).
            2. Augment with raw leaf nodes.
            3. Concatenate unique results, return as context string.

        Args:
            user_id:    Discord user_id.
            query_text: Raw query string.
            query_vec:  Pre-computed embedding (optional; computed if None).
            top_k:      Number of results per tier.

        Returns:
            Context string for LLM prompt injection.
        """
        if query_vec is None:
            query_vec = await self._embedder.encode_one(query_text)

        # Search summaries first (tree top-down)
        summary_hits = await self._store.search(
            user_id, query_vec, top_k=top_k, node_type_filter="summary"
        )
        # Then raw leaves
        leaf_hits = await self._store.search(
            user_id, query_vec, top_k=top_k, node_type_filter="leaf"
        )

        # Deduplicate and combine
        seen: set[str] = set()
        parts: List[str] = []
        for hit in summary_hits + leaf_hits:
            text = hit["text"]
            if text not in seen:
                seen.add(text)
                parts.append(text)

        if not parts:
            return ""  # caller falls back to pure LLM

        return "\n---\n".join(parts[:top_k * 2])
