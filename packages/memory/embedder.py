"""
packages/memory/embedder.py

Batch embedding layer using sentence-transformers all-MiniLM-L6-v2.
384-dim cosine-space vectors. CPU-only (HF Spaces free tier).

Design:
    - Lazy model load: model downloaded/loaded on first encode() call, not at import.
    - Async-safe: encode() runs in executor to avoid blocking the event loop.
    - Batch: encode list[str] → list[list[float]] in one model.encode() call.
    - Normalise: L2-norm so cosine sim == dot product (Zilliz COSINE metric).

Pre-registered gut-feel bugs (other files that could break this one):
    E1 [HIGH]   raptor.py calls encode() inside asyncio.gather() across multiple
                user_ids simultaneously. sentence-transformers model.encode() is
                NOT thread-safe when batch_size > 1. Fix: asyncio.Lock on encode().
    E2 [MED]    HF Space free tier has ~2GB RAM. all-MiniLM-L6-v2 uses ~90MB.
                If browser_agent.py Playwright is also loaded, OOM risk.
                Fix: track peak RAM at startup, warn via /health endpoint.
    E3 [LOW]    model.encode() returns numpy arrays. ZillizStore expects list[float].
                Implicit .tolist() missing → pymilvus TypeError on upsert.
                Fix: explicit .tolist() in encode() return.
"""

import asyncio
import logging
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


@lru_cache(maxsize=1)
def _load_model():
    """Load model once, cache in process. Called inside executor thread."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    return model


class Embedder:
    """
    Async batch embedder.

    Usage:
        embedder = Embedder()
        vecs = await embedder.encode(["hello world", "foo bar"])
        # vecs: List[List[float]], shape (2, 384)
    """

    def __init__(self):
        self._lock = asyncio.Lock()  # E1 guard: single encode at a time
        self._loop: asyncio.AbstractEventLoop | None = None

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a batch of texts to 384-dim normalised float vectors.

        Args:
            texts: Non-empty list of strings.

        Returns:
            List of float lists, one per input text.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("encode() called with empty texts list")

        async with self._lock:  # E1: serialise concurrent encode calls
            loop = asyncio.get_running_loop()
            # Run CPU-bound model.encode in thread pool — avoids blocking event loop
            embeddings = await loop.run_in_executor(
                None, self._sync_encode, texts
            )
        return embeddings

    def _sync_encode(self, texts: List[str]) -> List[List[float]]:
        """Blocking encode — called inside executor thread."""
        model = _load_model()
        # normalize_embeddings=True → L2 norm → cosine sim == dot product
        vecs = model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]  # E3: explicit .tolist()

    async def encode_one(self, text: str) -> List[float]:
        """Convenience: encode a single string."""
        results = await self.encode([text])
        return results[0]

    @property
    def dim(self) -> int:
        return EMBED_DIM
