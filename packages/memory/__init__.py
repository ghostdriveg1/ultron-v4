"""
packages/memory/__init__.py

Memory pipeline public API.

Exports:
    Embedder          — sentence-transformers batch embed, 384-dim
    ZillizStore       — async upsert/search via AsyncMilvusClient
    RaptorTree        — RAPTOR tree build + query (Groq summariser, no OpenAI)
    MemoryWorker      — Redis mem_buffer flush-to-Zilliz background loop

Pre-registered gut-feel bugs (other files that could break this one):
    W1 [HIGH]   worker.py flush fires while raptor.py build_tree() running on same
                user_id → partial tree written to Zilliz mid-build → corrupt nodes.
                Fix: asyncio.Lock per user_id before tree build.
    W2 [HIGH]   embedder.py loads model at import time → HF Space cold start adds
                ~8s. If main.py lifespan timeout is tight, Space may return 503
                before Brain is ready. Fix: lazy-load model inside Embedder.__init__
                only when first encode() call arrives.
    W3 [MED]    Zilliz free-tier has 1M vector cap per collection. If mem_buffer
                flush writes leaf + summary nodes without dedup, cap hit fast.
                Fix: upsert (not insert) keyed on chunk_id hash.
    W4 [MED]    RAPTOR summariser calls llm_router → pool key exhausted mid-build
                → AllKeysExhaustedError raised inside tree build → partial tree
                never cleaned up. Fix: catch in raptor.py, return partial tree.
    W5 [LOW]    sentence-transformers model download on first run (~90MB). HF Space
                build cache doesn't persist between deploys unless /data mount used.
                Fix: add all-MiniLM-L6-v2 to requirements.txt (triggers cache warm
                during build, not at runtime).
"""

from .embedder import Embedder
from .tier2_zilliz import ZillizStore
from .raptor import RaptorTree
from .worker import MemoryWorker

__all__ = ["Embedder", "ZillizStore", "RaptorTree", "MemoryWorker"]
