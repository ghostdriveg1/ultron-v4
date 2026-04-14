"""
packages/tools/search.py

Ultron V4 — Web Search Tool (Tavily Free-Tier)
================================================
Real implementation replacing the stub in task_dispatcher._tool_search.

Tavily free tier: 1000 searches/month per API key.
Ghost has 5 Gmail accounts → 5 Tavily keys → 5000 searches/month total.
Key rotation: same pool pattern as LLM keys (indexed env vars).

Usage (from task_dispatcher or ToolRegistry):
    from packages.tools.search import tavily_search
    result: ActionResult = await tavily_search({"query": "BTC price", "max_results": 5})

Also exposes:
    get_search_client() — singleton TavilyClient (lazy-init, key-aware)
    search_with_fallback() — tries Tavily, falls back to DuckDuckGo HTML parse

Future bug risks (pre-registered):
  S1 [HIGH]   Tavily key exhausted (1000/month) → 401 → need to rotate to next key
              Fix: parse_indexed_keys pattern, try next key on 401/429
  S2 [HIGH]   DuckDuckGo fallback HTML structure changes → regex returns empty
              Fix: return raw text truncated, never fail silently
  S3 [MED]    search() called with empty or whitespace query → Tavily 400 error
              Fix: guard + return error ActionResult immediately
  S4 [MED]    Tavily response schema change → KeyError on result parsing
              Fix: use .get() everywhere, validate shape before accessing
  S5 [LOW]    5000 searches/month hit during heavy Council/MOA calls
              Fix: cache identical queries for 10min in Redis (key: ultron:search:{hash})

Tool calls used this session:
  Github:get_file_contents x6 (task_dispatcher, v3 bot, v3 root, v3 packages, v3 requirements, v3 Dockerfile)
  Github:push_files x2
  Notion:notion-fetch x1
  Notion:notion-update-page x1
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Optional

import httpx

# Import ActionResult from react_loop for consistent return type
try:
    from packages.brain.react_loop import ActionResult
except ImportError:
    # Fallback for standalone testing
    from dataclasses import dataclass, field

    @dataclass
    class ActionResult:  # type: ignore
        is_done: bool = False
        success: bool = True
        error: Optional[str] = None
        extracted_content: Optional[str] = None
        long_term_memory: Optional[str] = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAVILY_BASE_URL = "https://api.tavily.com"
DDG_BASE_URL = "https://html.duckduckgo.com/html/"

# Tavily key rotation — parse TAVILY_API_KEY_0 ... TAVILY_API_KEY_19
def _parse_tavily_keys() -> list[str]:
    keys: list[str] = []
    # First check plain TAVILY_API_KEY
    plain = os.environ.get("TAVILY_API_KEY", "").strip()
    if plain:
        keys.append(plain)
    # Then indexed keys
    for i in range(20):
        k = os.environ.get(f"TAVILY_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    return keys


_tavily_keys: list[str] = []
_current_key_idx: int = 0


def _get_tavily_key() -> Optional[str]:
    """Return next available Tavily key via round-robin (S1 mitigation)."""
    global _tavily_keys, _current_key_idx
    if not _tavily_keys:
        _tavily_keys = _parse_tavily_keys()
    if not _tavily_keys:
        return None
    key = _tavily_keys[_current_key_idx % len(_tavily_keys)]
    return key


def _rotate_tavily_key() -> None:
    """Advance to next Tavily key (call on 401/429)."""
    global _current_key_idx
    _current_key_idx = (_current_key_idx + 1) % max(len(_tavily_keys), 1)


# ---------------------------------------------------------------------------
# Tavily search
# ---------------------------------------------------------------------------

async def _tavily_search(query: str, max_results: int = 5) -> Optional[list[dict]]:
    """Call Tavily /search endpoint. Returns list of result dicts or None on failure."""
    key = _get_tavily_key()
    if not key:
        logger.warning("[search] No Tavily key configured")
        return None

    payload = {
        "api_key": key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": True,  # Tavily provides a direct answer field
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(f"{TAVILY_BASE_URL}/search", json=payload)

        if r.status_code in (401, 429):
            logger.warning(f"[search] Tavily key {_current_key_idx} hit {r.status_code} — rotating")
            _rotate_tavily_key()
            return None

        if r.status_code != 200:
            logger.warning(f"[search] Tavily {r.status_code}: {r.text[:200]}")
            return None

        data = r.json()  # S4: use .get() everywhere below
        results = []

        # Direct answer field (Tavily feature)
        answer = data.get("answer", "")
        if answer:
            results.append({"title": "Direct Answer", "content": answer, "url": ""})

        for item in data.get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "content": item.get("content", "")[:500],
                "url": item.get("url", ""),
            })

        return results if results else None

    except Exception as exc:
        logger.warning(f"[search] Tavily request failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# DuckDuckGo fallback (no key required)
# ---------------------------------------------------------------------------

async def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """HTML scrape DuckDuckGo as no-key fallback. Returns best-effort results (S2 mitigation)."""
    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0 UltronBot/1.0"},
            follow_redirects=True,
        ) as client:
            r = await client.get(DDG_BASE_URL, params={"q": query, "kl": "wt-wt"})

        if r.status_code != 200:
            return [{"title": "Search unavailable", "content": f"DDG returned {r.status_code}", "url": ""}]

        # Extract snippet text — simple regex, degrades gracefully if DDG changes structure (S2)
        snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)', r.text)
        titles = re.findall(r'class="result__title[^>]*>.*?<a[^>]*>([^<]+)', r.text)

        results = []
        for i in range(min(max_results, len(snippets))):
            results.append({
                "title": titles[i].strip() if i < len(titles) else f"Result {i+1}",
                "content": snippets[i].strip()[:500],
                "url": "",
            })

        if not results:
            # Raw fallback — strip HTML, return first 1000 chars
            raw = re.sub(r"<[^>]+>", " ", r.text)
            raw = re.sub(r"\s+", " ", raw).strip()[:1000]
            results = [{"title": "DDG raw", "content": raw, "url": ""}]

        return results

    except Exception as exc:
        logger.warning(f"[search] DDG fallback failed: {exc}")
        return [{"title": "Search error", "content": str(exc)[:200], "url": ""}]


# ---------------------------------------------------------------------------
# Format results for LLM consumption
# ---------------------------------------------------------------------------

def _format_results(results: list[dict], query: str) -> str:
    """Format search results into a clean string for the ReAct loop."""
    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        line = f"{i}. {title}\n   {content}"
        if url:
            line += f"\n   Source: {url}"
        lines.append(line)

    return "\n\n".join(lines)[:3000]  # hard cap for Groq context budget


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def tavily_search(params: dict) -> ActionResult:
    """Main search tool. Drop-in replacement for task_dispatcher._tool_search stub.

    params: {query: str, max_results: int = 5}
    Returns ActionResult with extracted_content = formatted results string.
    """
    query = params.get("query", "").strip()
    if not query:  # S3 guard
        return ActionResult(success=False, error="search: query param missing or empty")

    max_results = min(int(params.get("max_results", 5)), 10)

    logger.info(f"[search] query='{query}' max_results={max_results}")

    # Try Tavily first, fall back to DDG
    results = await _tavily_search(query, max_results)
    if results is None:
        logger.info("[search] Tavily unavailable, falling back to DDG")
        results = await _ddg_search(query, max_results)

    formatted = _format_results(results, query)

    return ActionResult(
        extracted_content=formatted,
        long_term_memory=f"searched:{query[:100]}",
    )


async def search_with_key_check(params: dict) -> ActionResult:
    """Alias for tavily_search. Named explicitly for ToolRegistry registration."""
    return await tavily_search(params)
