"""
packages/brain/search.py

Ultron V4 — Web Search Tool (Tavily)
=====================================
Sole search provider: Tavily Search API (free tier, TAVILY_API_KEY).
Parallel AI permanently dropped (v33 decision).

Public API:
    result = await search(query="what is X", max_results=5, search_depth="basic")
    # returns list[SearchResult]

    SEARCH_TOOL_SCHEMA  — OpenAI-compatible tool schema for react_loop.py tool registry

SearchResult fields:
    title:   str   — page title
    url:     str   — page URL (empty string if Tavily returned None — SE5)
    content: str   — snippet / extracted content
    score:   float — Tavily relevance score 0.0-1.0

Rate-limit handling:
    Tavily free tier: 429 on burst. Exponential backoff: 1s, 2s, 4s (3 retries — SE2).

Pre-registered future bugs:
  SE1 [HIGH]  react_loop / task_dispatcher may call search() with positional args.
              Guard: all params after `query` are keyword-only (use `*` sentinel).
              If caller passes extra positional args, TypeError surfaces early — good.

  SE2 [MED]   Tavily 429 on free-tier burst. Backoff 1s/2s/4s, 3 retries.
              If all retries exhaust → returns [] (logged as WARNING). ReAct loop
              must handle empty search gracefully (already does — no crash, but
              LLM may hallucinate answer without context).

  SE3 [MED]   max_results clamped to [1, 10]. Tavily 400s on 0 or negative values.
              Clamp happens before API call. If caller passes max_results=0
              they get max_results=1 silently — log DEBUG note.

  SE4 [LOW]   search_depth validated: only 'basic' or 'advanced' accepted.
              Any other string → silently coerced to 'basic' + DEBUG log.
              eternal_loop_router.py may pass wrong string — this guards it.

  SE5 [LOW]   Tavily result['url'] can be None for some entries (no URL available).
              Normalized to empty string ''. Downstream code must handle ''
              (react_loop.py currently does `.startswith('http')` check — safe).

Tool calls used writing this file (v34):
    Github:get_file_contents x1 (packages/shared/config.py)
    Github:get_file_contents x1 (packages/brain/config_loader.py)
    Github:get_file_contents x1 (packages/brain/react_loop.py)  -- checked tool schema format
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from packages.shared.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    title:   str
    url:     str          # empty string if Tavily returned None (SE5)
    content: str
    score:   float = 0.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_MAX_RETRIES       = 3
_BACKOFF_BASE      = 1.0   # seconds; doubles each retry
_VALID_DEPTHS      = {"basic", "advanced"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_tavily_key() -> Optional[str]:
    """Return TAVILY_API_KEY or None if not configured."""
    s = get_settings()
    key = s.tavily_api_key
    return key if key else None


def _clamp_max_results(n: int) -> int:
    """SE3: clamp max_results to [1, 10]."""
    clamped = max(1, min(10, n))
    if clamped != n:
        logger.debug(f"[Search] max_results={n} clamped to {clamped} (SE3)")
    return clamped


def _validate_depth(depth: str) -> str:
    """SE4: coerce invalid search_depth to 'basic'."""
    if depth not in _VALID_DEPTHS:
        logger.debug(f"[Search] search_depth='{depth}' invalid, coerced to 'basic' (SE4)")
        return "basic"
    return depth


def _parse_results(raw: list[dict]) -> list[SearchResult]:
    """Normalize Tavily result dicts → SearchResult list."""
    out: list[SearchResult] = []
    for item in raw:
        url = item.get("url") or ""   # SE5: None → ""
        out.append(SearchResult(
            title   = str(item.get("title") or ""),
            url     = str(url),
            content = str(item.get("content") or ""),
            score   = float(item.get("score") or 0.0),
        ))
    return out


# ---------------------------------------------------------------------------
# Public: async search()
# ---------------------------------------------------------------------------

async def search(
    query: str,
    *,                        # SE1: all params after query are keyword-only
    max_results: int   = 5,
    search_depth: str  = "basic",
    include_answer: bool = False,
) -> list[SearchResult]:
    """
    Async Tavily search. Returns list[SearchResult], empty on error or no key.

    Args:
        query:          Search query string.
        max_results:    Number of results (1-10). Clamped if out of range (SE3).
        search_depth:   'basic' or 'advanced'. Coerced if invalid (SE4).
        include_answer: Ask Tavily for a synthesized answer (default off).

    Returns:
        list[SearchResult] — empty if key absent, network error, or all retries fail.
    """
    api_key = _get_tavily_key()
    if not api_key:
        logger.warning("[Search] TAVILY_API_KEY not set — returning empty results.")
        return []

    max_results  = _clamp_max_results(max_results)
    search_depth = _validate_depth(search_depth)

    payload = {
        "api_key":       api_key,
        "query":         query,
        "max_results":   max_results,
        "search_depth":  search_depth,
        "include_answer": include_answer,
    }

    delay = _BACKOFF_BASE
    async with httpx.AsyncClient(timeout=20.0) as client:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await client.post(_TAVILY_SEARCH_URL, json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    raw  = data.get("results") or []
                    results = _parse_results(raw)
                    logger.debug(
                        f"[Search] query={query!r} → {len(results)} results "
                        f"(depth={search_depth}, attempt={attempt})"
                    )
                    return results

                elif resp.status_code == 429:  # SE2: rate limit
                    if attempt < _MAX_RETRIES:
                        logger.warning(
                            f"[Search] Tavily 429 rate-limit (attempt {attempt}/{_MAX_RETRIES}). "
                            f"Retrying in {delay:.1f}s. (SE2)"
                        )
                        await asyncio.sleep(delay)
                        delay *= 2.0
                        continue
                    else:
                        logger.warning(
                            f"[Search] Tavily 429 — all {_MAX_RETRIES} retries exhausted. "
                            f"Returning empty. (SE2)"
                        )
                        return []

                elif resp.status_code == 401:
                    logger.error("[Search] Tavily 401 — invalid TAVILY_API_KEY. Search disabled.")
                    return []

                else:
                    body = resp.text[:200]
                    logger.warning(
                        f"[Search] Tavily {resp.status_code} on attempt {attempt}: {body}"
                    )
                    if attempt < _MAX_RETRIES:
                        await asyncio.sleep(delay)
                        delay *= 2.0
                        continue
                    return []

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                logger.warning(
                    f"[Search] Network error attempt {attempt}/{_MAX_RETRIES}: {exc}"
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(delay)
                    delay *= 2.0
                    continue
                return []

    return []  # unreachable, but satisfies type checker


# ---------------------------------------------------------------------------
# Convenience: search_to_context()
# ---------------------------------------------------------------------------

async def search_to_context(
    query: str,
    *,
    max_results: int = 5,
    search_depth: str = "basic",
) -> str:
    """
    Run search and format results as a context string for LLM injection.

    Returns a markdown-ish string:
        [1] Title\nURL\nSnippet\n\n[2] ...

    Returns empty string if no results.
    """
    results = await search(query, max_results=max_results, search_depth=search_depth)
    if not results:
        return ""

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        if r.url:
            lines.append(r.url)
        lines.append(r.content)
        lines.append("")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Tool schema for react_loop.py / task_dispatcher.py
# ---------------------------------------------------------------------------

SEARCH_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use when you need factual data, "
            "recent events, documentation, or anything not in your training knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10). Default 5.",
                    "default": 5,
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": "Search depth. 'basic' is faster; 'advanced' is thorough.",
                    "default": "basic",
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool handler for react_loop.py dispatch
# ---------------------------------------------------------------------------

async def handle_web_search_tool(tool_input: dict) -> str:
    """
    Called by react_loop.py tool dispatcher when LLM emits tool_use: web_search.

    Args:
        tool_input: dict with keys 'query', optionally 'max_results', 'search_depth'

    Returns:
        Formatted context string (SE1: uses keyword-only call to search()).
    """
    query        = str(tool_input.get("query", "")).strip()
    max_results  = int(tool_input.get("max_results", 5))
    search_depth = str(tool_input.get("search_depth", "basic"))

    if not query:
        logger.warning("[Search] handle_web_search_tool called with empty query.")
        return "No search query provided."

    context = await search_to_context(
        query,
        max_results=max_results,
        search_depth=search_depth,
    )
    return context if context else f"No results found for: {query}"
