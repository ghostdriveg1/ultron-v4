"""
packages/brain/llm_router.py

Ultron V4 — Multi-Provider LLM Router
======================================
PROBLEM THIS FILE SOLVES (v16 — Ghost clarification):
  task_dispatcher.py's _make_groq_llm_fn() hardcoded Groq's endpoint.
  Key rotation pool has 5 providers, but calls never reached Cerebras/Together/
  OpenRouter/Gemini. Result: pool circuit breaker useless, 1/5th of free quota used.

FIX:
  This module provides make_provider_llm_fn(pool) — a single async call function
  that reads key_obj["provider"] from pool.get_key() and routes to the correct
  API endpoint + format for each provider. task_dispatcher.py imports THIS instead
  of defining its own Groq-only function.

Provider support matrix:
  groq        → api.groq.com/openai/v1/chat/completions   (OpenAI-compat, JSON mode, tool_calls)
  cerebras    → api.cerebras.ai/v1/chat/completions        (OpenAI-compat, JSON mode, tool_calls)
  together    → api.together.xyz/v1/chat/completions       (OpenAI-compat, JSON mode, NO tool_calls)
  openrouter  → openrouter.ai/api/v1/chat/completions      (OpenAI-compat, JSON mode, tool_calls)
  gemini      → generativelanguage.googleapis.com          (different API shape entirely)

Key rotation pool interface assumed:
  key_obj = await pool.get_key()
  key_obj = {"key_id": str, "key": str, "provider": str, "model": str}
  await pool.report_success(key_id: str)
  await pool.report_failure(key_id: str)
  raises AllKeysExhaustedError when pool is empty

Council/MOA note (v16 LOCKED DECISION):
  Council calls spread across ALL providers, NOT all-Groq or all-Gemini.
  Each expert call = pool.get_key() → could be any provider.
  This maximises free quota: 5 providers × 5 keys each = ~25 keys total.
  Ghost has 5 Gmail accounts → one key per provider per account = full coverage.

Future bug risks (pre-registered):
  R1 [HIGH]   Together AI does NOT support tool_calls (function calling).
              If pool routes a tool-using ReAct step to Together → response has no
              tool_call block → react_loop parser gets plain text → falls back to
              action_type=done prematurely. Fix: together_call() strips tools param,
              prompts model to embed action JSON in text → parser handles both.
  R2 [HIGH]   Gemini API shape is completely different (not OpenAI-compat).
              gemini_call() must translate messages[] → Gemini contents[] format.
              Role mapping: system→system_instruction (Gemini 1.5+), user→user,
              assistant→model. Failure to remap = 400 Invalid JSON.
  R3 [MED]    OpenRouter adds rate limits per model, not per key. Even with 5 keys
              pointing to same model, rate limit hits. Fix: vary model per key_obj
              (mistral-7b for key1, llama-70b for key2 etc).
  R4 [MED]    Cerebras returns tool_calls in a slightly different nesting than Groq.
              Must handle both choices[0].message.tool_calls and
              choices[0].message.content with embedded JSON.
  R5 [LOW]    pool.get_key() is async but some callers may forget await → TypeError.
              Already guarded here but watch for copy-paste in council.py.

Tool calls used this session:
  Github:push_files x1, Notion:notion-update-page x1
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider endpoint config
# ---------------------------------------------------------------------------

PROVIDER_CONFIG: dict[str, dict] = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "default_model": "llama-3.3-70b-versatile",
        "supports_tools": True,
        "supports_json_mode": True,
        "auth_header": "Bearer",
    },
    "cerebras": {
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "default_model": "llama3.1-70b",
        "supports_tools": True,
        "supports_json_mode": True,
        "auth_header": "Bearer",
    },
    "together": {
        "url": "https://api.together.xyz/v1/chat/completions",
        "default_model": "meta-llama/Llama-3-70b-chat-hf",
        "supports_tools": False,   # Bug R1: no function calling on Together
        "supports_json_mode": True,
        "auth_header": "Bearer",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "default_model": "mistralai/mistral-7b-instruct",
        "supports_tools": True,
        "supports_json_mode": True,
        "auth_header": "Bearer",
        "extra_headers": {
            "HTTP-Referer": "https://github.com/ghostdriveg1/ultron-v4",
            "X-Title": "Ultron",
        },
    },
    "gemini": {
        # Gemini uses a completely different REST shape — handled separately
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "default_model": "gemini-1.5-flash",   # flash for speed/cost; pro for Sentinel
        "supports_tools": True,
        "supports_json_mode": True,
        "auth_header": "key",   # query param not header
    },
}


# ---------------------------------------------------------------------------
# OpenAI-compatible call (Groq / Cerebras / Together / OpenRouter)
# ---------------------------------------------------------------------------

async def _openai_compat_call(
    url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
    supports_tools: bool,
    supports_json_mode: bool,
    extra_headers: Optional[dict] = None,
) -> Optional[dict]:
    """
    Single async call to any OpenAI-compatible endpoint.
    Returns normalized dict: {"content": str} or {"tool_name": str, "tool_args": dict}
    Returns None on failure.

    Bug R1: if supports_tools=False (Together), we drop the tools param and
    instead inject tool schema into the system prompt as JSON, then parse
    the model's text response for action_type + action_params.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.3,
    }

    if supports_json_mode:
        payload["response_format"] = {"type": "json_object"}

    if supports_tools and tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    elif not supports_tools and tools:
        # Bug R1 mitigation: Together / no-tool providers
        # Inject tool list as system message instruction
        tool_names = [t["function"]["name"] for t in tools]
        tool_inject = (
            f"\n\nYou MUST respond with JSON only. Choose one action from: {tool_names}. "
            f"Format: {{\"action_type\": \"<tool_name>\", \"action_params\": {{...}}, \"memory\": \"...\"}}"
        )
        # Append to last system message or prepend new one
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {"role": "system", "content": msgs[0]["content"] + tool_inject}
        else:
            msgs.insert(0, {"role": "system", "content": tool_inject})
        payload["messages"] = msgs

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code == 429 or resp.status_code >= 500:
                logger.warning(f"[LLMRouter] {url} returned {resp.status_code}")
                return None  # caller reports failure to pool

            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]["message"]

            # Handle tool_calls (Groq / Cerebras / OpenRouter)
            tool_calls = choice.get("tool_calls") or []
            if tool_calls:
                tc = tool_calls[0]
                try:
                    args = json.loads(tc["function"].get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                return {
                    "content": json.dumps({
                        "memory": "",
                        "action_type": tc["function"]["name"],
                        "action_params": args,
                    })
                }

            # Plain text / JSON content
            return {"content": choice.get("content", "{}")}

    except Exception as exc:
        logger.error(f"[LLMRouter] openai_compat_call error ({url}): {exc}")
        return None


# ---------------------------------------------------------------------------
# Gemini-specific call  (Bug R2: completely different API shape)
# ---------------------------------------------------------------------------

async def _gemini_call(
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
) -> Optional[dict]:
    """
    Calls Google Gemini REST API. Translates OpenAI message format → Gemini format.

    Bug R2: role mapping is non-trivial.
      OpenAI system → Gemini system_instruction (separate field, not in contents[])
      OpenAI user   → Gemini role: "user"
      OpenAI assistant → Gemini role: "model"
    Missing this mapping = 400 error.

    Returns normalized {"content": str} or None on failure.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    # Translate messages
    system_instruction = None
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")
        if role == "system":
            # Gemini 1.5+: system goes to system_instruction
            system_instruction = {"parts": [{"text": text}]}
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
        else:
            contents.append({"role": "user", "parts": [{"text": text}]})

    if not contents:
        # Gemini requires at least one user turn
        contents.append({"role": "user", "parts": [{"text": "Begin."}]})

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 512,
            "temperature": 0.3,
            "responseMimeType": "application/json",  # JSON mode
        },
    }
    if system_instruction:
        payload["system_instruction"] = system_instruction

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
            )

            if resp.status_code == 429 or resp.status_code >= 500:
                logger.warning(f"[LLMRouter] Gemini {resp.status_code}")
                return None

            resp.raise_for_status()
            data = resp.json()

            # Gemini response: candidates[0].content.parts[0].text
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return {"content": text}
            except (KeyError, IndexError) as exc:
                logger.error(f"[LLMRouter] Gemini parse error: {exc} | raw={str(data)[:200]}")
                return None

    except Exception as exc:
        logger.error(f"[LLMRouter] gemini_call error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public: make_provider_llm_fn — replaces task_dispatcher._make_groq_llm_fn
# ---------------------------------------------------------------------------

async def make_provider_llm_fn(pool: Any):
    """
    Returns an async llm_call_fn(messages, tools) -> Optional[dict] bound to
    the full key_rotation pool.

    This is what task_dispatcher.py should import and pass to ReActLoop.
    The pool selects any provider (Groq/Cerebras/Together/OpenRouter/Gemini).
    The call is routed to the correct endpoint based on key_obj["provider"].

    Council/MOA spread (v16 LOCKED):
      Every Council expert call goes through this function.
      Pool weighted RR ensures spread across providers automatically.
      No code change needed for Council — just call this fn per expert.

    Bug R5: pool.get_key() is async. Always await it.
    """
    async def llm_call_fn(
        messages: list[dict],
        tools: list[dict],
    ) -> Optional[dict]:
        if pool is None:
            logger.error("[LLMRouter] No pool — cannot call LLM")
            return None

        key_obj = None
        try:
            key_obj = await pool.get_key()  # raises AllKeysExhaustedError
        except Exception as exc:
            logger.error(f"[LLMRouter] Pool exhausted or error: {exc}")
            return None

        if key_obj is None:
            logger.error("[LLMRouter] pool.get_key() returned None")
            return None

        provider = key_obj.get("provider", "groq").lower()
        api_key = key_obj.get("key", "")
        key_id = key_obj.get("key_id", "")
        model = key_obj.get("model") or PROVIDER_CONFIG.get(provider, {}).get("default_model", "")

        logger.info(f"[LLMRouter] routing to provider={provider} model={model}")

        result = None
        try:
            if provider == "gemini":
                result = await _gemini_call(
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    tools=tools,
                )
            else:
                cfg = PROVIDER_CONFIG.get(provider, PROVIDER_CONFIG["groq"])
                result = await _openai_compat_call(
                    url=cfg["url"],
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    tools=tools,
                    supports_tools=cfg["supports_tools"],
                    supports_json_mode=cfg["supports_json_mode"],
                    extra_headers=cfg.get("extra_headers"),
                )
        except Exception as exc:
            logger.error(f"[LLMRouter] call error for provider={provider}: {exc}")
            result = None

        # Report outcome to pool
        if result is None:
            try:
                await pool.report_failure(key_id)
            except Exception:
                pass
        else:
            try:
                await pool.report_success(key_id)
            except Exception:
                pass

        return result

    return llm_call_fn


# ---------------------------------------------------------------------------
# Convenience: one-shot call without pool (for Sentinel / Council orchestrator)
# ---------------------------------------------------------------------------

async def call_provider(
    provider: str,
    api_key: str,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
) -> Optional[dict]:
    """
    Direct provider call without pool. Used by Sentinel (Gemini-only, always)
    and Council orchestrator when pinning a specific expert to a provider.

    Council usage pattern:
        expert_results = await asyncio.gather(
            call_provider("groq",       groq_key,       GROQ_MODEL,       msgs),
            call_provider("cerebras",   cerebras_key,   CEREBRAS_MODEL,   msgs),
            call_provider("together",   together_key,   TOGETHER_MODEL,   msgs),
            call_provider("openrouter", openrouter_key, OR_MODEL,         msgs),
            call_provider("gemini",     gemini_key,     GEMINI_MODEL,     msgs),
        )
    → 5 parallel expert opinions, all free-tier, none blocking each other.
    """
    tools = tools or []
    if provider == "gemini":
        return await _gemini_call(api_key, model, messages, tools)

    cfg = PROVIDER_CONFIG.get(provider, PROVIDER_CONFIG["groq"])
    return await _openai_compat_call(
        url=cfg["url"],
        api_key=api_key,
        model=model,
        messages=messages,
        tools=tools,
        supports_tools=cfg["supports_tools"],
        supports_json_mode=cfg["supports_json_mode"],
        extra_headers=cfg.get("extra_headers"),
    )
