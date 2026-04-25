"""
packages/brain/llm_router.py

Ultron V4 — Multi-Provider LLM Router
======================================
v31 update: +SambaNova +Fireworks +HuggingFace Inference API providers.
All 3 new providers use OpenAI-compatible chat completions endpoints.

Provider support matrix (full):
  groq        → api.groq.com/openai/v1/chat/completions       (tools ✓, json ✓)
  cerebras    → api.cerebras.ai/v1/chat/completions            (tools ✓, json ✓)
  together    → api.together.xyz/v1/chat/completions           (tools ✗, json ✓) [R1]
  openrouter  → openrouter.ai/api/v1/chat/completions          (tools ✓, json ✓)
  gemini      → generativelanguage.googleapis.com              (different shape) [R2]
  sambanova   → api.sambanova.ai/v1/chat/completions           (tools ✓, json ✓)
  fireworks   → api.fireworks.ai/inference/v1/chat/completions (tools ✓, json ✓)
  hf          → api-inference.huggingface.co/...               (tools ✗, json ✓) [R6]

Future bug risks (pre-registered, v31 additions):
  R1 [HIGH]   Together: no tool_calls → R1 mitigation already implemented.
  R2 [HIGH]   Gemini: non-OpenAI shape → _gemini_call() handles.
  R3 [MED]    OpenRouter: per-model rate limits, not per-key.
  R4 [MED]    Cerebras: tool_calls nesting differs from Groq.
  R5 [LOW]    get_key() async — always await.
  R6 [MED]    HuggingFace Inference API: tool_calls support varies by model.
              Llama-3.3-70B-Instruct supports it but HF cold-start (~20s) may timeout.
              Set timeout=90s for HF calls. treat non-200 as failure → pool.report_failure.
  R7 [LOW]    SambaNova returns 429 with Retry-After header. Current code treats 429 as
              immediate report_failure. Future: parse Retry-After, set reset_at accordingly
              instead of using fixed PROVIDER_COOLDOWN_SECONDS.
  R8 [LOW]    Fireworks model string includes "accounts/fireworks/models/" prefix.
              If key_obj["model"] is set without this prefix (user config error),
              Fireworks returns 404 model-not-found. Validate model string at _fireworks_call.

Tool calls used writing this file (v31):
  Github:get_file_contents x1 (llm_router.py)
  Github:push_files x1 (batch commit)
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
        "timeout": 30,
    },
    "cerebras": {
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "default_model": "llama3.1-70b",
        "supports_tools": True,
        "supports_json_mode": True,
        "auth_header": "Bearer",
        "timeout": 30,
    },
    "together": {
        "url": "https://api.together.xyz/v1/chat/completions",
        "default_model": "meta-llama/Llama-3-70b-chat-hf",
        "supports_tools": False,   # R1: no function calling
        "supports_json_mode": True,
        "auth_header": "Bearer",
        "timeout": 30,
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
        "timeout": 30,
    },
    "gemini": {
        # Non-OpenAI shape — handled by _gemini_call()
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "default_model": "gemini-1.5-flash",
        "supports_tools": True,
        "supports_json_mode": True,
        "auth_header": "key",
        "timeout": 30,
    },
    "sambanova": {
        "url": "https://api.sambanova.ai/v1/chat/completions",
        "default_model": "Meta-Llama-3.1-70B-Instruct",
        "supports_tools": True,
        "supports_json_mode": True,
        "auth_header": "Bearer",
        "timeout": 60,   # R7: SambaNova can be slower
    },
    "fireworks": {
        "url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "default_model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "supports_tools": True,
        "supports_json_mode": True,
        "auth_header": "Bearer",
        "timeout": 60,
    },
    "hf": {
        "url": "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct/v1/chat/completions",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct",
        "supports_tools": False,  # R6: tool_calls varies by model on HF
        "supports_json_mode": True,
        "auth_header": "Bearer",
        "timeout": 90,  # R6: HF cold-start ~20s
    },
}


# ---------------------------------------------------------------------------
# OpenAI-compatible call (Groq / Cerebras / Together / OpenRouter / SambaNova / Fireworks / HF)
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
    timeout: int = 30,
) -> Optional[dict]:
    """
    Single async call to any OpenAI-compatible endpoint.
    Returns normalized dict: {"content": str} or {"tool_name": str, "tool_args": dict}
    Returns None on failure (caller reports failure to pool).

    R1 mitigation: if supports_tools=False, inject tool schema into system prompt.
    R6: HF uses timeout=90 — caller passes via PROVIDER_CONFIG.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload: dict[str, Any] = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  512,
        "temperature": 0.3,
    }

    if supports_json_mode:
        payload["response_format"] = {"type": "json_object"}

    if supports_tools and tools:
        payload["tools"]       = tools
        payload["tool_choice"] = "auto"
    elif not supports_tools and tools:
        # R1 / R6 mitigation: inject tool list into system prompt
        tool_names = [t["function"]["name"] for t in tools]
        tool_inject = (
            f"\n\nYou MUST respond with JSON only. Choose one action from: {tool_names}. "
            f'Format: {{"action_type": "<tool_name>", "action_params": {{...}}, "memory": "..."}}'
        )
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {"role": "system", "content": msgs[0]["content"] + tool_inject}
        else:
            msgs.insert(0, {"role": "system", "content": tool_inject})
        payload["messages"] = msgs

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code == 429 or resp.status_code >= 500:
                logger.warning(f"[LLMRouter] {url} returned {resp.status_code}")
                return None

            resp.raise_for_status()
            data   = resp.json()
            choice = data["choices"][0]["message"]

            # Handle tool_calls
            tool_calls = choice.get("tool_calls") or []
            if tool_calls:
                tc = tool_calls[0]
                try:
                    args = json.loads(tc["function"].get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                return {
                    "content": json.dumps({
                        "memory":       "",
                        "action_type":  tc["function"]["name"],
                        "action_params": args,
                    })
                }

            return {"content": choice.get("content", "{}")}

    except Exception as exc:
        logger.error(f"[LLMRouter] openai_compat_call error ({url}): {exc}")
        return None


# ---------------------------------------------------------------------------
# Gemini-specific call  (R2: completely different API shape)
# ---------------------------------------------------------------------------

async def _gemini_call(
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
) -> Optional[dict]:
    """
    Calls Google Gemini REST API. Translates OpenAI message format → Gemini format.
    R2: role mapping — system→system_instruction, assistant→model.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    system_instruction = None
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")
        if role == "system":
            system_instruction = {"parts": [{"text": text}]}
        elif role == "assistant":
            contents.append({"role": "model",  "parts": [{"text": text}]})
        else:
            contents.append({"role": "user",   "parts": [{"text": text}]})

    if not contents:
        contents.append({"role": "user", "parts": [{"text": "Begin."}]})

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 512,
            "temperature":     0.3,
            "responseMimeType": "application/json",
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
                return None
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"content": text}
    except Exception as exc:
        logger.error(f"[LLMRouter] gemini_call error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Provider-specific helpers for new providers (v31)
# ---------------------------------------------------------------------------

async def _sambanova_call(
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
) -> Optional[dict]:
    """SambaNova Cloud — OpenAI-compat. R7: 429 has Retry-After header (future)."""
    cfg = PROVIDER_CONFIG["sambanova"]
    return await _openai_compat_call(
        url=cfg["url"],
        api_key=api_key,
        model=model or cfg["default_model"],
        messages=messages,
        tools=tools,
        supports_tools=cfg["supports_tools"],
        supports_json_mode=cfg["supports_json_mode"],
        timeout=cfg["timeout"],
    )


async def _fireworks_call(
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
) -> Optional[dict]:
    """Fireworks AI — OpenAI-compat. R8: model must include accounts/fireworks/models/ prefix."""
    cfg = PROVIDER_CONFIG["fireworks"]
    # R8: auto-prefix if missing
    if model and not model.startswith("accounts/"):
        model = f"accounts/fireworks/models/{model}"
    return await _openai_compat_call(
        url=cfg["url"],
        api_key=api_key,
        model=model or cfg["default_model"],
        messages=messages,
        tools=tools,
        supports_tools=cfg["supports_tools"],
        supports_json_mode=cfg["supports_json_mode"],
        timeout=cfg["timeout"],
    )


async def _hf_call(
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
) -> Optional[dict]:
    """HuggingFace Inference API. R6: cold-start ~20s, timeout=90s, tools vary by model."""
    cfg = PROVIDER_CONFIG["hf"]
    # HF URL is model-specific — build dynamically if model differs from default
    _model = model or cfg["default_model"]
    _url = (
        f"https://api-inference.huggingface.co/models/{_model}/v1/chat/completions"
    )
    return await _openai_compat_call(
        url=_url,
        api_key=api_key,
        model=_model,
        messages=messages,
        tools=tools,
        supports_tools=False,  # R6: safer to disable, inject via system prompt
        supports_json_mode=cfg["supports_json_mode"],
        timeout=cfg["timeout"],
    )


# ---------------------------------------------------------------------------
# Router dispatch
# ---------------------------------------------------------------------------

async def _route_call(
    provider: str,
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
) -> Optional[dict]:
    """Route to provider-specific call function."""
    if provider == "gemini":
        return await _gemini_call(api_key, model, messages, tools)
    if provider == "sambanova":
        return await _sambanova_call(api_key, model, messages, tools)
    if provider == "fireworks":
        return await _fireworks_call(api_key, model, messages, tools)
    if provider == "hf":
        return await _hf_call(api_key, model, messages, tools)
    # groq / cerebras / together / openrouter
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
        timeout=cfg.get("timeout", 30),
    )


# ---------------------------------------------------------------------------
# Public: make_provider_llm_fn — replaces task_dispatcher._make_groq_llm_fn
# ---------------------------------------------------------------------------

async def make_provider_llm_fn(pool: Any):
    """
    Returns async llm_call_fn(messages, tools) -> Optional[dict] bound to pool.
    Pool selects any provider via weighted RR — now includes SambaNova/Fireworks/HF.
    R5: pool.get_key() is async. Always await.
    """
    async def llm_call_fn(
        messages: list[dict],
        tools: list[dict],
    ) -> Optional[dict]:
        if pool is None:
            logger.error("[LLMRouter] No pool")
            return None

        key_obj = None
        try:
            key_obj = await pool.get_key()
        except Exception as exc:
            logger.error(f"[LLMRouter] Pool exhausted: {exc}")
            return None

        if key_obj is None:
            return None

        provider = key_obj.get("provider", "groq").lower()
        api_key  = key_obj.get("key", "")
        key_id   = key_obj.get("key_id", "")
        model    = key_obj.get("model") or PROVIDER_CONFIG.get(provider, {}).get("default_model", "")

        logger.info(f"[LLMRouter] routing provider={provider} model={model}")

        result = None
        try:
            result = await _route_call(provider, api_key, model, messages, tools)
        except Exception as exc:
            logger.error(f"[LLMRouter] call error provider={provider}: {exc}")

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
# Convenience: one-shot call without pool (Sentinel / Council orchestrator)
# ---------------------------------------------------------------------------

async def call_provider(
    provider: str,
    api_key: str,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
) -> Optional[dict]:
    """
    Direct call without pool. Used by Sentinel (Gemini-only)
    and Council orchestrator when pinning an expert to a specific provider.
    """
    return await _route_call(provider, api_key, model, messages, tools or [])
