---
title: Ultron Brain V4
app_file: packages/brain/main.py
app_port: 7860
sdk: docker
---

# Ultron Brain V4

FastAPI brain for Ultron AI OS. Multi-provider LLM routing (Groq + Cerebras + Together + OpenRouter + Gemini), ReAct agentic loop, per-channel Redis context, Discord bot interface.

## Architecture

- **Brain:** FastAPI on port 7860
- **Bot:** Discord (discord.py, same container)
- **LLM Pool:** 5-provider circuit-breaker key pool
- **Memory:** Redis (Upstash) per-channel context
- **Search:** Tavily free-tier + DuckDuckGo fallback

## Environment Variables (set in HF Space Secrets)

```
DISCORD_BOT_TOKEN=
INTERNAL_AUTH_TOKEN=
DISCORD_GHOST_USER_ID=
ALLOWED_DISCORD_USERS=
REDIS_URL=
GROQ_KEY_0=
GROQ_KEY_1=
...
CEREBRAS_KEY_0=
TOGETHER_KEY_0=
OPENROUTER_KEY_0=
GEMINI_KEY_0=
GEMINI_SENTINEL_KEY=
TAVILY_API_KEY_0=
```

## Repo

`github.com/ghostdriveg1/ultron-v4`
