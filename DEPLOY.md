# Ultron V4 — Phase 7 Production Deploy Checklist

> Pre-deploy audit completed v34. All code wirings confirmed. One missing dep fixed (networkx).
> Follow this checklist in order.

---

## Step 0 — NANCE Stage 7 (CF Worker manual paste)

Blocked: `ghostss4cg4` repo returns 403 on token push. Ghost must paste manually.

1. Open `ghostss4cg4/ghost-memory-mcp` on GitHub UI
2. Edit `src/worker.js` → paste content from NANCE v0.6.0 (session v29)
3. Edit `wrangler.toml` → paste updated version (PROXY_CACHE binding placeholder)
4. In Cloudflare dashboard (acct `c2ed2ecab1a35b2cd2095849cb69ab10`):
   ```
   wrangler kv namespace create PROXY_CACHE
   ```
   Copy returned namespace ID → paste into `wrangler.toml` as `PROXY_CACHE` binding ID.
5. Deploy:
   ```
   wrangler deploy
   ```
6. Inject secrets into CF Worker:
   ```
   wrangler secret put GITHUB_TOKEN
   wrangler secret put HF_SPACE_URL   # https://ghostdrive1-ultron1.hf.space
   ```
   (Parallel AI key is dropped — skip)
7. Verify: `curl https://ultron-brain.ghostdriveg1.workers.dev/health`

---

## Step 1 — HF Space Secrets Inject

Go to `https://huggingface.co/spaces/ghostdrive1/ultron1/settings` → Secrets.

Required secrets (one per key, indexed pattern):

| Secret Name | Value |
|---|---|
| `GROQ_KEY_0` | Groq key 0 |
| `GROQ_KEY_1` | Groq key 1 (if available) |
| `CEREBRAS_KEY_0` | Cerebras key 0 |
| `TOGETHER_KEY_0` | Together key 0 |
| `OPENROUTER_KEY_0` | OpenRouter key 0 |
| `GEMINI_KEY_0` | Gemini key (general pool) |
| `GEMINI_SENTINEL_KEY` | Gemini 2.5 Pro key (Sentinel only) |
| `SAMBANOVA_KEY_0` | SambaNova key 0 |
| `FIREWORKS_KEY_0` | Fireworks key 0 |
| `HF_KEY_0` | HuggingFace Inference key 0 |
| `TAVILY_KEY_0` | Tavily search key 0 |
| `ZILLIZ_URI` | Zilliz cloud URI |
| `ZILLIZ_TOKEN` | Zilliz API token |
| `REDIS_URL` | Upstash Redis → `rediss://default:{TOKEN}@{HOST}:6379` |
| `ULTRON_AUTH_TOKEN` | `de76095c6d7693000b5d6331847b3c6d5cc3b900ed3cdd96e6c2a7d598fe48d3` |
| `DISCORD_TOKEN` | Discord bot token |
| `DISCORD_WEBHOOK_URL` | Discord webhook URL (for Sentinel alerts) |

Optional (deferred):
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anon key |

---

## Step 2 — HF Space Restart & Verify

1. After secrets set → HF Space auto-restarts. If not: Factory Reset.
2. Wait ~90s for cold start (torch + sentence-transformers download).
3. Verify:
   ```bash
   curl -H "X-Ultron-Token: de76095c6d7693000b5d6331847b3c6d5cc3b900ed3cdd96e6c2a7d598fe48d3" \
        https://ghostdrive1-ultron1.hf.space/health
   ```
   Expected response:
   ```json
   {
     "status": "ok",
     "version": "4.0.0",
     "sentinel_active": true,
     "memory_pipeline": true,
     "eternal_loop_active": true
   }
   ```

---

## Step 3 — Discord Bot Verify

1. Send `!ping` in Discord → `ultron#2628` should reply.
2. Send `!ask what is 2+2` → should route through TaskDispatcher → ReActLoop → response.
3. Check HF Space logs for `[Startup] Discord bot thread started`.

---

## Step 4 — Smoke Tests

```bash
BASE=https://ghostdrive1-ultron1.hf.space
TOKEN=de76095c6d7693000b5d6331847b3c6d5cc3b900ed3cdd96e6c2a7d598fe48d3

# Test infer
curl -X POST $BASE/infer \
  -H "X-Ultron-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"what is 2+2","channel_id":"test","user_id":"ghost"}'

# Test plan
curl -X POST $BASE/plan \
  -H "X-Ultron-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"goal":"research latest advances in chemical engineering","channel_id":"test","user_id":"ghost"}'

# Test keys status
curl -H "X-Ultron-Token: $TOKEN" $BASE/keys

# Test metacog state
curl -H "X-Ultron-Token: $TOKEN" $BASE/metacog/state
```

---

## Known Pre-registered Bugs to Watch

| ID | Level | Trigger |
|---|---|---|
| M1 | HIGH | Never run uvicorn with workers>1 — KeyPool breaks |
| M2 | MED | aioredis cross-loop risk — watch startup logs |
| EL1 | HIGH | RAPTOR query timeout in /llm/generate — 25s guard active |
| EL2 | MED | Sentinel fallback if GEMINI_SENTINEL_KEY absent |
| SB1 | HIGH | Supabase sync client → asyncio.to_thread wraps all calls |
| CL7 | HIGH | get_pool() lru_cache frozen — never reinit mid-run |
| CL8 | MED | KeyPoolConfigError if zero keys — check startup log FATAL |
| MG2 | MED | magma_graph >5k nodes → eviction |

---

## Post-Deploy

- Update Notion Master Graph session log with deploy timestamp + `/health` response
- Verify `/rd/history/1356180323058057326` returns empty list (no R&D history yet)
- Phase 8 planning: autonomous research loop generalization, LEANN Phase 6 eval
