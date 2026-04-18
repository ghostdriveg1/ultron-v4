# Ultron Mission Control — Website

React + Vite + Tailwind. Primary interface for managing Ultron V4.

## Dev Setup

```bash
cd website
npm install
cp .env.example .env.local
# Edit .env.local with your values
npm run dev
```

## Deploy

### 1. Deploy API Worker

```bash
cd website/cloudflare
npx wrangler deploy
npx wrangler secret put AUTH_TOKEN
```

### 2. Deploy Website (Cloudflare Pages)

Connect `ghostdriveg1/ultron-v4` repo to Cloudflare Pages:
- Root directory: `website`
- Build command: `npm run build`
- Build output directory: `dist`
- Env vars: `VITE_API_URL`, `VITE_AUTH_TOKEN`

## Env Vars

| Var | Description |
|-----|-------------|
| `VITE_API_URL` | CF Worker API URL (e.g. `https://ultron-api.ghostdriveg1.workers.dev`) |
| `VITE_AUTH_TOKEN` | Brain auth token |

## Architecture

```
Browser → CF Pages (static React) → CF Worker (ultron-api) → HF Space (Brain)
```

## Pages

| Page | Route | Data Source |
|------|-------|-------------|
| Dashboard | `/` | `/health` (real-time) |
| Credentials | `/credentials` | Local + future Brain endpoint |
| Memory | `/memory` | STM/MTM/LPM (partial stub) |
| Projects | `/projects` | Supabase (stub) |
| Sentinel | `/sentinel` | Brain `/sentinel/event` |
| Settings | `/settings` | LocalStorage |
