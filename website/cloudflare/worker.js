/**
 * website/cloudflare/worker.js — ultron-api
 * Proxies /api/* → Brain (active Space URL from KV).
 * Injects X-Ultron-Token from Worker secret.
 *
 * Env (set in CF dashboard):
 *   ULTRON_AUTH_TOKEN  — forwarded as X-Ultron-Token
 *   KV_ROUTING         — binding to KV namespace (ultron:routing:v4)
 */

const FALLBACK_BRAIN = 'https://ghostdrive1-ultron1.hf.space';
const KV_ROUTING_KEY = 'ultron:routing:v4';

async function getBrainUrl(env) {
  try {
    if (env.KV_ROUTING) {
      const raw = await env.KV_ROUTING.get(KV_ROUTING_KEY);
      if (raw) { const r = JSON.parse(raw); if (r?.primary) return r.primary.replace(/\/$/, ''); }
    }
  } catch {}
  return FALLBACK_BRAIN;
}

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

async function proxy(request, brainPath, brainUrl, env) {
  const init = {
    method: request.method,
    headers: { 'Content-Type': 'application/json', 'X-Ultron-Token': env.ULTRON_AUTH_TOKEN ?? '' },
  };
  if (request.method === 'POST') init.body = await request.text();
  const upstream = await fetch(`${brainUrl}${brainPath}`, init);
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { 'Content-Type': 'application/json', ...CORS },
  });
}

export default {
  async fetch(request, env) {
    const { pathname: path, search } = new URL(request.url);

    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: CORS });

    const brain = await getBrainUrl(env);

    if (path === '/api/health')         return proxy(request, '/health', brain, env);
    if (path === '/api/infer')          return proxy(request, '/infer', brain, env);
    if (path === '/api/keys')           return proxy(request, '/keys', brain, env);
    if (path === '/api/sentinel/event') return proxy(request, '/sentinel/event', brain, env);

    const stm = path.match(/^\/api\/memory\/stm\/(.+)$/);
    if (stm) return proxy(request, `/memory/stm/${stm[1]}`, brain, env);

    const rd = path.match(/^\/api\/rd\/history\/(.+)$/);
    if (rd) return proxy(request, `/rd/history/${rd[1]}${search}`, brain, env);

    // /api/infra/events — proxy to Brain; 404 → empty array (endpoint added in future phase)
    if (path === '/api/infra/events') {
      try {
        const r = await proxy(request, '/infra/events', brain, env);
        if (r.status === 404) return new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json', ...CORS } });
        return r;
      } catch {
        return new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json', ...CORS } });
      }
    }

    // /api/sentinel/reports — stub until Sentinel writes reports to retrievable store
    if (path === '/api/sentinel/reports') return new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json', ...CORS } });

    return new Response(JSON.stringify({ error: 'Not found' }), { status: 404, headers: { 'Content-Type': 'application/json', ...CORS } });
  },
};
