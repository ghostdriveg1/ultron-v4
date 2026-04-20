// website/src/lib/api.js
// All Brain calls go through the CF Worker (ultron-api) which proxies to Brain.
// Worker adds X-Ultron-Token from its own secret — website never holds the token.

const BASE = import.meta.env.VITE_API_URL || 'https://ultron-api.ghostdriveg1.workers.dev';

async function req(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    ...opts,
    headers: { 'Content-Type': 'application/json', ...opts.headers },
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  // GET /api/health → Brain /health
  health: () => req('/api/health'),

  // POST /api/infer → Brain /infer
  infer: (payload) => req('/api/infer', { method: 'POST', body: JSON.stringify(payload) }),

  // GET /api/keys → Brain /keys
  keys: {
    list: () => req('/api/keys'),
  },

  // Sentinel
  sentinel: {
    trigger: (type) =>
      req('/api/sentinel/event', {
        method: 'POST',
        body: JSON.stringify({ event_type: type, payload: { source: 'website' } }),
      }),
    reports: () => req('/api/sentinel/reports'),
  },

  // Memory
  memory: {
    stm: (channelId) => req(`/api/memory/stm/${channelId}`),
  },

  // R&D loop history
  rd: {
    history: (userId, limit = 20) => req(`/api/rd/history/${userId}?limit=${limit}`),
  },

  // Infrastructure — SpacePromoter topology + infra event log
  infra: {
    promoterStatus: () => req('/api/health').then((d) => d.promoter ?? null),
    events: () => req('/api/infra/events'),
  },
};
