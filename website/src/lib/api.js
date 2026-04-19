// website/src/lib/api.js
// All API calls proxied through CF API Worker → Brain
// Worker URL: https://ultron-api.ghostdriveg1.workers.dev

const API_BASE = import.meta.env.VITE_API_URL || 'https://ultron-api.ghostdriveg1.workers.dev';
const AUTH_TOKEN = import.meta.env.VITE_AUTH_TOKEN || '';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-Ultron-Token': AUTH_TOKEN,
      Authorization: `Bearer ${AUTH_TOKEN}`,
      ...options.headers,
    },
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  // ── Brain health (includes pool + lifecycle + promoter status) ────────
  health: () => request('/health'),

  // ── Infer (direct to brain) ───────────────────────────────────────────
  infer: (payload) =>
    request('/infer', { method: 'POST', body: JSON.stringify(payload) }),

  // ── Key pool ──────────────────────────────────────────────────────────
  keys: {
    status: () => request('/keys'),
  },

  // ── Sentinel ──────────────────────────────────────────────────────────
  sentinel: {
    trigger: (eventType, payload = {}) =>
      request('/sentinel/event', {
        method: 'POST',
        body: JSON.stringify({
          event_type: eventType,
          payload: { source: 'website', ...payload },
        }),
      }),
  },

  // ── Memory ────────────────────────────────────────────────────────────
  memory: {
    stm: (channelId) => request(`/memory/stm/${channelId}`),
  },

  // ── R&D loop history ─────────────────────────────────────────────────
  rd: {
    history: (userId, limit = 30) =>
      request(`/rd/history/${userId}?limit=${limit}`),
  },
};
