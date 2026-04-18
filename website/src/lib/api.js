const API_BASE = import.meta.env.VITE_API_URL || 'https://ultron-api.ghostdriveg1.workers.dev';
const AUTH_TOKEN = import.meta.env.VITE_AUTH_TOKEN || '';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${AUTH_TOKEN}`,
      ...options.headers,
    },
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  health: () => request('/api/health'),

  infer: (payload) =>
    request('/api/infer', { method: 'POST', body: JSON.stringify(payload) }),

  keys: {
    list: () => request('/api/keys'),
    add: (key) => request('/api/keys', { method: 'POST', body: JSON.stringify(key) }),
    remove: (keyId) => request(`/api/keys/${keyId}`, { method: 'DELETE' }),
  },

  sentinel: {
    trigger: (type) =>
      request('/api/sentinel/event', {
        method: 'POST',
        body: JSON.stringify({ type, source: 'website' }),
      }),
    reports: () => request('/api/sentinel/reports'),
  },

  memory: {
    stm: (channelId) => request(`/api/memory/stm/${channelId}`),
  },

  projects: {
    list: () => request('/api/projects'),
  },
};
