export function formatUptime(seconds) {
  if (!seconds && seconds !== 0) return '—';
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export function relativeTime(ts) {
  if (!ts) return '—';
  const diff = Date.now() - new Date(ts).getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export function maskKey(key) {
  if (!key || key.length < 8) return '••••••••';
  return key.slice(0, 4) + '•'.repeat(Math.min(16, key.length - 8)) + key.slice(-4);
}

export const PROVIDERS = [
  { id: 'groq', label: 'Groq', weight: 3, color: '#F55036' },
  { id: 'cerebras', label: 'Cerebras', weight: 3, color: '#7C3AED' },
  { id: 'together', label: 'Together', weight: 2, color: '#0EA5E9' },
  { id: 'openrouter', label: 'OpenRouter', weight: 2, color: '#10B981' },
  { id: 'gemini', label: 'Gemini', weight: 2, color: '#4285F4' },
];
