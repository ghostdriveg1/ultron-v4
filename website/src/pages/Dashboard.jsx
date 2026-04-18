import { useState, useEffect } from 'react';
import { api } from '../lib/api';
import { formatUptime, relativeTime, PROVIDERS } from '../lib/utils';
import Card from '../components/Card';
import Badge from '../components/Badge';

function StatTile({ label, value, sub, loading }) {
  return (
    <div className="bg-surface border border-border rounded-lg p-5">
      <p className="text-[11px] text-muted font-mono uppercase tracking-widest mb-2">{label}</p>
      <p className="text-2xl font-bold tracking-tight text-ink">
        {loading ? <span className="text-border">—</span> : value}
      </p>
      {sub && <p className="text-xs text-muted mt-1.5">{sub}</p>}
    </div>
  );
}

function ProviderRow({ provider, color, weight, status }) {
  const available = status?.available_keys ?? 0;
  const total = status?.total_keys ?? 0;
  const pct = total > 0 ? (available / total) * 100 : 0;
  const healthy = available > 0;
  const tripped = status?.tripped_keys ?? 0;

  return (
    <div className="flex items-center gap-4 py-3 border-b border-border last:border-0">
      <div className="flex items-center gap-2 w-32 flex-shrink-0">
        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
        <span className="text-sm font-medium text-ink capitalize">{provider}</span>
        <span className="text-[10px] font-mono text-muted">w{weight}</span>
      </div>
      <div className="flex-1 h-1 bg-bg rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: healthy ? '#16A34A' : '#DC2626' }}
        />
      </div>
      <div className="w-20 text-right">
        <span className="text-xs font-mono text-muted">{available}/{total}</span>
      </div>
      <Badge variant={healthy ? 'green' : tripped > 0 ? 'yellow' : 'gray'} dot>
        {healthy ? 'healthy' : tripped > 0 ? 'tripped' : 'no keys'}
      </Badge>
    </div>
  );
}

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastFetch, setLastFetch] = useState(null);

  const fetchHealth = async () => {
    try {
      const data = await api.health();
      setHealth(data);
      setLastFetch(new Date());
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const iv = setInterval(fetchHealth, 30_000);
    return () => clearInterval(iv);
  }, []);

  const pool = health?.pool_status ?? {};
  const totalKeys = Object.values(pool).reduce((s, p) => s + (p?.total_keys ?? 0), 0);
  const availKeys = Object.values(pool).reduce((s, p) => s + (p?.available_keys ?? 0), 0);
  const isOnline = health?.status === 'ok';

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-sm text-muted mt-0.5">
            {lastFetch ? `Refreshed ${relativeTime(lastFetch)}` : 'Connecting...'}
          </p>
        </div>
        <button
          onClick={fetchHealth}
          className="text-xs font-mono text-muted hover:text-ink border border-border px-3 py-1.5 rounded transition-colors"
        >
          ↻ Refresh
        </button>
      </div>

      {error && (
        <div className="border border-red-200 bg-red-50 text-status-red text-xs px-4 py-3 rounded-lg font-mono">
          ✗ Brain unreachable — {error}. Check HF Space is running.
        </div>
      )}

      <div className="grid grid-cols-4 gap-4">
        <StatTile
          label="Brain"
          value={isOnline ? 'Online' : 'Offline'}
          sub={isOnline ? `Up ${formatUptime(health?.uptime_seconds)}` : 'Check HF Space'}
          loading={loading}
        />
        <StatTile
          label="Keys Active"
          value={`${availKeys}/${totalKeys}`}
          sub="across all providers"
          loading={loading}
        />
        <StatTile
          label="Providers"
          value={Object.keys(pool).length || '5'}
          sub="in general pool"
          loading={loading}
        />
        <StatTile
          label="Sentinel"
          value={health?.sentinel_status ?? 'Standby'}
          sub={health?.sentinel_last_event ? relativeTime(health.sentinel_last_event) : 'No events'}
          loading={loading}
        />
      </div>

      <Card title="Provider Pool">
        <div className="px-5">
          {loading ? (
            <div className="py-8 text-center text-xs text-muted font-mono">Loading pool status...</div>
          ) : Object.keys(pool).length === 0 ? (
            <div className="py-8 text-center space-y-1">
              <p className="text-sm text-muted">No provider data returned.</p>
              <p className="text-xs font-mono text-muted">Ensure HF Space secrets are set (GROQ_KEY_0, etc).</p>
            </div>
          ) : (
            PROVIDERS.map((p) => (
              <ProviderRow key={p.id} {...p} status={pool[p.id]} />
            ))
          )}
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <Card title="Active Task">
          <div className="px-5 py-5 space-y-2">
            <p className="text-xs text-muted font-mono">No active task.</p>
            <p className="text-xs text-muted">Autonomous R&D loop idle. Give Ultron a task with a deadline.</p>
          </div>
        </Card>

        <Card title="System Health">
          <div className="px-5 py-5 space-y-3">
            {[
              { label: 'Brain (HF Space)', status: isOnline ? 'green' : 'red', text: isOnline ? 'Running' : 'Offline' },
              { label: 'CF Router', status: 'green', text: 'Active' },
              { label: 'Voice Space', status: 'gray', text: 'Not deployed' },
              { label: 'Discord Bot', status: 'gray', text: 'Unknown' },
            ].map(({ label, status, text }) => (
              <div key={label} className="flex items-center justify-between">
                <span className="text-xs text-muted">{label}</span>
                <Badge variant={status} dot>{text}</Badge>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
