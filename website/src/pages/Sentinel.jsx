// website/src/pages/Sentinel.jsx
// Live: SpacePromoter topology (from /health), infra event log, trigger buttons
import { useState, useEffect, useCallback } from 'react';
import { api } from '../lib/api';
import { relativeTime } from '../lib/utils';
import Card from '../components/Card';
import Badge from '../components/Badge';

const EVENT_COLORS = {
  promotion: 'yellow',
  recovery: 'green',
  health_degraded: 'red',
  kv_write_failed: 'red',
};

function NodeCard({ name, node }) {
  const alive = node?.is_alive;
  const status = node?.last_status;
  return (
    <div className="bg-surface border border-border rounded-lg p-5 space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-[11px] font-mono text-muted uppercase tracking-widest">{node?.role ?? name}</p>
        <Badge variant={alive ? (status === 'ok' ? 'green' : 'yellow') : 'red'} dot>
          {alive ? (status === 'ok' ? 'healthy' : 'degraded') : 'down'}
        </Badge>
      </div>
      <p className="text-sm font-medium text-ink break-all">{node?.name ?? name}</p>
      <p className="text-[11px] font-mono text-muted break-all">{node?.url ?? '—'}</p>
      <div className="flex gap-4 pt-1">
        <div className="text-center">
          <p className="text-[10px] font-mono text-muted">failures</p>
          <p className="text-sm font-bold text-ink">{node?.consecutive_failures ?? 0}</p>
        </div>
        <div className="text-center">
          <p className="text-[10px] font-mono text-muted">successes</p>
          <p className="text-sm font-bold text-ink">{node?.consecutive_successes ?? 0}</p>
        </div>
        <div className="flex-1 text-right">
          <p className="text-[10px] font-mono text-muted">last check</p>
          <p className="text-xs font-mono text-muted">
            {node?.last_check_at ? relativeTime(node.last_check_at) : '—'}
          </p>
        </div>
      </div>
    </div>
  );
}

function InfraEvent({ ev }) {
  let parsed;
  try { parsed = typeof ev === 'string' ? JSON.parse(ev) : ev; } catch { parsed = { event_type: 'unknown', message: String(ev) }; }
  return (
    <div className="flex gap-4 py-3 border-b border-border last:border-0">
      <Badge variant={EVENT_COLORS[parsed.event_type] ?? 'gray'} dot>
        {parsed.event_type ?? '?'}
      </Badge>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-ink leading-snug">{parsed.message ?? '—'}</p>
        {parsed.timestamp && (
          <p className="text-[11px] font-mono text-muted mt-0.5">{relativeTime(parsed.timestamp)}</p>
        )}
      </div>
    </div>
  );
}

export default function Sentinel() {
  const [health, setHealth] = useState(null);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [evLoading, setEvLoading] = useState(true);
  const [triggering, setTriggering] = useState(false);
  const [triggered, setTriggered] = useState(null);
  const [triggerError, setTriggerError] = useState(null);

  const fetchHealth = useCallback(async () => {
    try { const d = await api.health(); setHealth(d); } catch { /* offline */ }
    finally { setLoading(false); }
  }, []);

  const fetchEvents = useCallback(async () => {
    setEvLoading(true);
    try {
      const d = await api.infra.events();
      setEvents(Array.isArray(d) ? d : d?.events ?? []);
    } catch { setEvents([]); }
    finally { setEvLoading(false); }
  }, []);

  useEffect(() => {
    fetchHealth();
    fetchEvents();
    const iv = setInterval(fetchHealth, 30_000);
    return () => clearInterval(iv);
  }, [fetchHealth, fetchEvents]);

  const triggerEvent = async (type) => {
    setTriggering(true);
    setTriggerError(null);
    try {
      await api.sentinel.trigger(type);
      setTriggered(type);
      setTimeout(() => setTriggered(null), 4000);
    } catch (e) {
      setTriggerError(e.message);
    } finally {
      setTriggering(false);
    }
  };

  const promoter = health?.promoter ?? null;
  const nodes = promoter?.nodes ?? {};
  const sentinelActive = health?.sentinel_active;
  const promoterActive = health?.promoter_active;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Sentinel</h1>
          <p className="text-sm text-muted mt-0.5">God Layer. Gemini 2.5 Pro. Dedicated key. Watches everything.</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => { fetchHealth(); fetchEvents(); }}
            className="text-xs font-mono text-muted hover:text-ink border border-border px-3 py-1.5 rounded transition-colors"
          >
            ↻ Refresh
          </button>
          <button
            onClick={() => triggerEvent('weekly_audit')}
            disabled={triggering}
            className="text-sm font-mono px-4 py-2 border border-border rounded hover:bg-bg transition-colors disabled:opacity-40"
          >
            {triggering ? '...' : triggered === 'weekly_audit' ? '✓ Triggered' : 'Trigger Audit'}
          </button>
        </div>
      </div>

      {triggerError && (
        <div className="text-xs font-mono text-status-red border border-red-200 bg-red-50 px-4 py-3 rounded-lg">
          ✗ {triggerError}
        </div>
      )}

      <div className="grid grid-cols-3 gap-4">
        {[
          {
            label: 'Sentinel',
            value: loading ? '—' : (sentinelActive ? 'Active' : 'Inactive'),
            sub: 'Gemini 2.5 Pro. Dedicated key.',
            variant: loading ? 'gray' : (sentinelActive ? 'green' : 'red'),
          },
          {
            label: 'SpacePromoter',
            value: loading ? '—' : (promoterActive ? 'Running' : 'Off'),
            sub: promoter
              ? `${promoter.check_interval_seconds}s interval · failovers: ${promoter.failover_count ?? 0}`
              : 'Health loop + CF KV promotion',
            variant: loading ? 'gray' : (promoterActive ? 'green' : 'gray'),
          },
          {
            label: 'Degraded?',
            value: promoter?.in_degraded_state ? 'YES' : 'No',
            sub: `Max failovers: ${promoter?.max_failover_attempts ?? 5}`,
            variant: promoter?.in_degraded_state ? 'red' : 'green',
          },
        ].map(({ label, value, sub, variant }) => (
          <div key={label} className="bg-surface border border-border rounded-lg p-5">
            <div className="flex items-center justify-between mb-2">
              <p className="text-[11px] text-muted font-mono uppercase tracking-widest">{label}</p>
              <Badge variant={variant} dot>{value}</Badge>
            </div>
            <p className="text-xs text-muted leading-relaxed">{sub}</p>
          </div>
        ))}
      </div>

      <div>
        <h2 className="text-xs font-mono text-muted uppercase tracking-widest mb-3">Space Topology</h2>
        {loading ? (
          <div className="text-xs font-mono text-muted py-6 text-center">Loading topology...</div>
        ) : Object.keys(nodes).length === 0 ? (
          <div className="bg-surface border border-border rounded-lg px-5 py-8 text-center">
            <p className="text-sm text-muted">No topology data.</p>
            <p className="text-xs font-mono text-muted mt-1">SpacePromoter inactive or Brain offline.</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4">
            {Object.entries(nodes).map(([name, node]) => (
              <NodeCard key={name} name={name} node={node} />
            ))}
          </div>
        )}
      </div>

      <Card title="Trigger Schedule">
        <div className="px-5 py-2 space-y-0">
          {[
            { when: 'Every request', what: 'Reads KV routing table. Validates Space health. <50ms.' },
            { when: 'Every failure', what: 'Instant incident analysis → Notion page → KV rerouting → Discord DM.' },
            { when: 'Sunday 23:59', what: '1M context window reads full week logs → structured report → Notion + Discord.' },
            { when: 'Project start', what: 'Reads brief → writes operational plan: DevOps, memory arch, MOA config, tool assignments.' },
          ].map(({ when, what }) => (
            <div key={when} className="flex gap-5 py-3 border-b border-border last:border-0">
              <div className="w-36 text-xs font-mono text-ink font-medium flex-shrink-0 pt-0.5">{when}</div>
              <div className="text-xs text-muted leading-relaxed">{what}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Manual Actions">
        <div className="px-5 py-4 flex flex-wrap gap-3">
          {[
            { label: 'Trigger Audit', type: 'weekly_audit' },
            { label: 'Health Check', type: 'health_check' },
            { label: 'Force KV Sync', type: 'routing_override' },
          ].map(({ label, type }) => (
            <button
              key={type}
              onClick={() => triggerEvent(type)}
              disabled={triggering}
              className="text-xs font-mono px-4 py-2 border border-border rounded hover:bg-bg transition-colors disabled:opacity-40"
            >
              {triggering ? '...' : triggered === type ? `✓ ${label}` : label}
            </button>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <Card title="Infrastructure Events">
          {evLoading ? (
            <div className="px-5 py-6 text-xs font-mono text-muted text-center">Loading...</div>
          ) : events.length === 0 ? (
            <div className="px-5 py-6 text-center">
              <p className="text-xs font-mono text-muted">No events. System stable.</p>
            </div>
          ) : (
            <div className="px-5 py-2">
              {events.slice(-20).reverse().map((ev, i) => <InfraEvent key={i} ev={ev} />)}
            </div>
          )}
        </Card>

        <Card title="Weekly Reports">
          <div className="px-5 py-6 text-center space-y-3">
            <p className="text-xs font-mono text-muted">No reports yet.</p>
            <p className="text-xs text-muted">First report: this Sunday 23:59 UTC.</p>
            <button
              onClick={() => triggerEvent('weekly_audit')}
              disabled={triggering}
              className="text-xs font-mono px-4 py-2 border border-border rounded hover:bg-bg transition-colors disabled:opacity-40"
            >
              Force now
            </button>
          </div>
        </Card>
      </div>
    </div>
  );
}
