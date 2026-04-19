// website/src/pages/Projects.jsx
// R&D loop history + improvement timeline
// Data source: GET /rd/history/{user_id} → rd_loop.py

import { useState, useEffect, useCallback } from 'react';
import { api } from '../lib/api';
import { relativeTime } from '../lib/utils';
import Card from '../components/Card';
import Badge from '../components/Badge';

const GHOST_UID = '1356180323058057326';

const STATUS_VARIANT = {
  implemented: 'green',
  proposed: 'yellow',
  rejected: 'red',
  pending: 'gray',
};

const STATUS_ICON = {
  implemented: '✓',
  proposed: '◌',
  rejected: '✗',
  pending: '◷',
};

function ImprovementRow({ item }) {
  const [expanded, setExpanded] = useState(false);
  const status = item.status || 'pending';
  const ts = item.implemented_at || item.proposed_at || item.created_at;

  return (
    <div className="border-b border-border last:border-0">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-start gap-4 py-3.5 text-left hover:bg-bg/50 transition-colors px-5"
      >
        <span
          className={`mt-0.5 text-xs font-mono flex-shrink-0 ${
            status === 'implemented'
              ? 'text-status-green'
              : status === 'rejected'
              ? 'text-status-red'
              : 'text-status-yellow'
          }`}
        >
          {STATUS_ICON[status] || '◌'}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-ink leading-snug">
            {item.title || item.summary || 'Untitled improvement'}
          </p>
          <p className="text-xs text-muted mt-0.5">
            {item.category || 'general'} · {ts ? relativeTime(ts) : '—'}
          </p>
        </div>
        <Badge variant={STATUS_VARIANT[status] || 'gray'}>{status}</Badge>
        <span className="text-xs text-muted ml-2 mt-0.5">{expanded ? '▲' : '▼'}</span>
      </button>
      {expanded && (
        <div className="px-14 pb-4 space-y-2">
          {item.description && (
            <p className="text-xs text-muted leading-relaxed">{item.description}</p>
          )}
          {item.files_changed && item.files_changed.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {item.files_changed.map((f) => (
                <span
                  key={f}
                  className="font-mono text-[10px] bg-bg border border-border px-2 py-0.5 rounded"
                >
                  {f}
                </span>
              ))}
            </div>
          )}
          {item.sentinel_verdict && (
            <p className="text-xs text-muted border-l-2 border-border pl-3 italic">
              Sentinel: {item.sentinel_verdict}
            </p>
          )}
          {item.council_synthesis && (
            <p className="text-xs text-muted border-l-2 border-status-blue pl-3">
              Council: {item.council_synthesis}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function RdStatBar({ label, value, total, color }) {
  const pct = total > 0 ? Math.round((value / total) * 100) : 0;
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-[11px]">
        <span className="text-muted font-mono">{label}</span>
        <span className="font-mono text-ink">
          {value}
          <span className="text-muted">/{total}</span>
        </span>
      </div>
      <div className="h-1 bg-bg rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

export default function Projects() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastFetch, setLastFetch] = useState(null);
  const [filter, setFilter] = useState('all');

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.rd.history(GHOST_UID, 50);
      setData(res);
      setLastFetch(new Date());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const improvements = data?.improvements ?? [];
  const rdState = data?.rd_state ?? null;
  const total = improvements.length;
  const implemented = improvements.filter((i) => i.status === 'implemented').length;
  const proposed = improvements.filter((i) => i.status === 'proposed').length;
  const rejected = improvements.filter((i) => i.status === 'rejected').length;

  const filtered =
    filter === 'all'
      ? improvements
      : improvements.filter((i) => (i.status || 'pending') === filter);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Projects</h1>
          <p className="text-sm text-muted mt-0.5">
            {lastFetch ? `Refreshed ${relativeTime(lastFetch)}` : 'Loading R&D history...'}
          </p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="text-xs font-mono text-muted hover:text-ink border border-border px-3 py-1.5 rounded transition-colors disabled:opacity-40"
        >
          {loading ? '...' : '↻ Refresh'}
        </button>
      </div>

      {error && (
        <div className="border border-red-200 bg-red-50 text-status-red text-xs px-4 py-3 rounded-lg font-mono">
          ✗ {error} — RDLoop requires Redis + Brain running.
        </div>
      )}

      <div className="grid grid-cols-3 gap-4">
        <div className="bg-surface border border-border rounded-lg p-5">
          <p className="text-[11px] text-muted font-mono uppercase tracking-widest mb-3">Progress</p>
          <div className="space-y-3">
            <RdStatBar label="implemented" value={implemented} total={total || 1} color="#16A34A" />
            <RdStatBar label="proposed" value={proposed} total={total || 1} color="#CA8A04" />
            <RdStatBar label="rejected" value={rejected} total={total || 1} color="#DC2626" />
          </div>
        </div>
        <div className="bg-surface border border-border rounded-lg p-5 col-span-2">
          <p className="text-[11px] text-muted font-mono uppercase tracking-widest mb-3">R&D State</p>
          {rdState ? (
            <div className="grid grid-cols-2 gap-x-8 gap-y-2">
              {[
                ['Loop status', rdState.status || 'idle'],
                ['Proposals total', rdState.total_proposals ?? '—'],
                ['Accept rate', rdState.accept_rate ? `${Math.round(rdState.accept_rate * 100)}%` : '—'],
                ['Last run', rdState.last_run_at ? relativeTime(rdState.last_run_at) : 'never'],
                ['Active goal', rdState.active_goal || 'none'],
                ['Foresight signals', rdState.foresight_signal_count ?? '—'],
              ].map(([k, v]) => (
                <div key={k} className="flex gap-3">
                  <span className="text-[11px] font-mono text-muted w-32 flex-shrink-0">{k}</span>
                  <span className="text-[11px] font-mono text-ink truncate">{String(v)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted font-mono">
              {loading ? 'Loading...' : 'No RD state — RDLoop requires Redis.'}
            </p>
          )}
        </div>
      </div>

      <div className="flex gap-2">
        {['all', 'implemented', 'proposed', 'rejected'].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`text-xs font-mono px-3 py-1.5 rounded border transition-colors ${
              filter === f
                ? 'bg-ink text-white border-ink'
                : 'border-border text-muted hover:text-ink hover:border-ink/30'
            }`}
          >
            {f}
            <span className="ml-1.5 opacity-60">
              {f === 'all' ? total : f === 'implemented' ? implemented : f === 'proposed' ? proposed : rejected}
            </span>
          </button>
        ))}
      </div>

      <Card title="Autonomous R&D Log">
        {loading ? (
          <div className="px-5 py-10 text-center">
            <p className="text-xs text-muted font-mono">Loading improvements...</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="px-5 py-10 text-center space-y-2">
            <p className="text-3xl text-border">◷</p>
            <p className="text-sm font-medium text-ink">
              {filter === 'all' ? 'No improvements yet' : `No ${filter} improvements`}
            </p>
            <p className="text-xs text-muted max-w-sm mx-auto leading-relaxed">
              {filter === 'all'
                ? 'Give Ultron a task. After completion the R&D loop proposes autonomous improvements — Sentinel ranks, Council debates, best ones get implemented.'
                : 'Switch to "all" to see everything.'}
            </p>
          </div>
        ) : (
          <div>
            {filtered.map((item, i) => (
              <ImprovementRow key={item.id || i} item={item} />
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
