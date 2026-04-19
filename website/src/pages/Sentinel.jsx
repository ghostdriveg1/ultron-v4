// website/src/pages/Sentinel.jsx
// God Layer dashboard: SpacePromoter live topology + audit trigger
// Data: GET /health (promoter field, refreshed every 30s)

import { useState, useEffect, useCallback } from 'react';
import { api } from '../lib/api';
import { relativeTime } from '../lib/utils';
import Card from '../components/Card';
import Badge from '../components/Badge';

function NodeCard({ name, role, url, isAlive, lastStatus, consecutiveFailures, consecutiveSuccesses, lastCheckAt }) {
  const ok = lastStatus === 'ok';
  const unknown = lastStatus === 'unknown';
  return (
    <div className="bg-surface border border-border rounded-lg p-5">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full flex-shrink-0 ${
              unknown ? 'bg-border' : ok ? 'bg-status-green' : 'bg-status-red'
            }`}
          />
          <span className="text-xs font-mono font-medium text-ink">{name}</span>
        </div>
        <Badge variant={role === 'primary' ? 'green' : role === 'backup' ? 'yellow' : 'gray'}>
          {role}
        </Badge>
      </div>
      <p className="text-[11px] font-mono text-muted truncate mb-3" title={url}>
        {url || '—'}
      </p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
        <div>
          <p className="text-[10px] text-muted font-mono">STATUS</p>
          <p className={`text-xs font-mono font-medium ${
            ok ? 'text-status-green' : unknown ? 'text-muted' : 'text-status-red'
          }`}>
            {lastStatus}
          </p>
        </div>
        <div>
          <p className="text-[10px] text-muted font-mono">ALIVE</p>
          <p className={`text-xs font-mono font-medium ${isAlive ? 'text-status-green' : 'text-status-red'}`}>
            {isAlive ? 'yes' : 'no'}
          </p>
        </div>
        <div>
          <p className="text-[10px] text-muted font-mono">FAILS</p>
          <p className={`text-xs font-mono font-medium ${consecutiveFailures > 0 ? 'text-status-red' : 'text-ink'}`}>
            {consecutiveFailures}
          </p>
        </div>
        <div>
          <p className="text-[10px] text-muted font-mono">OK STREAK</p>
          <p className="text-xs font-mono font-medium text-ink">{consecutiveSuccesses}</p>
        </div>
      </div>
      {lastCheckAt && (
        <p className="text-[10px] text-muted font-mono mt-3">Checked {relativeTime(lastCheckAt)}</p>
      )}
    </div>
  );
}

export default function Sentinel() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastFetch, setLastFetch] = useState(null);
  const [triggering, setTriggering] = useState(false);
  const [triggered, setTriggered] = useState(false);
  const [triggerError, setTriggerError] = useState(null);

  const fetchHealth = useCallback(async () => {
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
  }, []);

  useEffect(() => {
    fetchHealth();
    const iv = setInterval(fetchHealth, 30_000);
    return () => clearInterval(iv);
  }, [fetchHealth]);

  const triggerAudit = async () => {
    setTriggering(true);
    setTriggerError(null);
    try {
      await api.sentinel.trigger('weekly_audit');
      setTriggered(true);
      setTimeout(() => setTriggered(false), 4000);
    } catch (e) {
      setTriggerError(e.message);
    } finally {
      setTriggering(false);
    }
  };

  const promoter = health?.promoter ?? null;
  const promoterActive = health?.promoter_active ?? false;
  const sentinelActive = health?.sentinel_active ?? false;
  const nodes = promoter?.nodes ? Object.values(promoter.nodes) : [];
  const failoverCount = promoter?.failover_count ?? 0;
  const inDegraded = promoter?.in_degraded_state ?? false;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Sentinel</h1>
          <p className="text-sm text-muted mt-0.5">
            {lastFetch ? `Refreshed ${relativeTime(lastFetch)}` : 'Connecting...'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchHealth}
            disabled={loading}
            className="text-xs font-mono text-muted hover:text-ink border border-border px-3 py-1.5 rounded transition-colors disabled:opacity-40"
          >
            {loading ? '...' : '↻'}
          </button>
          <button
            onClick={triggerAudit}
            disabled={triggering}
            className="text-sm font-mono px-4 py-2 border border-border rounded hover:bg-bg transition-colors disabled:opacity-40"
          >
            {triggering ? '...' : triggered ? '✓ Triggered' : 'Trigger Audit'}
          </button>
        </div>
      </div>

      {error && (
        <div className="text-xs font-mono text-status-red border border-red-200 bg-red-50 px-4 py-3 rounded-lg">
          ✗ {error}
        </div>
      )}
      {triggerError && (
        <div className="text-xs font-mono text-status-red border border-red-200 bg-red-50 px-4 py-3 rounded-lg">
          ✗ Trigger failed: {triggerError}
        </div>
      )}

      <div className="grid grid-cols-3 gap-4">
        {[
          {
            label: 'God Layer',
            value: sentinelActive ? 'Active' : 'Inactive',
            sub: sentinelActive ? 'Gemini 2.5 Pro · dedicated key' : 'Set GEMINI_SENTINEL_KEY in HF Space',
            ok: sentinelActive,
          },
          {
            label: 'SpacePromoter',
            value: promoterActive ? (inDegraded ? 'Degraded' : `${nodes.length} node${nodes.length !== 1 ? 's' : ''}`) : 'Inactive',
            sub: promoterActive
              ? `${failoverCount} failover${failoverCount !== 1 ? 's' : ''} · 30s check interval`
              : 'No Redis or init failed',
            ok: promoterActive && !inDegraded,
          },
          {
            label: 'Weekly Audit',
            value: 'Sunday 23:59',
            sub: 'GitHub Actions cron → /sentinel/event',
            ok: true,
          },
        ].map(({ label, value, sub, ok }) => (
          <div key={label} className="bg-surface border border-border rounded-lg p-5">
            <p className="text-[11px] text-muted font-mono uppercase tracking-widest mb-2">{label}</p>
            <div className="flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${ok ? 'bg-status-green' : 'bg-status-red'}`} />
              <p className="text-base font-bold text-ink">{value}</p>
            </div>
            <p className="text-xs text-muted mt-1">{sub}</p>
          </div>
        ))}
      </div>

      {promoterActive && nodes.length > 0 && (
        <div>
          <h2 className="text-xs font-mono text-muted uppercase tracking-widest mb-3">Live Space Topology</h2>
          <div className={`grid gap-4 ${nodes.length === 1 ? 'grid-cols-1' : nodes.length === 2 ? 'grid-cols-2' : 'grid-cols-3'}`}>
            {nodes.map((node) => (
              <NodeCard key={node.name} {...node} />
            ))}
          </div>
          {inDegraded && (
            <div className="mt-3 text-xs font-mono text-status-red border border-red-200 bg-red-50 px-4 py-3 rounded-lg">
              🚨 DEGRADED — max failovers reached or all nodes down. Manual intervention required.
            </div>
          )}
        </div>
      )}

      {promoterActive && nodes.length === 0 && (
        <Card title="Space Topology">
          <div className="px-5 py-6 text-center">
            <p className="text-xs text-muted font-mono">No nodes. Set BRAIN_PRIMARY_URL env var.</p>
          </div>
        </Card>
      )}

      <Card title="Trigger Schedule">
        <div className="px-5 py-5 space-y-0">
          {[
            { when: 'Every 30s', what: 'SpacePromoter pings all Space /health endpoints. 3 consecutive failures → backup promoted → CF KV updated.' },
            { when: 'Every failure', what: 'Sentinel instant analysis → Notion incident page → KV rerouting → Discord DM.' },
            { when: 'Sunday 23:59', what: '1M context window reads full week logs → structured report → Notion + Discord. Triggered by GitHub Actions.' },
            { when: 'Project start', what: 'Reads brief → writes operational plan: DevOps, memory arch, MOA config, tool assignments.' },
          ].map(({ when, what }) => (
            <div key={when} className="flex gap-5 py-3.5 border-b border-border last:border-0">
              <div className="w-36 text-xs font-mono text-ink font-medium flex-shrink-0 pt-0.5">{when}</div>
              <div className="text-xs text-muted leading-relaxed">{what}</div>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <Card title="Incident Log">
          <div className="px-5 py-6 text-center">
            <p className="text-xs text-muted font-mono">No incidents. System stable.</p>
            <p className="text-xs text-muted mt-1">SpacePromoter events appear here in real time.</p>
          </div>
        </Card>
        <Card title="Weekly Reports">
          <div className="px-5 py-6 text-center">
            <p className="text-xs text-muted font-mono">No reports yet.</p>
            <p className="text-xs text-muted mt-1">First report: this Sunday 23:59 UTC.</p>
          </div>
        </Card>
      </div>
    </div>
  );
}
