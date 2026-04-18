import { useState } from 'react';
import { api } from '../lib/api';
import Card from '../components/Card';
import Badge from '../components/Badge';

export default function Sentinel() {
  const [triggering, setTriggering] = useState(false);
  const [triggered, setTriggered] = useState(false);
  const [triggerError, setTriggerError] = useState(null);

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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Sentinel</h1>
          <p className="text-sm text-muted mt-0.5">God Layer. Gemini 2.5 Pro. Dedicated key. Watches everything.</p>
        </div>
        <button
          onClick={triggerAudit}
          disabled={triggering}
          className="text-sm font-mono px-4 py-2 border border-border rounded hover:bg-bg transition-colors disabled:opacity-40"
        >
          {triggering ? '...' : triggered ? '✓ Triggered' : 'Trigger Audit'}
        </button>
      </div>

      {triggerError && (
        <div className="text-xs font-mono text-status-red border border-red-200 bg-red-50 px-4 py-3 rounded-lg">
          ✗ {triggerError}
        </div>
      )}

      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Authority', value: 'God Layer', sub: 'Above all modules. Final word.' },
          { label: 'Model', value: 'Gemini 2.5 Pro', sub: 'Dedicated. Never shared with pool.' },
          { label: 'Schedule', value: 'Sunday 23:59', sub: 'Weekly audit + incident detection' },
        ].map(({ label, value, sub }) => (
          <div key={label} className="bg-surface border border-border rounded-lg p-5">
            <p className="text-[11px] text-muted font-mono uppercase tracking-widest mb-2">{label}</p>
            <p className="text-base font-bold text-ink">{value}</p>
            <p className="text-xs text-muted mt-1">{sub}</p>
          </div>
        ))}
      </div>

      <Card title="Trigger Schedule">
        <div className="px-5 py-5 space-y-0">
          {[
            { when: 'Every request', what: 'Reads KV routing table. Validates Space health. <50ms.' },
            { when: 'Every failure', what: 'Instant incident analysis → Notion page → KV rerouting → Discord DM to Ghost.' },
            { when: 'Every Sunday 23:59', what: '1M context window reads full week logs → structured report → Notion + Discord.' },
            { when: 'Project start', what: 'Reads brief → writes operational plan: DevOps, memory arch, MOA config, tool assignments.' },
          ].map(({ when, what }) => (
            <div key={when} className="flex gap-5 py-3 border-b border-border last:border-0">
              <div className="w-40 text-xs font-mono text-ink font-medium flex-shrink-0 pt-0.5">{when}</div>
              <div className="text-xs text-muted leading-relaxed">{what}</div>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <Card title="Incident Log">
          <div className="px-5 py-6 text-center">
            <p className="text-xs text-muted font-mono">No incidents. System stable.</p>
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
