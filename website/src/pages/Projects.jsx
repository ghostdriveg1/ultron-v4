// website/src/pages/Projects.jsx
// R&D history from rd_loop.py + project timeline
// API: GET /api/rd/history/:userId?limit=20
import { useState, useEffect, useCallback } from 'react';
import { api } from '../lib/api';
import { relativeTime } from '../lib/utils';
import Card from '../components/Card';
import Badge from '../components/Badge';

const GHOST_USER_ID = '1356180323058057326';

const STATE_COLOR = {
  proposed: 'yellow',
  ranked: 'blue',
  debating: 'blue',
  implementing: 'green',
  done: 'green',
  failed: 'red',
};

function RDEntry({ item }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-b border-border last:border-0">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-start gap-4 py-4 text-left hover:bg-bg/60 transition-colors px-5"
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Badge variant={STATE_COLOR[item.state] ?? 'gray'} dot>
              {item.state ?? 'unknown'}
            </Badge>
            <span className="text-[11px] font-mono text-muted">
              {relativeTime(item.proposed_at)}
            </span>
          </div>
          <p className="text-sm font-medium text-ink leading-snug truncate">
            {item.title ?? 'Untitled improvement'}
          </p>
          <p className="text-xs text-muted mt-0.5 leading-relaxed line-clamp-2">
            {item.rationale ?? '—'}
          </p>
        </div>
        <span className="text-xs font-mono text-muted flex-shrink-0 pt-0.5">
          {open ? '▲' : '▼'}
        </span>
      </button>
      {open && (
        <div className="px-5 pb-4 space-y-3">
          {item.council_verdict && (
            <div>
              <p className="text-[11px] font-mono text-muted uppercase tracking-widest mb-1">Council Verdict</p>
              <p className="text-xs text-ink leading-relaxed">{item.council_verdict}</p>
            </div>
          )}
          {item.implementation_notes && (
            <div>
              <p className="text-[11px] font-mono text-muted uppercase tracking-widest mb-1">Implementation</p>
              <p className="text-xs text-ink leading-relaxed">{item.implementation_notes}</p>
            </div>
          )}
          {item.completed_at && (
            <p className="text-[11px] font-mono text-muted">
              Completed {relativeTime(item.completed_at)}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function RDState({ state }) {
  if (!state) return null;
  return (
    <div className="grid grid-cols-3 gap-4">
      {[
        { label: 'Current Phase', value: state.phase ?? '—' },
        { label: 'Proposals Seen', value: state.proposals_count ?? 0 },
        { label: 'Implemented', value: state.implemented_count ?? 0 },
      ].map(({ label, value }) => (
        <div key={label} className="bg-surface border border-border rounded-lg p-4">
          <p className="text-[11px] font-mono text-muted uppercase tracking-widest mb-1.5">{label}</p>
          <p className="text-xl font-bold text-ink">{value}</p>
        </div>
      ))}
    </div>
  );
}

export default function Projects() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [userId, setUserId] = useState(GHOST_USER_ID);
  const [inputId, setInputId] = useState(GHOST_USER_ID);

  const fetchHistory = useCallback(async (uid) => {
    setLoading(true);
    setError(null);
    try {
      const d = await api.rd.history(uid);
      setData(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchHistory(userId); }, [userId, fetchHistory]);

  const improvements = data?.improvements ?? [];
  const rdState = data?.rd_state ?? null;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Projects</h1>
          <p className="text-sm text-muted mt-0.5">Tasks, deadlines, and autonomous R&D sessions logged here.</p>
        </div>
        <button
          onClick={() => fetchHistory(userId)}
          className="text-xs font-mono text-muted hover:text-ink border border-border px-3 py-1.5 rounded transition-colors"
        >
          ↻ Refresh
        </button>
      </div>

      <div className="flex gap-2 items-center">
        <input
          className="flex-1 text-xs font-mono border border-border rounded px-3 py-2 bg-surface text-ink focus:outline-none focus:ring-1 focus:ring-ink/20"
          placeholder="Discord user ID"
          value={inputId}
          onChange={(e) => setInputId(e.target.value)}
        />
        <button
          onClick={() => setUserId(inputId.trim())}
          className="text-xs font-mono px-4 py-2 border border-border rounded hover:bg-bg transition-colors"
        >
          Load
        </button>
      </div>

      {error && (
        <div className="border border-red-200 bg-red-50 text-status-red text-xs px-4 py-3 rounded-lg font-mono">
          ✗ {error} — Brain may be offline or Redis unavailable.
        </div>
      )}

      {!loading && rdState && <RDState state={rdState} />}

      <Card title="Autonomous R&D Log">
        {loading ? (
          <div className="px-5 py-10 text-center text-xs font-mono text-muted">Loading...</div>
        ) : improvements.length === 0 ? (
          <div className="px-5 py-10 text-center space-y-1.5">
            <p className="text-sm text-muted">No R&D sessions yet.</p>
            <p className="text-xs font-mono text-muted">
              Post-deadline improvement sessions appear here. Powered by rd_loop.py.
            </p>
          </div>
        ) : (
          improvements.map((item, i) => <RDEntry key={item.id ?? i} item={item} />)
        )}
      </Card>

      <Card title="Project Timeline">
        <div className="px-5 py-5 space-y-0">
          {[
            { phase: 'Phase 1–3', what: 'LLM router, key pool, ReAct loop, task dispatcher', done: true },
            { phase: 'Phase 4', what: 'Discord bot, per-channel Redis context window', done: true },
            { phase: 'Phase 5', what: 'Memory: RAPTOR, Zilliz, lifecycle, MAGMA graph, R&D loop', done: true },
            { phase: 'Phase 5b', what: 'Sentinel, Council MoA, voice STT/TTS, SpacePromoter', done: true },
            { phase: 'Phase 6', what: 'Multi-space active-active, key expansion, credential injection', done: false },
            { phase: 'Phase 7', what: 'Production deploy: HF Spaces + CF Worker + Discord live', done: false },
          ].map(({ phase, what, done }) => (
            <div key={phase} className="flex gap-4 py-3 border-b border-border last:border-0 items-start">
              <span className="w-20 text-[11px] font-mono text-muted flex-shrink-0 pt-0.5">{phase}</span>
              <span className="flex-1 text-xs text-ink leading-relaxed">{what}</span>
              <Badge variant={done ? 'green' : 'gray'} dot>{done ? 'done' : 'pending'}</Badge>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
