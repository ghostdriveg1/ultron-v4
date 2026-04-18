import { useState } from 'react';
import Card from '../components/Card';

const TABS = ['STM', 'MTM', 'LPM', 'MAGMA', 'Foresight'];

// Mock shard data — replace with real Zilliz stats when endpoint added
const SHARDS = Array.from({ length: 15 }, (_, i) => ({
  id: i,
  fillPct: 0,
  vectors: 0,
}));

export default function Memory() {
  const [tab, setTab] = useState('STM');

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Memory</h1>
        <p className="text-sm text-muted mt-0.5">STM (Redis) → MTM (RAPTOR) → LPM (Zilliz x15) unified engine.</p>
      </div>

      <div className="flex gap-0.5 p-1 bg-bg border border-border rounded-lg w-fit">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-1.5 text-sm rounded transition-colors font-mono ${
              tab === t ? 'bg-surface text-ink shadow-sm font-medium border border-border' : 'text-muted hover:text-ink'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'STM' && (
        <Card title="Short-Term Memory — Redis">
          <div className="px-5 py-5">
            <div className="flex gap-8 mb-4">
              {[['Window', '20 msgs'], ['TTL', '3600s'], ['Scope', 'per channel_id']].map(([k, v]) => (
                <div key={k}>
                  <p className="text-[11px] font-mono text-muted uppercase tracking-wider">{k}</p>
                  <p className="text-sm font-medium text-ink mt-0.5">{v}</p>
                </div>
              ))}
            </div>
            <div className="border border-dashed border-border rounded-lg py-8 text-center">
              <p className="text-xs text-muted font-mono">No active channel. Send a Discord message to populate STM.</p>
            </div>
          </div>
        </Card>
      )}

      {tab === 'MTM' && (
        <Card title="Mid-Term Memory — RAPTOR">
          <div className="px-5 py-5">
            <p className="text-sm text-muted mb-4">Recursive cluster → summarize → collapse. Heat-based promotion from STM. Queries tree before raw chunks.</p>
            <div className="border border-dashed border-border rounded-lg py-8 text-center">
              <p className="text-xs text-muted font-mono">No topic clusters. RAPTOR builds after first session memory write.</p>
            </div>
          </div>
        </Card>
      )}

      {tab === 'LPM' && (
        <Card title="Long-Term Memory — Zilliz x15">
          <div className="px-5 py-5">
            <div className="flex gap-8 mb-5">
              {[['Shards', '15'], ['Embedding', '384-dim'], ['Metric', 'COSINE'], ['Model', 'all-MiniLM-L6-v2']].map(([k, v]) => (
                <div key={k}>
                  <p className="text-[11px] font-mono text-muted uppercase tracking-wider">{k}</p>
                  <p className="text-sm font-medium text-ink mt-0.5">{v}</p>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-5 gap-2">
              {SHARDS.map((shard) => (
                <div key={shard.id} className="border border-border rounded-lg p-3 text-center">
                  <div className="text-[10px] font-mono text-muted mb-2">shard {shard.id}</div>
                  <div className="h-10 bg-bg rounded overflow-hidden relative">
                    <div
                      className="absolute bottom-0 left-0 right-0 bg-status-blue/20 transition-all"
                      style={{ height: `${shard.fillPct}%` }}
                    />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-[10px] font-mono text-muted">{shard.fillPct}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted font-mono mt-3">Shard assignment: hash(user_id % 15). Live fill % when Zilliz stats endpoint added to Brain.</p>
          </div>
        </Card>
      )}

      {tab === 'MAGMA' && (
        <Card title="MAGMA Graph — 4-Graph Layer">
          <div className="px-5 py-10 text-center">
            <p className="text-3xl mb-4 text-border">◈</p>
            <p className="text-sm font-medium text-ink">MAGMA Graph (Cognee)</p>
            <p className="text-xs text-muted font-mono mt-1">magma_graph.py — not yet written</p>
            <div className="mt-6 grid grid-cols-4 gap-3 max-w-sm mx-auto">
              {['SEMANTIC', 'TEMPORAL', 'CAUSAL', 'ENTITY'].map((g) => (
                <div key={g} className="border border-border rounded px-2 py-2 text-center">
                  <p className="text-[10px] font-mono text-muted">{g}</p>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted mt-4 max-w-xs mx-auto">
              D3 force-directed graph wired after magma_graph.py committed. +45.5% reasoning accuracy (MAGMA, arXiv:2601.03236).
            </p>
          </div>
        </Card>
      )}

      {tab === 'Foresight' && (
        <Card title="Foresight — EverMemOS Lifecycle">
          <div className="px-5 py-10 text-center">
            <p className="text-3xl mb-4 text-border">◷</p>
            <p className="text-sm font-medium text-ink">Foresight Engine</p>
            <p className="text-xs text-muted font-mono mt-1">lifecycle.py — not yet written</p>
            <div className="mt-6 flex items-center justify-center gap-2 text-xs font-mono text-muted">
              <span className="border border-border rounded px-2 py-1">MemCell</span>
              <span>→</span>
              <span className="border border-border rounded px-2 py-1">MemScene</span>
              <span>→</span>
              <span className="border border-border rounded px-2 py-1">Foresight</span>
            </div>
            <p className="text-xs text-muted mt-4 max-w-xs mx-auto">
              Predicts what Ghost will need next. Feeds autonomous R&D loop directly (arXiv:2601.02163).
            </p>
          </div>
        </Card>
      )}
    </div>
  );
}
