import { useState, useEffect, useCallback } from 'react';
import { api } from '../lib/api';
import { relativeTime } from '../lib/utils';
import Card from '../components/Card';
import Badge from '../components/Badge';

// Channel lookup: tab-name → Redis key prefix used by discord_bot.py
const TABS = ['STM', 'Lifecycle', 'LPM', 'MAGMA', 'Foresight'];

const SHARDS = Array.from({ length: 15 }, (_, i) => ({ id: i, fillPct: 0, vectors: 0 }));

// STM tab: reads live from /memory/stm/:channelId
function STMTab() {
  const [channelId, setChannelId] = useState('');
  const [inputId, setInputId] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchSTM = useCallback(async (cid) => {
    if (!cid) return;
    setLoading(true); setError(null);
    try { const d = await api.memory.stm(cid); setData(d); }
    catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, []);

  return (
    <Card title="Short-Term Memory — Redis">
      <div className="px-5 py-5 space-y-4">
        <div className="flex gap-8">
          {[['Window', '20 msgs'], ['TTL', '7200s'], ['Scope', 'per channel_id']].map(([k, v]) => (
            <div key={k}>
              <p className="text-[11px] font-mono text-muted uppercase tracking-wider">{k}</p>
              <p className="text-sm font-medium text-ink mt-0.5">{v}</p>
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            className="flex-1 text-xs font-mono border border-border rounded px-3 py-2 bg-surface text-ink focus:outline-none focus:ring-1 focus:ring-ink/20"
            placeholder="Discord channel ID"
            value={inputId}
            onChange={(e) => setInputId(e.target.value)}
          />
          <button
            onClick={() => { setChannelId(inputId.trim()); fetchSTM(inputId.trim()); }}
            className="text-xs font-mono px-4 py-2 border border-border rounded hover:bg-bg transition-colors"
          >
            Load
          </button>
        </div>
        {error && <p className="text-xs font-mono text-status-red">✗ {error}</p>}
        {loading && <p className="text-xs font-mono text-muted">Loading...</p>}
        {data && !loading && (
          <div className="space-y-3">
            <div className="flex gap-6">
              <div>
                <p className="text-[11px] font-mono text-muted">CONTEXT WINDOW</p>
                <p className="text-sm font-bold text-ink">{data.context_window_count}</p>
              </div>
              <div>
                <p className="text-[11px] font-mono text-muted">LIFECYCLE CELLS</p>
                <p className="text-sm font-bold text-ink">{data.lifecycle_cell_count}</p>
              </div>
            </div>
            {data.context_window?.length > 0 ? (
              <div className="border border-border rounded-lg overflow-hidden">
                {data.context_window.map((msg, i) => (
                  <div key={i} className="px-4 py-2.5 border-b border-border last:border-0 text-xs font-mono text-ink">
                    {msg}
                  </div>
                ))}
              </div>
            ) : (
              <div className="border border-dashed border-border rounded-lg py-6 text-center">
                <p className="text-xs text-muted font-mono">No context. Send a Discord message to populate.</p>
              </div>
            )}
            {data.lifecycle_cells?.length > 0 && (
              <div>
                <p className="text-[11px] font-mono text-muted uppercase tracking-widest mb-2">Lifecycle Cells</p>
                {data.lifecycle_cells.map((cell, i) => (
                  <div key={i} className="border-b border-border last:border-0 py-2.5">
                    <div className="flex items-center gap-2">
                      <Badge variant="blue" dot>{cell.tier ?? 'stm'}</Badge>
                      <span className="text-[11px] font-mono text-muted">{cell.created_at ? relativeTime(cell.created_at) : ''}</span>
                    </div>
                    <p className="text-xs text-ink mt-1 leading-relaxed">{cell.content ?? ''}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        {!data && !loading && !error && (
          <div className="border border-dashed border-border rounded-lg py-8 text-center">
            <p className="text-xs text-muted font-mono">Enter a Discord channel ID above to view STM.</p>
          </div>
        )}
      </div>
    </Card>
  );
}

export default function Memory() {
  const [tab, setTab] = useState('STM');

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Memory</h1>
        <p className="text-sm text-muted mt-0.5">STM (Redis) → Lifecycle → RAPTOR → Zilliz x15 unified pipeline.</p>
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

      {tab === 'STM' && <STMTab />}

      {tab === 'Lifecycle' && (
        <Card title="Lifecycle Engine — EverMemOS MemCell/MemScene/Foresight">
          <div className="px-5 py-5 space-y-4">
            <div className="flex gap-8">
              {[['Status', 'Active'], ['Heat TTL', '6h'], ['Tiers', 'STM→MTM→LPM']].map(([k, v]) => (
                <div key={k}>
                  <p className="text-[11px] font-mono text-muted uppercase tracking-wider">{k}</p>
                  <p className="text-sm font-medium text-ink mt-0.5">{v}</p>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-2 py-3 px-4 bg-bg rounded-lg">
              {['MemCell', 'heat ↑', 'MemScene', 'abstract', 'Foresight'].map((label, i, arr) => (
                <span key={label} className="flex items-center gap-2">
                  <span className={`text-xs font-mono ${
                    i % 2 === 0 ? 'border border-border rounded px-2.5 py-1 text-ink' : 'text-muted'
                  }`}>{label}</span>
                  {i < arr.length - 1 && i % 2 === 0 && <span className="text-muted text-xs">→</span>}
                </span>
              ))}
            </div>
            <p className="text-xs text-muted leading-relaxed">
              lifecycle.py active. Ingests every Discord message + /infer call. MemCells promoted to MemScene
              after heat threshold (6h TTL). Foresight signal feeds rd_loop.py proposal generation.
            </p>
          </div>
        </Card>
      )}

      {tab === 'LPM' && (
        <Card title="Long-Term Memory — Zilliz x15 + RAPTOR">
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
                    <div className="absolute bottom-0 left-0 right-0 bg-status-blue/20" style={{ height: `${shard.fillPct}%` }} />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-[10px] font-mono text-muted">{shard.fillPct}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted font-mono mt-3">Shard: hash(user_id) % 15. RAPTOR tree summarises clusters before upsert. Live fill % available after Zilliz stats endpoint added.</p>
          </div>
        </Card>
      )}

      {tab === 'MAGMA' && (
        <Card title="MAGMA Graph — 4-Graph Layer">
          <div className="px-5 py-5 space-y-4">
            <div className="flex gap-8">
              {[['Status', 'Active'], ['Backend', 'NetworkX + Redis'], ['Graphs', '4']].map(([k, v]) => (
                <div key={k}>
                  <p className="text-[11px] font-mono text-muted uppercase tracking-wider">{k}</p>
                  <p className="text-sm font-medium text-ink mt-0.5">{v}</p>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-4 gap-3">
              {[
                { name: 'SEMANTIC', desc: 'concept ↔ concept edges' },
                { name: 'TEMPORAL', desc: 'event ordering + time' },
                { name: 'CAUSAL', desc: 'cause → effect chains' },
                { name: 'ENTITY', desc: 'named entities + relations' },
              ].map(({ name, desc }) => (
                <div key={name} className="border border-border rounded-lg p-4 text-center">
                  <p className="text-[11px] font-mono text-ink font-medium">{name}</p>
                  <p className="text-[10px] text-muted mt-1 leading-snug">{desc}</p>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted leading-relaxed">
              magma_graph.py active. NetworkX in-memory + Redis persistence. D3 force-directed visualisation
              available in next phase when graph query endpoint is added.
              +45.5% reasoning accuracy (arXiv:2601.03236).
            </p>
          </div>
        </Card>
      )}

      {tab === 'Foresight' && (
        <Card title="Foresight — Lifecycle Prediction">
          <div className="px-5 py-5 space-y-4">
            <div className="flex gap-8">
              {[['Status', 'Active'], ['Source', 'lifecycle.py'], ['Feeds', 'rd_loop.py']].map(([k, v]) => (
                <div key={k}>
                  <p className="text-[11px] font-mono text-muted uppercase tracking-wider">{k}</p>
                  <p className="text-sm font-medium text-ink mt-0.5">{v}</p>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-2 py-3 px-4 bg-bg rounded-lg">
              {['MemScene heat', '→', 'Foresight signal', '→', 'RDLoop.propose()', '→', 'Council debate', '→', 'Implement'].map((label, i) => (
                <span key={i} className={`text-xs font-mono ${
                  label === '→' ? 'text-muted' : 'border border-border rounded px-2 py-1 text-ink'
                }`}>{label}</span>
              ))}
            </div>
            <p className="text-xs text-muted leading-relaxed">
              Foresight predicts what Ghost will need before asking. Drives autonomous R&D proposals
              visible in Projects tab. Signal strength determined by MemScene heat score over 6h window.
            </p>
          </div>
        </Card>
      )}
    </div>
  );
}
