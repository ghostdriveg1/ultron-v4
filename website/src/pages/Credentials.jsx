import { useState, useRef } from 'react';
import { PROVIDERS, maskKey } from '../lib/utils';
import Card from '../components/Card';
import Badge from '../components/Badge';

const EMPTY_FORM = { provider: 'groq', key: '', pool_type: 'general' };

function parseEnvFile(text) {
  const keys = [];
  const providerPrefixes = { GROQ: 'groq', CEREBRAS: 'cerebras', TOGETHER: 'together', OPENROUTER: 'openrouter', GEMINI: 'gemini' };

  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    // Match PROVIDER_API_KEY_N=value or PROVIDER_KEY_N=value or GEMINI_SENTINEL_KEY=value
    const match = trimmed.match(/^([A-Z_]+)\s*=\s*(.+)$/);
    if (!match) continue;

    const [, envKey, rawVal] = match;
    const value = rawVal.replace(/^["']|["']$/g, '').trim();
    if (!value) continue;

    if (envKey === 'GEMINI_SENTINEL_KEY') {
      keys.push({ provider: 'gemini', key: value, env_key: envKey, pool_type: 'sentinel' });
      continue;
    }

    // Pattern: GROQ_API_KEY_0, GROQ_KEY_0, CEREBRAS_API_KEY_1, etc.
    for (const [prefix, provider] of Object.entries(providerPrefixes)) {
      if (envKey.startsWith(prefix) && /_\d+$/.test(envKey)) {
        keys.push({ provider, key: value, env_key: envKey, pool_type: 'general' });
        break;
      }
    }
  }
  return keys;
}

export default function Credentials() {
  const [form, setForm] = useState(EMPTY_FORM);
  const [localKeys, setLocalKeys] = useState([]);
  const [importPreview, setImportPreview] = useState(null);
  const [showKey, setShowKey] = useState({});
  const [copied, setCopied] = useState(null);
  const fileRef = useRef();

  const handleAdd = () => {
    if (!form.key.trim()) return;
    setLocalKeys((prev) => [
      ...prev,
      { ...form, id: `${form.provider}_${Date.now()}`, addedAt: new Date().toISOString() },
    ]);
    setForm(EMPTY_FORM);
  };

  const handleRemove = (id) => setLocalKeys((prev) => prev.filter((k) => k.id !== id));

  const readFile = (file) => {
    const reader = new FileReader();
    reader.onload = (ev) => setImportPreview(parseEnvFile(ev.target.result));
    reader.readAsText(file);
  };

  const confirmImport = () => {
    setLocalKeys((prev) => [
      ...prev,
      ...importPreview.map((k) => ({ ...k, id: `${k.provider}_${Date.now()}_${Math.random().toString(36).slice(2)}` })),
    ]);
    setImportPreview(null);
    if (fileRef.current) fileRef.current.value = '';
  };

  const copyEnvSnippet = () => {
    const lines = localKeys
      .filter((k) => k.pool_type === 'general')
      .reduce((acc, k) => {
        const providerKeys = localKeys.filter((x) => x.provider === k.provider && x.pool_type === 'general');
        const idx = providerKeys.indexOf(k);
        const prefix = k.provider.toUpperCase();
        acc.push(`${prefix}_API_KEY_${idx}=${k.key}`);
        return acc;
      }, []);
    const sentinel = localKeys.find((k) => k.pool_type === 'sentinel');
    if (sentinel) lines.push(`GEMINI_SENTINEL_KEY=${sentinel.key}`);
    navigator.clipboard.writeText(lines.join('\n'));
    setCopied('env');
    setTimeout(() => setCopied(null), 2000);
  };

  const providerGroups = PROVIDERS.map((p) => ({
    ...p,
    keys: localKeys.filter((k) => k.provider === p.id),
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Credentials</h1>
          <p className="text-sm text-muted mt-0.5">API key manager. Import .env or add manually.</p>
        </div>
        {localKeys.length > 0 && (
          <button
            onClick={copyEnvSnippet}
            className="text-xs font-mono text-muted hover:text-ink border border-border px-3 py-1.5 rounded transition-colors"
          >
            {copied === 'env' ? '✓ Copied' : '⊕ Copy .env'}
          </button>
        )}
      </div>

      {/* .env Import */}
      <Card title="Import .env File">
        <div className="px-5 py-4">
          <div
            className="border-2 border-dashed border-border rounded-lg px-6 py-10 text-center cursor-pointer hover:border-ink/30 transition-colors group"
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) readFile(f); }}
          >
            <p className="text-3xl mb-2 group-hover:scale-110 transition-transform">↓</p>
            <p className="text-sm font-medium text-ink">Drop .env file here</p>
            <p className="text-xs text-muted mt-1">Auto-parses GROQ_KEY_0, GEMINI_SENTINEL_KEY, etc.</p>
          </div>
          <input ref={fileRef} type="file" accept=".env,text/plain,.txt" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) readFile(f); }} />
        </div>
      </Card>

      {/* Import Preview */}
      {importPreview && (
        <Card title={`Preview — ${importPreview.length} key${importPreview.length !== 1 ? 's' : ''} found`}>
          <div className="px-5 py-4 space-y-1">
            {importPreview.map((k, i) => (
              <div key={i} className="flex items-center justify-between py-2 border-b border-border last:border-0">
                <div className="flex items-center gap-3">
                  <Badge variant={k.pool_type === 'sentinel' ? 'blue' : 'gray'}>{k.pool_type}</Badge>
                  <span className="text-xs text-muted capitalize">{k.provider}</span>
                  <span className="text-xs font-mono text-muted">{k.env_key}</span>
                </div>
                <span className="text-xs font-mono text-muted">{maskKey(k.key)}</span>
              </div>
            ))}
            <div className="flex gap-3 pt-3">
              <button
                onClick={confirmImport}
                className="bg-ink text-white text-sm px-4 py-2 rounded font-medium hover:bg-ink/80 transition-colors"
              >
                Import {importPreview.length} keys
              </button>
              <button
                onClick={() => { setImportPreview(null); if (fileRef.current) fileRef.current.value = ''; }}
                className="text-sm px-4 py-2 rounded border border-border hover:bg-bg transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </Card>
      )}

      {/* Add Manual */}
      <Card title="Add Key">
        <div className="px-5 py-4">
          <div className="grid grid-cols-4 gap-3">
            <div>
              <label className="text-[11px] text-muted font-mono block mb-1.5">Provider</label>
              <select
                value={form.provider}
                onChange={(e) => setForm((f) => ({ ...f, provider: e.target.value }))}
                className="w-full text-sm border border-border rounded px-3 py-2 bg-surface text-ink focus:outline-none focus:ring-1 focus:ring-ink/20"
              >
                {PROVIDERS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
              </select>
            </div>
            <div className="col-span-2">
              <label className="text-[11px] text-muted font-mono block mb-1.5">API Key</label>
              <input
                type="text"
                value={form.key}
                onChange={(e) => setForm((f) => ({ ...f, key: e.target.value }))}
                onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
                placeholder="gsk_..."
                className="w-full text-sm font-mono border border-border rounded px-3 py-2 bg-surface text-ink placeholder:text-border focus:outline-none focus:ring-1 focus:ring-ink/20"
              />
            </div>
            <div>
              <label className="text-[11px] text-muted font-mono block mb-1.5">Pool</label>
              <select
                value={form.pool_type}
                onChange={(e) => setForm((f) => ({ ...f, pool_type: e.target.value }))}
                className="w-full text-sm border border-border rounded px-3 py-2 bg-surface text-ink focus:outline-none focus:ring-1 focus:ring-ink/20"
              >
                <option value="general">General</option>
                <option value="sentinel">Sentinel</option>
              </select>
            </div>
          </div>
          <button
            onClick={handleAdd}
            disabled={!form.key.trim()}
            className="mt-3 bg-ink text-white text-sm px-4 py-2 rounded font-medium hover:bg-ink/80 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Add Key
          </button>
        </div>
      </Card>

      {/* Keys by Provider */}
      {providerGroups.map(({ id, label, color, keys }) => (
        <Card
          key={id}
          title={
            <>
              <span className="w-2 h-2 rounded-full inline-block mr-2" style={{ backgroundColor: color }} />
              {label}
              <span className="text-xs font-mono text-muted ml-1">({keys.length})</span>
            </>
          }
        >
          {keys.length === 0 ? (
            <div className="px-5 py-4 text-xs text-muted font-mono">No keys added</div>
          ) : (
            <div className="px-5">
              {keys.map((k) => (
                <div key={k.id} className="flex items-center justify-between py-3 border-b border-border last:border-0">
                  <div className="flex items-center gap-3">
                    <Badge variant={k.pool_type === 'sentinel' ? 'blue' : 'gray'}>{k.pool_type}</Badge>
                    <span className="text-xs font-mono text-muted">
                      {showKey[k.id] ? k.key : maskKey(k.key)}
                    </span>
                  </div>
                  <div className="flex items-center gap-4">
                    <button
                      onClick={() => setShowKey((s) => ({ ...s, [k.id]: !s[k.id] }))}
                      className="text-[11px] font-mono text-muted hover:text-ink transition-colors"
                    >
                      {showKey[k.id] ? 'hide' : 'show'}
                    </button>
                    <button
                      onClick={() => handleRemove(k.id)}
                      className="text-[11px] font-mono text-status-red hover:text-red-700 transition-colors"
                    >
                      remove
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      ))}

      <div className="text-xs text-muted font-mono bg-bg border border-border rounded-lg px-4 py-3 leading-relaxed">
        <strong className="text-ink">Note:</strong> Keys stored in session only. To persist: copy .env snippet above → add as HF Space Repository Secrets.
        Name format: <span className="text-ink">GROQ_API_KEY_0</span>, <span className="text-ink">GROQ_API_KEY_1</span>, ...
        Sentinel key: <span className="text-ink">GEMINI_SENTINEL_KEY</span>.
      </div>
    </div>
  );
}
