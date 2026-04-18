import { useState, useEffect } from 'react';
import Card from '../components/Card';

const DEFAULTS = {
  brainUrl: 'https://ghostdrive1-ultron1.hf.space',
  cfRouterUrl: 'https://ultron-brain.ghostdriveg1.workers.dev',
  apiWorkerUrl: 'https://ultron-api.ghostdriveg1.workers.dev',
  authToken: '',
  discordGhostUid: '1356180323058057326',
};

function Field({ label, value, onChange, type = 'text', mono = false, readOnly = false }) {
  return (
    <div>
      <label className="text-[11px] text-muted font-mono block mb-1.5">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        readOnly={readOnly}
        className={`w-full text-sm border border-border rounded px-3 py-2 bg-surface text-ink focus:outline-none focus:ring-1 focus:ring-ink/20 ${
          mono ? 'font-mono' : ''
        } ${readOnly ? 'text-muted cursor-default' : ''}`}
      />
    </div>
  );
}

export default function Settings() {
  const [s, setS] = useState(DEFAULTS);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    try {
      const stored = localStorage.getItem('ultron_settings');
      if (stored) setS({ ...DEFAULTS, ...JSON.parse(stored) });
    } catch {}
  }, []);

  const set = (key) => (val) => setS((prev) => ({ ...prev, [key]: val }));

  const save = () => {
    localStorage.setItem('ultron_settings', JSON.stringify(s));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-sm text-muted mt-0.5">Saved to browser local storage.</p>
      </div>

      <Card title="Infrastructure URLs">
        <div className="px-5 py-5 space-y-4">
          <Field label="Brain URL (HF Space)" value={s.brainUrl} onChange={set('brainUrl')} mono />
          <Field label="CF Router URL" value={s.cfRouterUrl} onChange={set('cfRouterUrl')} mono />
          <Field label="API Worker URL (this website)" value={s.apiWorkerUrl} onChange={set('apiWorkerUrl')} mono />
          <Field label="Auth Token" value={s.authToken} onChange={set('authToken')} type="password" mono />
        </div>
      </Card>

      <Card title="Discord">
        <div className="px-5 py-5">
          <Field label="Ghost Discord UID" value={s.discordGhostUid} onChange={set('discordGhostUid')} mono />
        </div>
      </Card>

      <Card title="Quick Reference">
        <div className="px-5 py-5">
          <div className="grid grid-cols-2 gap-x-8 gap-y-2.5">
            {[
              ['Repo', 'ghostdriveg1/ultron-v4'],
              ['HF Space', 'ghostdrive1/ultron-brain'],
              ['CF Account', 'c2ed2ecab1a35b2...'],
              ['KV ID (v3)', '77184c17886d47f...'],
              ['KV routing key', 'ultron:routing:v3'],
              ['Discord bot', 'ultron#2628'],
              ['Brain port', '7860 (HF) / 8000 (local)'],
              ['Python', '3.10.11'],
            ].map(([k, v]) => (
              <div key={k} className="flex gap-3">
                <span className="text-[11px] font-mono text-muted w-32 flex-shrink-0">{k}</span>
                <span className="text-[11px] font-mono text-ink truncate">{v}</span>
              </div>
            ))}
          </div>
        </div>
      </Card>

      <div className="flex justify-end">
        <button
          onClick={save}
          className="bg-ink text-white text-sm px-6 py-2 rounded font-medium hover:bg-ink/80 transition-colors"
        >
          {saved ? '✓ Saved' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
}
