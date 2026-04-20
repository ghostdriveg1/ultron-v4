// website/src/components/CopyButton.jsx
import { useState } from 'react';
export default function CopyButton({ text, className = '' }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000); } catch {}
  };
  return (
    <button onClick={copy} className={`text-[11px] font-mono text-muted hover:text-ink transition-colors ${className}`} title="Copy">
      {copied ? '✓ copied' : 'copy'}
    </button>
  );
}
