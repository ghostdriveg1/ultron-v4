// website/src/components/Spinner.jsx
export default function Spinner({ size = 'sm' }) {
  const cls = size === 'sm' ? 'h-3 w-3 border' : 'h-5 w-5 border-2';
  return <span className={`inline-block rounded-full border-ink/20 border-t-ink animate-spin ${cls}`} aria-label="Loading" />;
}
