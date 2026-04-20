// website/src/components/StatusDot.jsx
export default function StatusDot({ variant = 'gray' }) {
  const colors = { green: 'bg-status-green', red: 'bg-status-red', yellow: 'bg-status-yellow', blue: 'bg-status-blue', gray: 'bg-border' };
  const pulse = variant === 'green' || variant === 'yellow';
  return (
    <span className="relative flex h-2 w-2 flex-shrink-0">
      {pulse && <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-60 ${colors[variant] ?? colors.gray}`} />}
      <span className={`relative inline-flex rounded-full h-2 w-2 ${colors[variant] ?? colors.gray}`} />
    </span>
  );
}
