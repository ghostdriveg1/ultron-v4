const VARIANTS = {
  green: 'bg-green-50 text-status-green border-green-200',
  red: 'bg-red-50 text-status-red border-red-200',
  yellow: 'bg-yellow-50 text-status-yellow border-yellow-200',
  blue: 'bg-blue-50 text-status-blue border-blue-200',
  gray: 'bg-bg text-muted border-border',
};

const DOT_COLORS = {
  green: 'bg-status-green',
  red: 'bg-status-red',
  yellow: 'bg-status-yellow',
  blue: 'bg-status-blue',
  gray: 'bg-muted',
};

export default function Badge({ variant = 'gray', children, dot = false }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-mono border ${
        VARIANTS[variant] ?? VARIANTS.gray
      }`}
    >
      {dot && (
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${DOT_COLORS[variant] ?? DOT_COLORS.gray}`} />
      )}
      {children}
    </span>
  );
}
