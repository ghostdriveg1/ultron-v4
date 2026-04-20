// website/src/components/EmptyState.jsx
export default function EmptyState({ icon = '◎', title, sub }) {
  return (
    <div className="py-10 text-center space-y-2">
      <p className="text-3xl text-border">{icon}</p>
      {title && <p className="text-sm font-medium text-ink">{title}</p>}
      {sub && <p className="text-xs text-muted max-w-sm mx-auto leading-relaxed">{sub}</p>}
    </div>
  );
}
