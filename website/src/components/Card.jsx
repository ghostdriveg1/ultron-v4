export default function Card({ title, action, children, className = '' }) {
  return (
    <div className={`bg-surface border border-border rounded-lg overflow-hidden ${className}`}>
      {(title || action) && (
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          {typeof title === 'string' ? (
            <h3 className="text-sm font-medium text-ink">{title}</h3>
          ) : (
            <div className="text-sm font-medium text-ink flex items-center gap-2">{title}</div>
          )}
          {action && <div>{action}</div>}
        </div>
      )}
      {children}
    </div>
  );
}
