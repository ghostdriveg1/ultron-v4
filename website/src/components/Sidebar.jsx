import { NavLink } from 'react-router-dom';

const NAV = [
  { to: '/', label: 'Dashboard', icon: '◎' },
  { to: '/credentials', label: 'Credentials', icon: '⊞' },
  { to: '/memory', label: 'Memory', icon: '◈' },
  { to: '/projects', label: 'Projects', icon: '◷' },
  { to: '/sentinel', label: 'Sentinel', icon: '◉' },
  { to: '/settings', label: 'Settings', icon: '⊙' },
];

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-screen w-52 border-r border-border bg-surface flex flex-col z-10">
      <div className="px-5 py-6 border-b border-border">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold tracking-tight text-ink">ULTRON</span>
          <span className="text-[10px] font-mono text-muted bg-bg px-1.5 py-0.5 rounded border border-border">V4</span>
        </div>
        <p className="text-[11px] text-muted mt-0.5 font-mono">Mission Control</p>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {NAV.map(({ to, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded text-sm transition-colors ${
                isActive
                  ? 'bg-ink text-white font-medium'
                  : 'text-muted hover:text-ink hover:bg-bg'
              }`
            }
          >
            <span className="font-mono text-xs w-4 flex-shrink-0">{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="px-5 py-4 border-t border-border">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-status-green" />
          <span className="text-xs text-muted font-mono">ghost@svnit</span>
        </div>
      </div>
    </aside>
  );
}
