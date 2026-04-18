import Card from '../components/Card';

export default function Projects() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Projects</h1>
        <p className="text-sm text-muted mt-0.5">Every task, deadline, and autonomous R&D session logged here.</p>
      </div>

      <Card title="Project Timeline">
        <div className="px-5 py-10 text-center">
          <p className="text-3xl mb-4 text-border">◷</p>
          <p className="text-sm font-medium text-ink">No projects yet</p>
          <p className="text-xs text-muted mt-1">Give Ultron a task with a deadline via the brain /infer endpoint.</p>
          <p className="text-xs text-muted mt-3 max-w-sm mx-auto leading-relaxed">
            Each project records: what was built by deadline, autonomous R&D improvements
            made after completion, and Sentinel's analysis.
          </p>
        </div>
      </Card>

      <Card title="Autonomous R&D Log">
        <div className="px-5 py-6 text-center">
          <p className="text-xs text-muted font-mono">Post-deadline improvement sessions will appear here.</p>
          <p className="text-xs text-muted mt-1">Powered by rd_loop.py + EverMemOS Foresight signals.</p>
        </div>
      </Card>
    </div>
  );
}
