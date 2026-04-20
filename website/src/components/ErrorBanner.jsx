// website/src/components/ErrorBanner.jsx
export default function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="border border-red-200 bg-red-50 text-status-red text-xs px-4 py-3 rounded-lg font-mono">
      ✗ {message}
    </div>
  );
}
