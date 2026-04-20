import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Credentials from './pages/Credentials';
import Memory from './pages/Memory';
import Projects from './pages/Projects';
import Sentinel from './pages/Sentinel';
import Settings from './pages/Settings';
import Chat from './pages/Chat';

export default function App() {
  return (
    <div className="flex min-h-screen bg-bg">
      <Sidebar />
      <main className="flex-1 ml-52 px-8 py-8">
        <div className="max-w-4xl">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/credentials" element={<Credentials />} />
            <Route path="/memory" element={<Memory />} />
            <Route path="/projects" element={<Projects />} />
            <Route path="/sentinel" element={<Sentinel />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
