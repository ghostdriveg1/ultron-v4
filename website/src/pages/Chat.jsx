// website/src/pages/Chat.jsx
// Cross-project persistent chat interface.
// Architecture:
//   - Messages stored in localStorage per project_id.
//   - Each message sent via Brain /infer endpoint (POST /api/infer).
//   - Brain receives channel_id = "web_" + project_id — Discord bot uses same key namespace.
//   - This means: chat in web + chat in Discord = SAME context window in Redis.
//   - Projects sidebar: create/switch/delete local projects. Active project persists in localStorage.
//   - Ghost user_id hardcoded to Discord UID (matches lifecycle.ingest user_id key).
//
// Cross-session persistence model ("same as claude.ai memory"):
//   localStorage stores message history per project.
//   Redis (Brain) stores context window per channel_id ("web_" + project_id).
//   LifecycleEngine.ingest() called on every message — MemCell heat builds across sessions.
//   Even if localStorage is cleared, Brain retains Redis context + lifecycle memory.
//   Fully consistent across ALL interfaces (web, Discord) via shared channel_id.

import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../lib/api';

const GHOST_USER_ID = '1356180323058057326';
const STORAGE_PROJECTS_KEY = 'ultron_chat_projects';
const STORAGE_ACTIVE_KEY = 'ultron_chat_active_project';
const MAX_LOCAL_MESSAGES = 200; // per project

function genId() {
  return Math.random().toString(36).slice(2, 10);
}

function loadProjects() {
  try { return JSON.parse(localStorage.getItem(STORAGE_PROJECTS_KEY)) ?? []; } catch { return []; }
}
function saveProjects(projects) {
  localStorage.setItem(STORAGE_PROJECTS_KEY, JSON.stringify(projects));
}
function loadActiveId() {
  return localStorage.getItem(STORAGE_ACTIVE_KEY) ?? null;
}
function saveActiveId(id) {
  localStorage.setItem(STORAGE_ACTIVE_KEY, id);
}

// Massage reply text: strip internal ReAct markers
function cleanReply(text) {
  return text
    .replace(/^=== MEMORY GRAPH ===.*?(?=^===|\Z)/gms, '')
    .replace(/^\[COMPACTED HISTORY SUMMARY\].*?(?=^\[|\Z)/gms, '')
    .replace(/^\[OBSERVATION\].*?(?=^\[|\Z)/gms, '')
    .replace(/^\[LOOP WARNING\].*$/gm, '')
    .replace(/^\[TOOL (ERROR|RESULT|OK)\].*$/gm, '')
    .trim();
}

// Single message bubble
function Bubble({ msg }) {
  const isUser = msg.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      <div
        className={`max-w-[75%] px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? 'bg-ink text-white rounded-br-sm'
            : 'bg-surface border border-border text-ink rounded-bl-sm'
        }`}
      >
        {msg.content}
        {msg.error && (
          <p className="text-[11px] mt-1.5 text-red-400 font-mono">⚠️ {msg.error}</p>
        )}
        <p className={`text-[10px] mt-1.5 font-mono ${
          isUser ? 'text-white/50' : 'text-muted'
        }`}>
          {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
        </p>
      </div>
    </div>
  );
}

// Typing indicator
function Typing() {
  return (
    <div className="flex justify-start mb-3">
      <div className="bg-surface border border-border px-4 py-3 rounded-2xl rounded-bl-sm flex gap-1.5 items-center">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-muted animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
    </div>
  );
}

export default function Chat() {
  const [projects, setProjects] = useState(() => {
    const stored = loadProjects();
    if (stored.length === 0) {
      const defaultProject = { id: genId(), name: 'General', createdAt: new Date().toISOString(), messages: [] };
      saveProjects([defaultProject]);
      return [defaultProject];
    }
    return stored;
  });

  const [activeId, setActiveId] = useState(() => {
    const stored = loadActiveId();
    const projects = loadProjects();
    if (stored && projects.find((p) => p.id === stored)) return stored;
    return projects[0]?.id ?? null;
  });

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [showNewProject, setShowNewProject] = useState(false);
  const [renamingId, setRenamingId] = useState(null);
  const [renameValue, setRenameValue] = useState('');

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const activeProject = projects.find((p) => p.id === activeId) ?? projects[0];
  const messages = activeProject?.messages ?? [];

  // Persist on every projects change
  useEffect(() => { saveProjects(projects); }, [projects]);
  useEffect(() => { if (activeId) saveActiveId(activeId); }, [activeId]);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const updateActiveMessages = useCallback((updater) => {
    setProjects((prev) =>
      prev.map((p) =>
        p.id === activeId
          ? { ...p, messages: updater(p.messages).slice(-MAX_LOCAL_MESSAGES) }
          : p
      )
    );
  }, [activeId]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || loading || !activeProject) return;
    setInput('');

    const userMsg = { id: genId(), role: 'user', content: text, timestamp: Date.now() };
    updateActiveMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      // channel_id = "web_" + project_id — shared with Redis context window
      const result = await api.infer({
        message: text,
        channel_id: `web_${activeProject.id}`,
        user_id: GHOST_USER_ID,
        username: 'Ghost',
      });

      const replyText = cleanReply(result.reply ?? result.response ?? '(no response)');
      const botMsg = { id: genId(), role: 'assistant', content: replyText, timestamp: Date.now() };
      updateActiveMessages((prev) => [...prev, botMsg]);
    } catch (e) {
      const errMsg = {
        id: genId(),
        role: 'assistant',
        content: 'Brain unreachable.',
        error: e.message,
        timestamp: Date.now(),
      };
      updateActiveMessages((prev) => [...prev, errMsg]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [input, loading, activeProject, updateActiveMessages]);

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const createProject = () => {
    const name = newProjectName.trim() || `Project ${projects.length + 1}`;
    const p = { id: genId(), name, createdAt: new Date().toISOString(), messages: [] };
    setProjects((prev) => [...prev, p]);
    setActiveId(p.id);
    setNewProjectName('');
    setShowNewProject(false);
  };

  const deleteProject = (id) => {
    if (projects.length === 1) return; // keep at least one
    const remaining = projects.filter((p) => p.id !== id);
    setProjects(remaining);
    if (activeId === id) setActiveId(remaining[0].id);
  };

  const startRename = (p) => { setRenamingId(p.id); setRenameValue(p.name); };
  const confirmRename = () => {
    if (!renameValue.trim()) { setRenamingId(null); return; }
    setProjects((prev) => prev.map((p) => p.id === renamingId ? { ...p, name: renameValue.trim() } : p));
    setRenamingId(null);
  };

  const clearChat = () => {
    updateActiveMessages(() => []);
  };

  return (
    <div className="flex h-[calc(100vh-4rem)] -mt-8 -mx-8 overflow-hidden">
      {/* Projects sidebar */}
      <div className="w-56 border-r border-border bg-surface flex flex-col flex-shrink-0">
        <div className="px-4 py-4 border-b border-border">
          <div className="flex items-center justify-between">
            <p className="text-xs font-mono text-muted uppercase tracking-widest">Projects</p>
            <button
              onClick={() => setShowNewProject((v) => !v)}
              className="text-xs font-mono text-muted hover:text-ink transition-colors"
              title="New project"
            >
              +
            </button>
          </div>
          {showNewProject && (
            <div className="mt-2 flex gap-1.5">
              <input
                autoFocus
                className="flex-1 text-xs border border-border rounded px-2 py-1.5 bg-bg text-ink font-mono focus:outline-none"
                placeholder="Project name"
                value={newProjectName}
                onChange={(e) => setNewProjectName(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') createProject(); if (e.key === 'Escape') setShowNewProject(false); }}
              />
              <button onClick={createProject} className="text-xs font-mono px-2 py-1.5 border border-border rounded hover:bg-bg">✓</button>
            </div>
          )}
        </div>

        <nav className="flex-1 overflow-y-auto py-1">
          {projects.map((p) => (
            <div
              key={p.id}
              className={`group flex items-center gap-2 px-3 py-2 cursor-pointer transition-colors ${
                p.id === activeId ? 'bg-ink text-white' : 'text-muted hover:bg-bg hover:text-ink'
              }`}
              onClick={() => setActiveId(p.id)}
            >
              {renamingId === p.id ? (
                <input
                  autoFocus
                  className="flex-1 text-xs bg-transparent border-b border-white/40 outline-none text-white"
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onBlur={confirmRename}
                  onKeyDown={(e) => { if (e.key === 'Enter') confirmRename(); if (e.key === 'Escape') setRenamingId(null); }}
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <span className="flex-1 text-xs font-medium truncate">{p.name}</span>
              )}
              <div className={`hidden group-hover:flex gap-1 ${
                p.id === activeId ? 'flex' : ''
              }`}>
                <button
                  onClick={(e) => { e.stopPropagation(); startRename(p); }}
                  className={`text-[10px] font-mono ${
                    p.id === activeId ? 'text-white/60 hover:text-white' : 'text-muted hover:text-ink'
                  } transition-colors`}
                  title="Rename"
                >✎</button>
                {projects.length > 1 && (
                  <button
                    onClick={(e) => { e.stopPropagation(); deleteProject(p.id); }}
                    className={`text-[10px] font-mono ${
                      p.id === activeId ? 'text-white/60 hover:text-white' : 'text-muted hover:text-status-red'
                    } transition-colors`}
                    title="Delete"
                  >×</button>
                )}
              </div>
            </div>
          ))}
        </nav>

        <div className="px-4 py-3 border-t border-border">
          <p className="text-[10px] font-mono text-muted leading-relaxed">
            channel_id = web_{activeProject?.id?.slice(0, 6)}&hellip;<br />
            Shared with Discord context.
          </p>
        </div>
      </div>

      {/* Chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-surface flex-shrink-0">
          <div>
            <p className="text-sm font-bold text-ink">{activeProject?.name ?? 'Chat'}</p>
            <p className="text-[11px] font-mono text-muted mt-0.5">
              {messages.length} message{messages.length !== 1 ? 's' : ''}
              {' · '}
              <span className="text-status-green">● live</span>
            </p>
          </div>
          <button
            onClick={clearChat}
            className="text-[11px] font-mono text-muted hover:text-status-red transition-colors"
            title="Clear local history"
          >
            clear history
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {messages.length === 0 && !loading && (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <p className="text-4xl mb-4">⚡</p>
              <p className="text-base font-bold text-ink">Start a conversation</p>
              <p className="text-xs text-muted mt-1 max-w-xs leading-relaxed">
                Messages are persistent across sessions. Both this interface and Discord share the same
                Brain context window for this project.
              </p>
            </div>
          )}
          {messages.map((msg) => <Bubble key={msg.id} msg={msg} />)}
          {loading && <Typing />}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="px-6 py-4 border-t border-border bg-surface flex-shrink-0">
          <div className="flex gap-3 items-end">
            <textarea
              ref={inputRef}
              rows={1}
              className="flex-1 text-sm border border-border rounded-xl px-4 py-3 bg-bg text-ink resize-none focus:outline-none focus:ring-1 focus:ring-ink/20 font-sans leading-relaxed max-h-40 overflow-y-auto"
              placeholder="Message Ultron... (Enter to send, Shift+Enter for newline)"
              value={input}
              onChange={(e) => { setInput(e.target.value); e.target.style.height = 'auto'; e.target.style.height = e.target.scrollHeight + 'px'; }}
              onKeyDown={handleKey}
              disabled={loading}
            />
            <button
              onClick={send}
              disabled={loading || !input.trim()}
              className="flex-shrink-0 bg-ink text-white px-4 py-3 rounded-xl text-sm font-medium hover:bg-ink/80 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? '...' : '↑'}
            </button>
          </div>
          <p className="text-[10px] font-mono text-muted mt-2">
            Ultron V4 · ReAct + MOA · {['Groq', 'Cerebras', 'Together', 'OpenRouter', 'Gemini'].join(' · ')}
          </p>
        </div>
      </div>
    </div>
  );
}
