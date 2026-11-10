import { useEffect, useMemo, useRef, useState } from "react";
import { ask, getDocuments } from "./api";

const EXAMPLES = [
  {
    icon: "home",
    title: "Арендная плата",
    text: "Какая арендная плата по договору с ООО «Северный Склад» и до какого числа её вносить?",
  },
  {
    icon: "briefcase",
    title: "Испытательный срок",
    text: "Какой максимальный срок испытания можно установить главному бухгалтеру?",
  },
 ];

const STORAGE_KEYS = {
  view: "osnovanie.activeView",
  messages: "osnovanie.messages",
  draft: "osnovanie.draft",
  sidebarCollapsed: "osnovanie.sidebarCollapsed",
  sessions: "osnovanie.sessions",
  activeSessionId: "osnovanie.activeSessionId",
};

function readStored(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function writeStored(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // localStorage может быть недоступен — не ломаем интерфейс
  }
}


function makeSessionId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `chat-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function makeSessionTitle(messages) {
  const firstQuestion = messages.find((item) => item.role === "user");
  if (!firstQuestion) return "";
  const text = firstQuestion.text.replace(/\s+/g, " ").trim();
  return text.length > 38 ? `${text.slice(0, 38)}…` : text;
}

function createSession(messages = []) {
  return {
    id: makeSessionId(),
    title: makeSessionTitle(messages),
    messages,
    updatedAt: Date.now(),
  };
}

function loadSessions() {
  const stored = readStored(STORAGE_KEYS.sessions, null);
  if (Array.isArray(stored) && stored.length > 0) {
    return stored
      .filter((session) => session && Array.isArray(session.messages))
      .map((session) => ({
        ...session,
        title: session.title || makeSessionTitle(session.messages),
        updatedAt: session.updatedAt || Date.now(),
      }));
  }

  // мягкая миграция со старого формата: один чат в osnovanie.messages
  const legacyMessages = readStored(STORAGE_KEYS.messages, []);
  return [createSession(Array.isArray(legacyMessages) ? legacyMessages : [])];
}

function loadInitialSessionState() {
  const sessions = loadSessions();
  const storedId = readStored(STORAGE_KEYS.activeSessionId, null);
  const activeId = sessions.some((session) => session.id === storedId)
    ? storedId
    : sessions[0].id;
  return { sessions, activeId };
}

function Icon({ name, size = 20, strokeWidth = 1.8 }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": "true",
  };

  const paths = {
    chat: (
      <>
        <path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 9.6 9.6 0 0 1-3.2-.6L3 21l1.7-5.8A8.4 8.4 0 0 1 12 3.1a8.4 8.4 0 0 1 9 8.4Z" />
        <path d="M8.5 11.5h.01M12 11.5h.01M15.5 11.5h.01" />
      </>
    ),
    file: (
      <>
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" />
        <path d="M14 2v6h6" />
        <path d="M8 13h8M8 17h6" />
      </>
    ),
    history: (
      <>
        <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
        <path d="M3 3v5h5" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    send: (
      <>
        <path d="M12 19V5" />
        <path d="m5 12 7-7 7 7" />
      </>
    ),
    sparkle: (
      <>
        <path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9Z" />
        <path d="M19 3v4M21 5h-4M5 17v3M6.5 18.5h-3" />
      </>
    ),
    shield: (
      <>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
        <path d="m9 12 2 2 4-4" />
      </>
    ),
    plus: (
      <>
        <path d="M12 5v14M5 12h14" />
      </>
    ),
    trash: (
      <>
        <path d="M3 6h18" />
        <path d="M8 6V4h8v2" />
        <path d="M19 6l-1 14H6L5 6" />
        <path d="M10 11v6M14 11v6" />
      </>
    ),
    edit: (
      <>
        <path d="M12 20h9" />
        <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
      </>
    ),
    more: (
      <>
        <path d="M5 12h.01M12 12h.01M19 12h.01" />
      </>
    ),
    arrow: (
      <>
        <path d="M5 12h14M13 6l6 6-6 6" />
      </>
    ),
    chevronLeft: (
      <>
        <path d="m15 18-6-6 6-6" />
      </>
    ),
    chevronRight: (
      <>
        <path d="m9 18 6-6-6-6" />
      </>
    ),
    scale: (
      <>
        <path d="M12 3v18M5 6h14M7 6l-3 7h6L7 6ZM17 6l-3 7h6l-3-7Z" />
        <path d="M8 21h8" />
      </>
    ),
    home: (
      <>
        <path d="m3 10 9-7 9 7" />
        <path d="M5 9v11h14V9" />
        <path d="M9 20v-6h6v6" />
      </>
    ),
    briefcase: (
      <>
        <rect x="3" y="7" width="18" height="13" rx="2" />
        <path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M3 12h18" />
        <path d="M10 12v2h4v-2" />
      </>
    ),
    book: (
      <>
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z" />
      </>
    ),
    user: (
      <>
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21a8 8 0 0 1 16 0" />
      </>
    ),
    check: (
      <>
        <path d="m5 12 4 4L19 6" />
      </>
    ),
  };

  return <svg {...common}>{paths[name] || paths.sparkle}</svg>;
}

function BrandMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <Icon name="scale" size={24} strokeWidth={2} />
    </div>
  );
}

function LegalIllustration() {
  return (
    <svg
      className="legal-illustration"
      viewBox="0 0 420 320"
      role="img"
      aria-label="Иллюстрация: весы правосудия и юридические документы"
    >
      <defs>
        <linearGradient id="paper" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="1" stopColor="#eef6ff" />
        </linearGradient>
        <linearGradient id="blue" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#2563eb" />
          <stop offset="1" stopColor="#0ea5e9" />
        </linearGradient>
        <filter id="softShadow" x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0" dy="12" stdDeviation="14" floodColor="#1d4ed8" floodOpacity="0.14" />
        </filter>
      </defs>

      <circle cx="220" cy="155" r="132" fill="#dbeafe" opacity="0.7" />
      <circle cx="80" cy="70" r="38" fill="#e0f2fe" opacity="0.9" />
      <circle cx="350" cy="230" r="52" fill="#e0f2fe" opacity="0.8" />

      <g filter="url(#softShadow)">
        <rect x="54" y="76" width="132" height="172" rx="18" fill="url(#paper)" stroke="#c7d9f4" />
        <rect x="76" y="102" width="74" height="10" rx="5" fill="#bfdbfe" />
        <rect x="76" y="128" width="88" height="7" rx="3.5" fill="#dbeafe" />
        <rect x="76" y="145" width="70" height="7" rx="3.5" fill="#dbeafe" />
        <rect x="76" y="162" width="82" height="7" rx="3.5" fill="#dbeafe" />
        <rect x="76" y="190" width="54" height="8" rx="4" fill="#93c5fd" />
        <rect x="76" y="208" width="76" height="8" rx="4" fill="#dbeafe" />
      </g>

      <g filter="url(#softShadow)">
        <circle cx="286" cy="132" r="82" fill="#ffffff" stroke="#c7d9f4" />
        <path d="M286 72v112" stroke="#1d4ed8" strokeWidth="5" strokeLinecap="round" />
        <path d="M236 98h100" stroke="#1d4ed8" strokeWidth="5" strokeLinecap="round" />
        <path d="M236 98l-18 40h36Z" fill="#dbeafe" stroke="#1d4ed8" strokeWidth="3" strokeLinejoin="round" />
        <path d="M336 98l-18 40h36Z" fill="#dbeafe" stroke="#1d4ed8" strokeWidth="3" strokeLinejoin="round" />
        <circle cx="286" cy="72" r="8" fill="url(#blue)" />
        <path d="M262 184h48" stroke="#1d4ed8" strokeWidth="5" strokeLinecap="round" />
      </g>

      <g filter="url(#softShadow)">
        <rect x="300" y="214" width="82" height="64" rx="16" fill="#ffffff" stroke="#c7d9f4" />
        <circle cx="326" cy="246" r="15" fill="#dcfce7" />
        <path d="M319 246l5 5 9-11" stroke="#15803d" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        <rect x="348" y="238" width="18" height="6" rx="3" fill="#dbeafe" />
        <rect x="348" y="250" width="14" height="6" rx="3" fill="#dbeafe" />
      </g>

      <g fill="#60a5fa">
        <circle cx="60" cy="264" r="5" />
        <circle cx="100" cy="286" r="3" />
        <circle cx="368" cy="86" r="4" />
        <circle cx="386" cy="70" r="2.5" />
      </g>
      <path d="M40 180l8-8M40 180l-8-8" stroke="#93c5fd" strokeWidth="3" strokeLinecap="round" />
      <path d="M382 150l8-8M382 150l-8-8" stroke="#93c5fd" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

function Sidebar({ activeView, onChangeView, onNewChat, onHome, collapsed, onToggleCollapse, sessions, activeSessionId, onSelectSession, onRenameSession, onDeleteSession }) {
  const items = [
    { id: "documents", label: "Документы", icon: "file" },
  ];
  const [openMenuId, setOpenMenuId] = useState(null);
  const [menuPosition, setMenuPosition] = useState({ top: 0, left: 0 });

  const menuSession = sessions.find((session) => session.id === openMenuId);

  useEffect(() => {
    if (!openMenuId) return;
    const close = () => setOpenMenuId(null);
    window.addEventListener("click", close);
    window.addEventListener("resize", close);
    window.addEventListener("scroll", close, true);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("resize", close);
      window.removeEventListener("scroll", close, true);
    };
  }, [openMenuId]);

  function openChatMenu(event, sessionId) {
    event.stopPropagation();
    const rect = event.currentTarget.getBoundingClientRect();
    setMenuPosition({
      top: Math.max(12, Math.min(rect.bottom + 6, window.innerHeight - 104)),
      left: Math.max(12, rect.right - 168),
    });
    setOpenMenuId(sessionId);
  }

  function handleRename() {
    if (!menuSession) return;
    const nextTitle = window.prompt("Новое название чата", menuSession.title || "");
    if (nextTitle === null) return;
    const title = nextTitle.trim();
    if (!title) return;
    onRenameSession(menuSession.id, title);
    setOpenMenuId(null);
  }

  function handleDelete() {
    if (!menuSession) return;
    const confirmed = window.confirm(`Удалить чат «${menuSession.title || "Новый чат"}»?`);
    if (!confirmed) return;
    onDeleteSession(menuSession.id);
    setOpenMenuId(null);
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <button type="button" className="brand" onClick={onHome} title="На главную">
          <BrandMark />
          <span>
            <span className="brand-name">Основание</span>
            <span className="brand-sub">юридический ассистент</span>
          </span>
        </button>
        <button
          type="button"
          className="sidebar-toggle"
          onClick={onToggleCollapse}
          title={collapsed ? "Развернуть меню" : "Свернуть меню"}
          aria-label={collapsed ? "Развернуть меню" : "Свернуть меню"}
        >
          <Icon name={collapsed ? "chevronRight" : "chevronLeft"} size={17} />
        </button>
      </div>

      <nav className="nav" aria-label="Основная навигация">
        {items.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`nav-item ${activeView === item.id ? "active" : ""}`}
            onClick={() => onChangeView(item.id)}
            title={item.label}
          >
            <Icon name={item.icon} size={19} />
            <span className="nav-label">{item.label}</span>
            {item.id === "chat" && <span className="nav-badge">AI</span>}
          </button>
        ))}
      </nav>

      <div className="sidebar-history">
        <button
          type="button"
          className="sidebar-history-add"
          onClick={onNewChat}
          title="Новый чат"
          aria-label="Новый чат"
        >
          <Icon name="plus" size={18} />
        </button>
        <div className="sidebar-history-list">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`history-chat-row ${session.id === activeSessionId ? "active" : ""}`}
            >
              <button
                type="button"
                className="history-chat-item"
                onClick={() => onSelectSession(session.id)}
                title={session.title || "Новый чат"}
              >
                <Icon name="chat" size={15} />
                <span>{session.title || "Новый чат"}</span>
              </button>
              <button
                type="button"
                className="history-chat-more"
                onClick={(event) => openChatMenu(event, session.id)}
                title="Действия с чатом"
                aria-label="Действия с чатом"
              >
                <Icon name="more" size={16} strokeWidth={2.6} />
              </button>
            </div>
          ))}
        </div>
      </div>

      {openMenuId && menuSession && (
        <div
          className="session-menu"
          style={{ top: menuPosition.top, left: menuPosition.left }}
          onClick={(event) => event.stopPropagation()}
        >
          <button type="button" onClick={handleRename}>
            <Icon name="edit" size={15} />
            Переименовать
          </button>
          <button type="button" className="danger" onClick={handleDelete}>
            <Icon name="trash" size={15} />
            Удалить
          </button>
        </div>
      )}

      <div className="sidebar-note">
        <div className="note-icon">
          <Icon name="shield" size={20} />
        </div>
        <h3>Только проверенные источники</h3>
        <p>
          Ответы строятся по базе знаний. Если подтверждения нет, ассистент честно
          откажется.
        </p>
      </div>

      <div className="sidebar-footer">
        <button type="button" className="new-chat" onClick={onNewChat} title="Новый диалог">
          <Icon name="plus" size={18} />
          <span>Новый диалог</span>
        </button>
        <div className="user-card">
          <div className="user-avatar">Ю</div>
          <div>
            <div className="user-name">Юрист</div>
            <div className="user-role">демо-режим</div>
          </div>
        </div>
      </div>
    </aside>
  );
}

function Topbar({ activeView }) {
  const labels = {
    home: "Главная",
    chat: "Чат",
    documents: "Документы",
  };

  return (
    <header className="topbar">
      <div className="topbar-title">
        <div className="topbar-eyebrow">{labels[activeView]}</div>
        <h1>Основание</h1>
        <p>Интеллектуальный помощник юриста</p>
      </div>
      <div className="topbar-actions">
        <span className="status-pill">
          <span className="status-dot" />
          Демо-режим
        </span>
      </div>
    </header>
  );
}

function Hero() {
  return (
    <section className="hero">
      <div className="hero-copy">
        <div className="eyebrow">
          <span className="eyebrow-icon">
            <Icon name="sparkle" size={15} />
          </span>
          Юридический ИИ-ассистент
        </div>
        <h2>
          Найдите норму.
          <br />
          <span>Обоснуйте ответ.</span>
        </h2>
        <p>
          Задайте вопрос по договорам, регламентам и законам. «Основание» найдёт
          нужные положения и покажет источники, на которые можно опереться.
        </p>
      </div>
      <div className="hero-art">
        <LegalIllustration />
      </div>
    </section>
  );
}

function ExampleCard({ example, onClick }) {
  return (
    <button type="button" className="example-card" onClick={() => onClick(example.text)}>
      <span className="example-icon">
        <Icon name={example.icon} size={20} />
      </span>
      <span className="example-copy">
        <strong>{example.title}</strong>
        <span>{example.text}</span>
      </span>
      <span className="example-arrow">
        <Icon name="arrow" size={17} />
      </span>
    </button>
  );
}

function EmptyChat() {
  return (
    <div className="empty-state">
      <p>Опишите ситуацию или вставьте пункт документа.</p>
    </div>
  );
}

function Message({ item }) {
  const isUser = item.role === "user";

  return (
    <article className={`message-row ${isUser ? "user" : "assistant"}`}>
      <div className="message-content">
        <div className="bubble">{item.text}</div>

        {item.refusal && (
          <div className="refusal-badge">
            <Icon name="shield" size={15} />
            Обоснованный отказ
          </div>
        )}

        {item.sources?.length > 0 && (
          <div className="sources-box">
            <div className="sources-title">
              <Icon name="book" size={15} />
              Источники
            </div>
            {item.sources.map((source, index) => (
              <div className="source-item" key={`${source.doc}-${index}`}>
                <div className="source-head">
                  <span>{source.doc}</span>
                  {source.clause && <span>{source.clause}</span>}
                </div>
                {source.text && <p>{source.text}</p>}
              </div>
            ))}
          </div>
        )}
      </div>
    </article>
  );
}

function TypingIndicator() {
  return (
    <div className="message-row assistant" role="status">
      <div className="typing" aria-label="Ассистент ищет ответ">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}

function Composer({ message, setMessage, loading, onSubmit, inputRef }) {
  function handleKeyDown(event) {
    if (event.nativeEvent.isComposing) return;
    if (event.key === "Enter" && !event.shiftKey) {
      onSubmit(event);
    }
  }

  return (
    <form className="composer" onSubmit={onSubmit}>
      <div className="composer-box">
        <span className="composer-icon">
          <Icon name="sparkle" size={19} />
        </span>
        <textarea
          ref={inputRef}
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          onKeyDown={handleKeyDown}
          rows={2}
          placeholder="Например: какая арендная плата по договору с ООО «Северный Склад»?"
          aria-label="Вопрос ассистенту"
        />
        <button
          type="submit"
          className="send-button"
          disabled={loading || !message.trim()}
          aria-label="Отправить сообщение"
          title="Отправить"
        >
          <Icon name="send" size={20} strokeWidth={2.1} />
        </button>
      </div>
      <p className="composer-hint">
        Основание может ошибаться. Проверяйте ссылки на документы перед принятием
        решений.
      </p>
    </form>
  );
}

function HomeView({ onExample, onStartChat }) {
  return (
    <div className="workspace-content home-page">
      <Hero />

      <section className="examples">
        <div className="section-heading">
          <h2>С чего начать</h2>
        </div>
        <div className="examples-grid">
          {EXAMPLES.map((example) => (
            <ExampleCard key={example.title} example={example} onClick={onExample} />
          ))}
        </div>
      </section>

      <section className="start-chat-bar">
        <div className="start-chat-copy">
          <div className="start-chat-title">Готовы задать вопрос?</div>
          <p>Откройте чат — там уже ждёт поле ввода и история диалога.</p>
        </div>
        <button type="button" className="start-chat-button" onClick={onStartChat}>
          Начать чат
          <Icon name="arrow" size={18} />
        </button>
      </section>
    </div>
  );
}

function ChatView({
  messages,
  loading,
  message,
  setMessage,
  onSubmit,
  inputRef,
  chatScrollRef,
  chatEndRef,
}) {
  return (
    <div className="workspace-content chat-page">
      <section className="chat-card">
        <div className="chat-scroll" ref={chatScrollRef}>
          {messages.length === 0 && !loading && <EmptyChat />}
          {messages.map((item, index) => (
            <Message item={item} key={index} />
          ))}
          {loading && <TypingIndicator />}
          <div ref={chatEndRef} />
        </div>

        <Composer
          message={message}
          setMessage={setMessage}
          loading={loading}
          onSubmit={onSubmit}
          inputRef={inputRef}
        />
      </section>
    </div>
  );
}

function DocumentsView({ documents, loading, error, onRetry }) {
  return (
    <div className="workspace-content">
      <section className="page-panel">
        <div className="panel-heading">
          <div>
            <div className="eyebrow">База знаний</div>
            <h2>Документы</h2>
            <p>
              Нормативные документы, договоры и внутренние регламенты, по которым
              ассистент ищет ответы.
            </p>
          </div>
          <button type="button" className="ghost-button" onClick={onRetry}>
            <Icon name="history" size={15} />
            Обновить
          </button>
        </div>

        {loading && <div className="panel-state">Загружаю список документов…</div>}

        {!loading && error && (
          <div className="panel-state error">
            <Icon name="file" size={24} />
            <strong>Не удалось получить документы</strong>
            <span>{error}</span>
          </div>
        )}

        {!loading && !error && documents.length === 0 && (
          <div className="panel-state">
            <div className="panel-state-icon">
              <Icon name="book" size={28} />
            </div>
            <strong>База знаний пока не подключена</strong>
            <span>
              Backend уже готов принять документы. Как только модуль RAG будет
              готов, список появится здесь.
            </span>
          </div>
        )}

        {!loading && !error && documents.length > 0 && (
          <div className="documents-list">
            {documents.map((document, index) => (
              <article className="document-row" key={document.id || document.doc_id || index}>
                <div className="document-icon">
                  <Icon name="file" size={20} />
                </div>
                <div>
                  <strong>{document.title || document.name || document.doc_id}</strong>
                  <span>
                    {[document.type, document.number, document.date].filter(Boolean).join(" · ") ||
                      "Документ базы знаний"}
                  </span>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

export default function App() {
  const [message, setMessage] = useState(() => {
    const value = readStored(STORAGE_KEYS.draft, "");
    return typeof value === "string" ? value : "";
  });
  const [initialSessionState] = useState(loadInitialSessionState);
  const [sessions, setSessions] = useState(initialSessionState.sessions);
  const [activeSessionId, setActiveSessionId] = useState(initialSessionState.activeId);
  const [loading, setLoading] = useState(false);
  const [activeView, setActiveView] = useState(() => {
    const value = readStored(STORAGE_KEYS.view, "home");
    return ["home", "chat", "documents"].includes(value) ? value : "home";
  });
  const [documents, setDocuments] = useState([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState("");
  const [documentsRefreshKey, setDocumentsRefreshKey] = useState(0);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() =>
    Boolean(readStored(STORAGE_KEYS.sidebarCollapsed, false))
  );

  const activeSession = sessions.find((session) => session.id === activeSessionId) || sessions[0];
  const messages = activeSession?.messages || [];
  const orderedSessions = useMemo(
    () => [...sessions].sort((a, b) => b.updatedAt - a.updatedAt),
    [sessions]
  );

  const inputRef = useRef(null);
  const chatScrollRef = useRef(null);
  const chatEndRef = useRef(null);
  const workspaceScrollRef = useRef(null);

  useEffect(() => {
    const element = chatScrollRef.current;
    if (element) {
      element.scrollTop = element.scrollHeight;
    }
  }, [messages, loading]);

  useEffect(() => {
    workspaceScrollRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [activeView]);

  useEffect(() => {
    writeStored(STORAGE_KEYS.view, activeView);
  }, [activeView]);

  useEffect(() => {
    writeStored(STORAGE_KEYS.sessions, sessions);
  }, [sessions]);

  useEffect(() => {
    writeStored(STORAGE_KEYS.activeSessionId, activeSessionId);
  }, [activeSessionId]);

  useEffect(() => {
    writeStored(STORAGE_KEYS.draft, message);
  }, [message]);

  useEffect(() => {
    writeStored(STORAGE_KEYS.sidebarCollapsed, sidebarCollapsed);
  }, [sidebarCollapsed]);

  useEffect(() => {
    if (activeView !== "documents") return;

    let cancelled = false;
    setDocumentsLoading(true);
    setDocumentsError("");

    getDocuments()
      .then((data) => {
        if (cancelled) return;
        setDocuments(Array.isArray(data.documents) ? data.documents : []);
      })
      .catch((error) => {
        if (cancelled) return;
        setDocumentsError(error.message);
      })
      .finally(() => {
        if (!cancelled) setDocumentsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [activeView, documentsRefreshKey]);

  function appendMessage(sessionId, item) {
    setSessions((prev) =>
      prev.map((session) => {
        if (session.id !== sessionId) return session;
        const nextMessages = [...session.messages, item];
        return {
          ...session,
          messages: nextMessages,
          title: session.title || makeSessionTitle(nextMessages),
          updatedAt: Date.now(),
        };
      })
    );
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const text = message.trim();
    if (!text || loading) return;

    const sessionId = activeSessionId;
    const history = messages;

    setMessage("");
    appendMessage(sessionId, { role: "user", text });
    setLoading(true);

    try {
      const data = await ask(text, history);
      appendMessage(sessionId, {
        role: "assistant",
        text: data.answer,
        sources: data.sources || [],
        refusal: data.refusal,
      });
    } catch (error) {
      appendMessage(sessionId, {
        role: "assistant",
        text: `Не удалось получить ответ: ${error.message}`,
        sources: [],
        refusal: false,
      });
    } finally {
      setLoading(false);
    }
  }

  function startNewSession(initialMessage = "") {
    const session = createSession([]);
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(session.id);
    setMessage(initialMessage);
    setActiveView("chat");
    requestAnimationFrame(() => inputRef.current?.focus());
  }

  function handleExample(question) {
    startNewSession(question);
  }

  function handleStartChat() {
    startNewSession("");
  }

  function handleNewChat() {
    startNewSession("");
  }

  function handleSelectSession(sessionId) {
    setActiveSessionId(sessionId);
    setMessage("");
    setActiveView("chat");
    requestAnimationFrame(() => inputRef.current?.focus());
  }

  function handleRenameSession(sessionId, title) {
    setSessions((prev) =>
      prev.map((session) =>
        session.id === sessionId ? { ...session, title, updatedAt: Date.now() } : session
      )
    );
  }

  function handleDeleteSession(sessionId) {
    const remaining = sessions.filter((session) => session.id !== sessionId);
    const nextSessions = remaining.length > 0 ? remaining : [createSession([])];
    setSessions(nextSessions);
    if (activeSessionId === sessionId) {
      setActiveSessionId(nextSessions[0].id);
      setActiveView("chat");
    }
  }

  function reloadDocuments() {
    setDocumentsRefreshKey((value) => value + 1);
  }

  return (
    <div className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <Sidebar
        activeView={activeView}
        onChangeView={setActiveView}
        onNewChat={handleNewChat}
        onHome={() => setActiveView("home")}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((value) => !value)}
        sessions={orderedSessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onRenameSession={handleRenameSession}
        onDeleteSession={handleDeleteSession}
      />

      <main className="workspace">
        <Topbar activeView={activeView} />

        <div
          className={`workspace-scroll ${activeView === "chat" ? "workspace-scroll-chat" : ""}`}
          ref={workspaceScrollRef}
        >
          {activeView === "home" && (
            <HomeView onExample={handleExample} onStartChat={handleStartChat} />
          )}

          {activeView === "chat" && (
            <ChatView
              messages={messages}
              loading={loading}
              message={message}
              setMessage={setMessage}
              onSubmit={handleSubmit}
              inputRef={inputRef}
              chatScrollRef={chatScrollRef}
              chatEndRef={chatEndRef}
            />
          )}

          {activeView === "documents" && (
            <DocumentsView
              documents={documents}
              loading={documentsLoading}
              error={documentsError}
              onRetry={reloadDocuments}
            />
          )}
        </div>
      </main>
    </div>
  );
}
