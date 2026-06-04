/**
 * Doc-Intel Frontend — app.js
 * Connects the chat UI to the FastAPI RAG backend via SSE streaming.
 */

const API_BASE = window.location.origin;

// ── State ─────────────────────────────────────────────────────────────────────
let isLoading = false;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const messagesEl    = document.getElementById('messages');
const welcomeEl     = document.getElementById('welcome');
const queryInput    = document.getElementById('queryInput');
const sendBtn       = document.getElementById('sendBtn');
const newChatBtn    = document.getElementById('newChatBtn');
const modelLabel    = document.getElementById('modelLabel');
const statusDot     = document.getElementById('statusDot');

// Nav
const navChat       = document.getElementById('navChat');
const navIngest     = document.getElementById('navIngest');
const navDocs       = document.getElementById('navDocs');
const navStatus     = document.getElementById('navStatus');
const viewChat      = document.getElementById('viewChat');
const viewIngest    = document.getElementById('viewIngest');
const viewDocs      = document.getElementById('viewDocs');
const viewStatus    = document.getElementById('viewStatus');

// Ingest
const ingestBtn     = document.getElementById('ingestBtn');
const dataDirInput  = document.getElementById('dataDir');
const ingestLog     = document.getElementById('ingestLog');
const logContent    = document.getElementById('logContent');
const clearLog      = document.getElementById('clearLog');

// Status
const statusGrid    = document.getElementById('statusGrid');
const refreshStatus = document.getElementById('refreshStatus');

// Documents
const docsFileList      = document.getElementById('docsFileList');
const docsPlaceholder   = document.getElementById('docsPlaceholder');
const docsContentWrap   = document.getElementById('docsContentWrap');
const docsContentHeader = document.getElementById('docsContentHeader');
const docsContentBody   = document.getElementById('docsContentBody');
const refreshDocs       = document.getElementById('refreshDocs');

// ── Nav ───────────────────────────────────────────────────────────────────────
function showView(view) {
  [viewChat, viewIngest, viewDocs, viewStatus].forEach(v => v.classList.remove('active'));
  [navChat, navIngest, navDocs, navStatus].forEach(n => n.classList.remove('active'));
  view.classList.add('active');
}

navChat.addEventListener('click', () => {
  showView(viewChat);
  navChat.classList.add('active');
});

navIngest.addEventListener('click', () => {
  showView(viewIngest);
  navIngest.classList.add('active');
});

navDocs.addEventListener('click', () => {
  showView(viewDocs);
  navDocs.classList.add('active');
  loadDocuments();
});

navStatus.addEventListener('click', () => {
  showView(viewStatus);
  navStatus.classList.add('active');
  loadStatus();
});

// ── Health Check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('not ok');
    const data = await res.json();
    modelLabel.textContent = data.llm_model ?? 'Unknown';
    statusDot.className = 'status-dot online';
  } catch {
    modelLabel.textContent = 'Offline';
    statusDot.className = 'status-dot offline';
  }
}

// ── Textarea auto-resize ──────────────────────────────────────────────────────
queryInput.addEventListener('input', () => {
  queryInput.style.height = 'auto';
  queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
});

queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});

sendBtn.addEventListener('click', sendQuery);
newChatBtn.addEventListener('click', clearChat);

// ── Suggestion chips ──────────────────────────────────────────────────────────
document.querySelectorAll('.suggestion-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    queryInput.value = chip.dataset.query;
    sendQuery();
  });
});

// ── Message rendering ─────────────────────────────────────────────────────────
function hideWelcome() {
  if (welcomeEl) welcomeEl.style.display = 'none';
}

function createMessage(role) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'U' : 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollToBottom();
  return bubble;
}

function createTypingIndicator() {
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  wrap.id = 'typingIndicator';

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = `<div class="typing"><span></span><span></span><span></span></div>`;

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function removeTypingIndicator() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

function renderSources(sources, container) {
  if (!sources || sources.length === 0) return;

  const wrapper = document.createElement('div');
  wrapper.className = 'sources';

  const label = document.createElement('div');
  label.className = 'sources-label';
  label.textContent = `Sources · ${sources.length}`;
  wrapper.appendChild(label);

  sources.forEach(src => {
    const chip = document.createElement('div');
    chip.className = 'source-chip';

    const name = src.source_file
      ? src.source_file.split(/[/\\]/).pop()
      : src.id;

    chip.innerHTML = `
      <div style="flex:1; overflow:hidden;">
        <div style="display:flex; align-items:center; gap:0.375rem;">
          <span class="source-chip-name" title="${src.source_file ?? ''}">${name}</span>
          <span class="source-chip-score">score ${src.score?.toFixed(3) ?? '—'}</span>
        </div>
        ${src.excerpt ? `<div class="source-chip-excerpt">${escapeHtml(src.excerpt)}</div>` : ''}
      </div>`;

    wrapper.appendChild(chip);
  });

  container.appendChild(wrapper);
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Core: Send Query via SSE ──────────────────────────────────────────────────
async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query || isLoading) return;

  isLoading = true;
  setInputState(true);
  hideWelcome();

  // Render user message
  const userBubble = createMessage('user');
  userBubble.textContent = query;

  // Reset input
  queryInput.value = '';
  queryInput.style.height = 'auto';

  // Show typing indicator
  createTypingIndicator();

  let sources = [];
  let assistantBubble = null;
  let fullText = '';

  try {
    const response = await fetch(`${API_BASE}/api/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_n: 20, top_k: 4, alpha: 0.7 }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE lines
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line in buffer

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        let event;
        try { event = JSON.parse(jsonStr); } catch { continue; }

        if (event.type === 'sources') {
          sources = event.content;
          // Remove typing indicator and start assistant bubble
          removeTypingIndicator();
          assistantBubble = createMessage('assistant');

        } else if (event.type === 'token') {
          // First token: remove typing if sources weren't sent yet
          if (!assistantBubble) {
            removeTypingIndicator();
            assistantBubble = createMessage('assistant');
          }
          fullText += event.content;
          assistantBubble.textContent = fullText;
          scrollToBottom();

        } else if (event.type === 'done') {
          if (!assistantBubble) {
            removeTypingIndicator();
            assistantBubble = createMessage('assistant');
            assistantBubble.textContent = fullText;
          }
          renderSources(sources, assistantBubble.parentElement);
          scrollToBottom();

        } else if (event.type === 'error') {
          removeTypingIndicator();
          const errBubble = createMessage('assistant');
          errBubble.innerHTML = `<div class="error-bubble">⚠ ${escapeHtml(event.content)}</div>`;
        }
      }
    }

  } catch (err) {
    removeTypingIndicator();
    const errBubble = createMessage('assistant');
    errBubble.innerHTML = `<div class="error-bubble">⚠ ${escapeHtml(err.message)}</div>`;
  } finally {
    isLoading = false;
    setInputState(false);
    queryInput.focus();
  }
}

function setInputState(loading) {
  queryInput.disabled = loading;
  sendBtn.disabled = loading;
}

function clearChat() {
  messagesEl.innerHTML = '';
  // Re-insert welcome
  const welcome = document.createElement('div');
  welcome.className = 'welcome';
  welcome.id = 'welcome';
  welcome.innerHTML = `
    <div class="welcome-icon">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
      </svg>
    </div>
    <h1 class="welcome-title">What would you like to know?</h1>
    <p class="welcome-subtitle">Ask questions across your indexed document corpus using hybrid semantic + keyword search.</p>
    <div class="suggestions">
      <button class="suggestion-chip" data-query="Summarize the key topics covered in the available documents.">Summarize key topics</button>
      <button class="suggestion-chip" data-query="What are the main risks and challenges mentioned in the documents?">Main risks &amp; challenges</button>
      <button class="suggestion-chip" data-query="What financial metrics or KPIs are discussed?">Financial metrics &amp; KPIs</button>
      <button class="suggestion-chip" data-query="List the key findings and recommendations from the documents.">Key findings</button>
    </div>`;

  messagesEl.appendChild(welcome);

  // Re-attach chip listeners
  welcome.querySelectorAll('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      queryInput.value = chip.dataset.query;
      sendQuery();
    });
  });
}

// ── Ingest ────────────────────────────────────────────────────────────────────
function addLogLine(text, type = 'info') {
  ingestLog.style.display = 'block';
  const line = document.createElement('div');
  line.className = `log-line-${type}`;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
  logContent.appendChild(line);
  logContent.scrollTop = logContent.scrollHeight;
}

ingestBtn.addEventListener('click', async () => {
  const dataDir = dataDirInput.value.trim() || 'Data/Input';

  ingestBtn.disabled = true;
  ingestBtn.innerHTML = `
    <svg class="spin" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
    </svg>
    Running...`;

  addLogLine(`Starting ingestion from: ${dataDir}`, 'info');

  try {
    const res = await fetch(`${API_BASE}/api/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data_dir: dataDir }),
    });

    const data = await res.json();

    if (res.ok) {
      addLogLine(data.message ?? 'Ingestion complete.', 'ok');
    } else {
      addLogLine(`Error: ${data.detail ?? 'Unknown error'}`, 'err');
    }
  } catch (err) {
    addLogLine(`Network error: ${err.message}`, 'err');
  } finally {
    ingestBtn.disabled = false;
    ingestBtn.innerHTML = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>
        <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
      </svg>
      Start Ingestion`;
  }
});

clearLog.addEventListener('click', () => {
  logContent.innerHTML = '';
  ingestLog.style.display = 'none';
});

// ── Status ────────────────────────────────────────────────────────────────────
async function loadStatus() {
  statusGrid.innerHTML = `
    <div class="status-card skeleton"></div>
    <div class="status-card skeleton"></div>
    <div class="status-card skeleton"></div>
    <div class="status-card skeleton"></div>`;

  try {
    const res = await fetch(`${API_BASE}/api/status`);
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail ?? 'Failed');

    const embModel = (data.embedding_model ?? '').split('/').pop();
    const llmModel = data.llm_model ?? '—';
    const totalVec = (data.total_vectors ?? 0).toLocaleString();
    const indexName = data.index_name ?? '—';

    statusGrid.innerHTML = `
      <div class="status-card">
        <div class="status-card-label">Total Vectors</div>
        <div class="status-card-value">${totalVec}</div>
        <div class="status-card-sub">Pinecone Index</div>
      </div>
      <div class="status-card">
        <div class="status-card-label">Index Name</div>
        <div class="status-card-value" style="font-size:1rem;">${indexName}</div>
        <div class="status-card-sub">Hybrid (dense + sparse)</div>
      </div>
      <div class="status-card">
        <div class="status-card-label">Embedding Model</div>
        <div class="status-card-value" style="font-size:0.875rem;">${embModel}</div>
        <div class="status-card-sub">Sentence Transformers</div>
      </div>
      <div class="status-card">
        <div class="status-card-label">LLM Model</div>
        <div class="status-card-value" style="font-size:0.875rem;">${llmModel}</div>
        <div class="status-card-sub">Ollama (local)</div>
      </div>`;

  } catch (err) {
    statusGrid.innerHTML = `
      <div class="status-card" style="grid-column:1/-1;">
        <div class="error-bubble">⚠ Could not load status: ${escapeHtml(err.message)}</div>
      </div>`;
  }
}

refreshStatus.addEventListener('click', loadStatus);

// ── Documents ─────────────────────────────────────────────────────────────────
let activeDocFilename = null;

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function loadDocuments() {
  docsFileList.innerHTML = '<div class="docs-empty">Loading...</div>';

  try {
    const res = await fetch(`${API_BASE}/api/documents`);
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail ?? 'Failed');

    const docs = data.documents ?? [];
    if (docs.length === 0) {
      docsFileList.innerHTML = '<div class="docs-empty">No documents found.<br>Run ingestion first.</div>';
      return;
    }

    docsFileList.innerHTML = '';
    docs.forEach(doc => {
      const item = document.createElement('div');
      item.className = 'docs-file-item';
      item.dataset.filename = doc.filename;

      item.innerHTML = `
        <div class="docs-file-name" title="${doc.filename}">${doc.name}</div>
        <div class="docs-file-meta">
          <span class="docs-badge">${formatBytes(doc.size_bytes)}</span>
          ${doc.chunks > 0 ? `<span class="docs-badge chunks">${doc.chunks} chunks</span>` : ''}
        </div>`;

      item.addEventListener('click', () => openDocument(doc, item));
      docsFileList.appendChild(item);
    });

  } catch (err) {
    docsFileList.innerHTML = `<div class="docs-empty">Error: ${escapeHtml(err.message)}</div>`;
  }
}

async function openDocument(doc, itemEl) {
  // Update active state in file list
  document.querySelectorAll('.docs-file-item').forEach(el => el.classList.remove('active'));
  itemEl.classList.add('active');
  activeDocFilename = doc.filename;

  // Show loading state in viewer
  docsPlaceholder.style.display = 'none';
  docsContentWrap.style.display = 'flex';
  docsContentHeader.innerHTML = `
    <span class="docs-content-title">${doc.name}</span>
    <span class="docs-badge">${formatBytes(doc.size_bytes)}</span>
    ${doc.chunks > 0 ? `<span class="docs-badge chunks">${doc.chunks} chunks</span>` : ''}`;
  docsContentBody.innerHTML = '<div style="color:var(--text-dim);font-size:0.8rem;">Loading content...</div>';

  try {
    const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(doc.filename)}`);
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail ?? 'Failed');

    // Render markdown-like content
    docsContentBody.innerHTML = renderMarkdown(data.content);

    // Update header with fresh chunk count
    docsContentHeader.innerHTML = `
      <span class="docs-content-title">${data.name}</span>
      <span class="docs-badge">${formatBytes(data.size_bytes)}</span>
      ${data.chunks > 0 ? `<span class="docs-badge chunks">${data.chunks} chunks</span>` : ''}
      <button class="primary-btn" style="margin-left:auto;padding:0.35rem 0.75rem;font-size:0.75rem;" onclick="queryDoc('${data.name}')">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        Ask about this doc
      </button>`;

  } catch (err) {
    docsContentBody.innerHTML = `<div class="error-bubble">Failed to load: ${escapeHtml(err.message)}</div>`;
  }
}

function renderMarkdown(text) {
  // Lightweight markdown renderer — headings, bold, horizontal rules, code blocks
  const lines = text.split('\n');
  let html = '';
  let inCode = false;

  for (const raw of lines) {
    const line = raw;

    if (line.startsWith('```')) {
      if (!inCode) {
        html += '<code class="md-code">';
        inCode = true;
      } else {
        html += '</code>';
        inCode = false;
      }
      continue;
    }

    if (inCode) {
      html += escapeHtml(line) + '\n';
      continue;
    }

    if (/^### /.test(line)) {
      html += `<div class="md-h3">${escapeHtml(line.slice(4))}</div>`;
    } else if (/^## /.test(line)) {
      html += `<div class="md-h2">${escapeHtml(line.slice(3))}</div>`;
    } else if (/^# /.test(line)) {
      html += `<div class="md-h1">${escapeHtml(line.slice(2))}</div>`;
    } else if (/^---+$/.test(line.trim())) {
      html += '<hr class="md-hr" />';
    } else {
      // Inline bold: **text**
      const inlined = escapeHtml(line).replace(
        /\*\*(.+?)\*\*/g,
        '<span class="md-bold">$1</span>'
      );
      html += inlined + '\n';
    }
  }

  return html;
}

function queryDoc(docName) {
  // Switch to Chat and pre-fill a query about this document
  showView(viewChat);
  navChat.classList.add('active');
  queryInput.value = `Summarize the key points from "${docName}".`;
  queryInput.focus();
  queryInput.style.height = 'auto';
  queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
}

refreshDocs.addEventListener('click', loadDocuments);

// ── Boot ──────────────────────────────────────────────────────────────────────
checkHealth();
