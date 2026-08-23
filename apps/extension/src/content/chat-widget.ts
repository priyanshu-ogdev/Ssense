// apps/extension/src/content/chat-widget.ts
//
// Floating chatbot bubble injected on every page. Renders inside a Shadow
// DOM root so the host page's CSS can never bleed into the widget (and
// vice versa) — critical for a script that runs on arbitrary third-party
// sites. Talks to the background service worker only; never calls the SLM
// server directly (keeps API keys/HMAC secret out of the page context
// entirely, since content scripts share the page's JS realm with any other
// script that page chooses to load).

export {}; // marks this file as an ES module so `declare global` below is valid

declare global {
  interface Window {
    __ssenseChatWidgetLoaded?: boolean;
  }
}

// Guard the whole file against re-injection into an already-loaded page
// realm (see the identical guard + comment in dark-pattern-blocker.ts for
// why this happens and what it fixes).
if (!window.__ssenseChatWidgetLoaded) {
window.__ssenseChatWidgetLoaded = true;

console.log('[Ssense] Chat widget injected.');

type Msg = { role: 'user' | 'ai'; text: string; pending?: boolean };

const WIDGET_CSS = `
  * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  .wrap { position: relative; }
  .bubble {
    width: 48px; height: 48px; border-radius: 50%; border: none; cursor: pointer;
    background: linear-gradient(135deg, #06B6D4, #3B82F6);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    margin: 16px; transition: transform 0.15s ease;
  }
  .bubble:hover { transform: scale(1.06); }
  .bubble.open { transform: scale(0.94); }
  .bubble[hidden] { display: none; }
  .panel {
    position: absolute; bottom: 76px; right: 16px; width: 320px; height: 420px;
    background: #09090B; border: 1px solid rgba(255,255,255,0.1); border-radius: 14px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5); display: flex; flex-direction: column; overflow: hidden;
  }
  .panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.08);
  }
  .brand { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .brand-icon {
    width: 22px; height: 22px; border-radius: 6px; flex-shrink: 0;
    background: linear-gradient(135deg, #06B6D4, #3B82F6);
    display: flex; align-items: center; justify-content: center;
  }
  .brand-text { min-width: 0; }
  .brand-name { font-size: 12px; font-weight: 700; color: #F4F4F5; line-height: 1.2; }
  .header-actions { display: flex; align-items: center; gap: 2px; flex-shrink: 0; }
  .expand { background: none; border: none; color: #A1A1AA; cursor: pointer; font-size: 13px; padding: 2px 6px; border-radius: 4px; }
  .expand:hover { color: #06B6D4; background: rgba(6,182,212,0.1); }
  .domain {
    font-size: 12px; font-weight: 600; color: #F4F4F5;
    font-family: ui-monospace, 'SF Mono', Menlo, monospace;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .close { background: none; border: none; color: #A1A1AA; cursor: pointer; font-size: 13px; padding: 2px 6px; }
  .messages { flex: 1; overflow-y: auto; padding: 10px 12px; display: flex; flex-direction: column; gap: 8px; }
  .empty-note { font-size: 11.5px; color: #A1A1AA; line-height: 1.5; padding: 8px 2px; }
  .empty-note.error { color: #F87171; }
  .msg { display: flex; }
  .msg.user { justify-content: flex-end; }
  .msg.ai { justify-content: flex-start; }
  .bubble-text {
    max-width: 82%; font-size: 12.5px; line-height: 1.45; padding: 8px 10px; border-radius: 10px;
    white-space: pre-wrap; word-break: break-word;
  }
  .msg.user .bubble-text { background: #06B6D4; color: #09090B; font-weight: 500; }
  .msg.ai .bubble-text { background: rgba(255,255,255,0.06); color: #F4F4F5; }
  .bubble-text.pending { opacity: 0.6; font-style: italic; }
  .settings-link {
    align-self: flex-start; margin-top: 4px; background: rgba(6,182,212,0.12);
    border: 1px solid rgba(6,182,212,0.3); color: #06B6D4; font-size: 11px; font-weight: 600;
    padding: 5px 10px; border-radius: 6px; cursor: pointer;
  }
  .composer { display: flex; gap: 6px; padding: 10px; border-top: 1px solid rgba(255,255,255,0.08); }
  .composer input {
    flex: 1; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; color: #F4F4F5; font-size: 12.5px; padding: 8px 10px; outline: none;
  }
  .composer input:disabled { opacity: 0.5; }
  .composer button {
    background: #06B6D4; border: none; border-radius: 8px; color: #09090B;
    font-size: 14px; width: 34px; cursor: pointer; flex-shrink: 0;
  }
  .composer button:disabled { opacity: 0.5; cursor: default; }
`;

const DOMAIN = window.location.hostname;
if (DOMAIN) {
  initWidget();
}


function initWidget() {
  const host = document.createElement('div');
  host.id = 'ssense-chat-widget-host';
  // Beat host-page z-index wars; stay out of the page's own layout flow.
  host.style.cssText = 'all: initial; position: fixed; bottom: 0; right: 0; z-index: 2147483647;';
  document.documentElement.appendChild(host);

  const shadow = host.attachShadow({ mode: 'open' });
  shadow.innerHTML = `
    <style>${WIDGET_CSS}</style>
    <div class="wrap">
      <button class="bubble" id="toggle" title="Ask Ssense about ${escapeHtml(DOMAIN)}">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
      </button>
      <div class="panel" id="panel" hidden>
        <div class="panel-header">
          <div class="brand">
            <div class="brand-icon">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            </div>
            <div class="brand-text">
              <div class="brand-name">Ssense</div>
              <div class="domain">${escapeHtml(DOMAIN)}</div>
            </div>
          </div>
          <div class="header-actions">
            <button class="expand" id="expand" title="Open full report in Ssense">⤢</button>
            <button class="close" id="close" title="Close">✕</button>
          </div>
        </div>
        <div class="messages" id="messages"></div>
        <div class="composer">
          <input id="input" type="text" placeholder="Ask about this site's privacy policy…" autocomplete="off" />
          <button id="send" title="Send">➤</button>
        </div>
      </div>
    </div>
  `;

  const $ = <T extends Element>(sel: string) => shadow.querySelector(sel) as T;
  const toggleBtn = $('#toggle') as HTMLButtonElement;
  const closeBtn = $('#close') as HTMLButtonElement;
  const expandBtn = $('#expand') as HTMLButtonElement;
  const panel = $('#panel') as HTMLDivElement;
  const messagesEl = $('#messages') as HTMLDivElement;
  const input = $('#input') as HTMLInputElement;
  const sendBtn = $('#send') as HTMLButtonElement;

  expandBtn.addEventListener('click', () => {
    // Hands off to the real side panel — the fuller surface with history,
    // full audit reports, and settings all in one place — instead of the
    // bubble trying to duplicate that UI in a cramped 320px popup.
    chrome.runtime.sendMessage({ type: 'OPEN_SIDE_PANEL' });
  });

  let opened = false;
  let historyLoaded = false;
  let sending = false;

  function escapeHtml(s: string): string {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

  function renderMessages(msgs: Msg[]) {
    messagesEl.innerHTML = msgs
      .map(
        (m) => `
        <div class="msg ${m.role}">
          <div class="bubble-text ${m.pending ? 'pending' : ''}">${escapeHtml(m.text)}</div>
        </div>`
      )
      .join('');
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  let localMessages: Msg[] = [];

  async function loadHistory() {
    if (historyLoaded) return;
    historyLoaded = true;
    messagesEl.innerHTML = `<div class="empty-note">Loading conversation…</div>`;
    try {
      const res = await chrome.runtime.sendMessage({ type: 'GET_CHAT_HISTORY', domain: DOMAIN });
      const msgs: Msg[] = (res?.messages || []).map((m: any) => ({ role: m.role, text: m.text }));
      localMessages = msgs;
      if (msgs.length === 0) {
        messagesEl.innerHTML = `<div class="empty-note">Ask anything about ${escapeHtml(DOMAIN)}'s privacy practices — answers are grounded in the audited policy text, not a generic guess.</div>`;
      } else {
        renderMessages(localMessages);
      }
    } catch {
      messagesEl.innerHTML = `<div class="empty-note error">Could not load conversation history.</div>`;
    }
  }

  async function handleSend() {
    const text = input.value.trim();
    if (!text || sending) return;
    sending = true;
    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;

    localMessages.push({ role: 'user', text });
    localMessages.push({ role: 'ai', text: 'Thinking…', pending: true });
    renderMessages(localMessages);

    try {
      const config = await chrome.runtime.sendMessage({ type: 'GET_ENGINE_CONFIG' });
      // configured only reflects Cloud credentials (API key + HMAC secret) — it says
      // nothing about Offline Mode, which needs neither. A correctly-set-up offline-only
      // install (no Cloud key, Offline Mode on, models downloaded) was being wrongly
      // blocked here as "not configured" just because the Cloud fields were empty.
      if (!config?.configured && !config?.offlineMode) {
        localMessages.pop();
        localMessages.push({ role: 'ai', text: 'Ssense is not configured yet. Click below to open settings.' });
        renderMessages(localMessages);
        appendSettingsLink();
        return;
      }

      const res = await chrome.runtime.sendMessage({ type: 'CHAT', domain: DOMAIN, userPrompt: text });
      localMessages.pop(); // remove "Thinking…"
      if (res?.success && res.message) {
        localMessages.push({ role: 'ai', text: res.message });
      } else {
        localMessages.push({ role: 'ai', text: res?.error || 'Something went wrong reaching the audit service.' });
      }
      renderMessages(localMessages);
    } catch (err: any) {
      localMessages.pop();
      localMessages.push({ role: 'ai', text: err?.message || 'Could not reach the extension background service.' });
      renderMessages(localMessages);
    } finally {
      sending = false;
      input.disabled = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function appendSettingsLink() {
    const link = document.createElement('button');
    link.className = 'settings-link';
    link.textContent = 'Open Settings';
    link.onclick = () => chrome.runtime.sendMessage({ type: 'OPEN_OPTIONS_PAGE' });
    messagesEl.appendChild(link);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  toggleBtn.addEventListener('click', () => {
    opened = !opened;
    panel.hidden = !opened;
    toggleBtn.hidden = opened;
    toggleBtn.classList.toggle('open', opened);
    if (opened) {
      loadHistory();
      setTimeout(() => input.focus(), 50);
    }
  });
  closeBtn.addEventListener('click', () => {
    opened = false;
    panel.hidden = true;
    toggleBtn.hidden = false;
    toggleBtn.classList.remove('open');
  });
  sendBtn.addEventListener('click', handleSend);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') handleSend(); });
}

} // end __ssenseChatWidgetLoaded guard
