import { useEffect, useMemo, useState } from 'react';

// ═══════════════════════════════════════════════════════════════
// Ssense Options — configures the ONLY backend this extension talks
// to: the SLM server. There is no local-daemon mode anymore, so this
// page just needs three fields: server URL, API key, HMAC secret.
//
// These credentials must match what the SLM server was booted with
// (SSENSE_API_KEYS / SSENSE_HMAC_SECRET in its .env). Nothing here is
// pre-filled with a default — api-client.ts refuses to send requests
// until both fields are non-empty (see `configured` in getServerConfig).
// ═══════════════════════════════════════════════════════════════

type SavedState = 'idle' | 'saving' | 'saved' | 'error';
type TestState = 'idle' | 'testing' | 'ok' | 'fail';

const STORAGE_KEYS = ['ssense_server_url', 'ssense_api_key', 'ssense_hmac_secret'] as const;

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '0.0.0.0', '::1']);

function classifyUrl(raw: string): { kind: 'local' | 'remote' | 'invalid' | 'empty'; host?: string; secure?: boolean } {
  const trimmed = raw.trim();
  if (!trimmed) return { kind: 'empty' };
  try {
    const u = new URL(trimmed);
    const host = u.hostname.toLowerCase();
    const isPrivate =
      LOCAL_HOSTS.has(host) ||
      /^10\./.test(host) ||
      /^192\.168\./.test(host) ||
      /^172\.(1[6-9]|2\d|3[0-1])\./.test(host) ||
      host.endsWith('.local');
    return { kind: isPrivate ? 'local' : 'remote', host: u.host, secure: u.protocol === 'https:' };
  } catch {
    return { kind: 'invalid' };
  }
}

export default function Options() {
  const [serverUrl, setServerUrl] = useState('http://localhost:8080');
  const [apiKey, setApiKey] = useState('');
  const [hmacSecret, setHmacSecret] = useState('');
  const [showSecrets, setShowSecrets] = useState(false);
  const [saveState, setSaveState] = useState<SavedState>('idle');
  const [testState, setTestState] = useState<TestState>('idle');
  const [testDetail, setTestDetail] = useState<string>('');

  useEffect(() => {
    const styleTag = document.createElement('style');
    styleTag.innerHTML = OPTIONS_CSS;
    document.head.appendChild(styleTag);
    return () => { document.head.removeChild(styleTag); };
  }, []);

  useEffect(() => {
    chrome.storage.local.get(STORAGE_KEYS, (data) => {
      if (data.ssense_server_url) setServerUrl(data.ssense_server_url);
      if (data.ssense_api_key) setApiKey(data.ssense_api_key);
      if (data.ssense_hmac_secret) setHmacSecret(data.ssense_hmac_secret);
    });
  }, []);

  const urlInfo = useMemo(() => classifyUrl(serverUrl), [serverUrl]);
  const isValid = serverUrl.trim().length > 0 && apiKey.trim().length > 0 && hmacSecret.trim().length > 0 && urlInfo.kind !== 'invalid';

  const handleSave = async () => {
    setSaveState('saving');
    try {
      await chrome.storage.local.set({
        ssense_server_url: serverUrl.trim().replace(/\/$/, ''),
        ssense_api_key: apiKey.trim(),
        ssense_hmac_secret: hmacSecret.trim(),
      });
      setSaveState('saved');
      setTestState('idle');
      setTimeout(() => setSaveState('idle'), 2000);
    } catch {
      setSaveState('error');
    }
  };

  const handleClear = async () => {
    await chrome.storage.local.remove(STORAGE_KEYS as unknown as string[]);
    setApiKey('');
    setHmacSecret('');
    setServerUrl('http://localhost:8080');
    setSaveState('idle');
    setTestState('idle');
  };

  const handleTest = async () => {
    setTestState('testing');
    setTestDetail('');
    try {
      // Save first so the service worker reads the values currently on screen,
      // not whatever was previously persisted.
      await chrome.storage.local.set({
        ssense_server_url: serverUrl.trim().replace(/\/$/, ''),
        ssense_api_key: apiKey.trim(),
        ssense_hmac_secret: hmacSecret.trim(),
      });
      const res = await chrome.runtime.sendMessage({ type: 'HEALTH_CHECK' });
      if (res?.success) {
        setTestState('ok');
        setTestDetail(`Connected — ${res.avgTokensPerSecond ?? '?'} tok/s, model loaded: ${res.modelLoaded ?? '?'}`);
      } else {
        setTestState('fail');
        setTestDetail(res?.error || 'Server responded but reported unhealthy.');
      }
    } catch (err: any) {
      setTestState('fail');
      setTestDetail(err?.message || 'Could not reach the extension background service.');
    }
  };

  const envBadge = () => {
    if (urlInfo.kind === 'empty') return null;
    if (urlInfo.kind === 'invalid') {
      return <span className="opt-badge opt-badge--invalid"><span className="opt-badge-dot" />Invalid URL</span>;
    }
    if (urlInfo.kind === 'local') {
      return <span className="opt-badge opt-badge--local"><span className="opt-badge-dot" />Local · Dev</span>;
    }
    return (
      <span className={`opt-badge ${urlInfo.secure ? 'opt-badge--remote' : 'opt-badge--warn'}`}>
        <span className="opt-badge-dot" />
        {urlInfo.secure ? 'Online · Production' : 'Online · Insecure (HTTP)'}
      </span>
    );
  };

  return (
    <div className="opt-page">
      <div className="opt-shell">
        <header className="opt-masthead">
          <div className="opt-masthead-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          </div>
          <div>
            <div className="opt-title">Ssense Settings</div>
            <div className="opt-subtitle">Server connection &amp; credentials</div>
          </div>
        </header>

        <div className="opt-card">
          <div className="opt-card-head">
            <div className="opt-card-head-text">
              <div className="opt-eyebrow">Connection</div>
              <div className="opt-card-title">SLM Server</div>
            </div>
            {envBadge()}
          </div>

          <div className="opt-field">
            <label className="opt-label">Server URL</label>
            <input
              className={`opt-input opt-input--mono ${urlInfo.kind === 'invalid' ? 'opt-input--error' : ''}`}
              type="text"
              value={serverUrl}
              onChange={(e) => setServerUrl(e.target.value)}
              placeholder="http://localhost:8080"
              spellCheck={false}
            />
            <div className="opt-help">
              Use <code>http://localhost:8080</code> for local development, or your production domain
              (e.g. <code>https://api.yourdomain.example.com</code>) served behind Nginx/TLS.
            </div>
          </div>

          <div className="opt-divider" />

          <div className="opt-field">
            <label className="opt-label">API Key</label>
            <input
              className="opt-input opt-input--mono"
              type={showSecrets ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Paste your SSENSE_API_KEYS value"
              autoComplete="off"
              spellCheck={false}
            />
          </div>

          <div className="opt-field opt-field--last">
            <label className="opt-label">HMAC Secret</label>
            <input
              className="opt-input opt-input--mono"
              type={showSecrets ? 'text' : 'password'}
              value={hmacSecret}
              onChange={(e) => setHmacSecret(e.target.value)}
              placeholder="Paste your SSENSE_HMAC_SECRET value"
              autoComplete="off"
              spellCheck={false}
            />
          </div>

          <label className="opt-checkbox-row">
            <input type="checkbox" checked={showSecrets} onChange={(e) => setShowSecrets(e.target.checked)} />
            <span>Show values</span>
          </label>

          <div className="opt-button-row">
            <button className="opt-btn opt-btn--primary" disabled={!isValid || saveState === 'saving'} onClick={handleSave}>
              {saveState === 'saving' ? 'Saving…' : saveState === 'saved' ? '✓ Saved' : 'Save'}
            </button>
            <button className="opt-btn opt-btn--secondary" disabled={!isValid || testState === 'testing'} onClick={handleTest}>
              {testState === 'testing' ? 'Testing…' : 'Test Connection'}
            </button>
            <button className="opt-btn opt-btn--ghost" onClick={handleClear}>
              Clear
            </button>
          </div>

          {testState === 'ok' && (
            <div className="opt-status opt-status--ok">
              <span className="opt-status-icon">✓</span>
              <span>{testDetail}</span>
            </div>
          )}
          {testState === 'fail' && (
            <div className="opt-status opt-status--fail">
              <span className="opt-status-icon">!</span>
              <span>{testDetail}</span>
            </div>
          )}

          {!isValid && (
            <div className="opt-hint">
              All three fields are required. Key and secret must match <code>SSENSE_API_KEYS</code> /
              <code> SSENSE_HMAC_SECRET</code> configured on the server's own <code>.env</code> — there is
              no default or development key.
            </div>
          )}
        </div>

        <footer className="opt-footer">
          <span>SSENSE EDGE</span>
          <span className="opt-footer-dot">•</span>
          <span>Settings are stored locally in this browser profile</span>
        </footer>
      </div>
    </div>
  );
}

const OPTIONS_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --opt-bg-deep: #09090B;
    --opt-bg-surface: #131316;
    --opt-bg-elevated: rgba(255,255,255,0.035);
    --opt-border: rgba(255,255,255,0.08);
    --opt-border-strong: rgba(255,255,255,0.14);
    --opt-text-primary: #FAFAFA;
    --opt-text-secondary: #A1A1AA;
    --opt-text-muted: #71717A;
    --opt-cyan: #22D3EE;
    --opt-blue: #3B82F6;
    --opt-violet: #8B5CF6;
    --opt-green: #34D399;
    --opt-amber: #FBBF24;
    --opt-red: #FB7185;
    --opt-gradient: linear-gradient(135deg, var(--opt-cyan) 0%, var(--opt-violet) 100%);
  }

  * { box-sizing: border-box; }

  .opt-page {
    min-height: 100vh;
    background:
      radial-gradient(1200px 600px at 15% -10%, rgba(139,92,246,0.10), transparent 60%),
      radial-gradient(1000px 500px at 100% 0%, rgba(34,211,238,0.08), transparent 55%),
      var(--opt-bg-deep);
    color: var(--opt-text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
    display: flex;
    justify-content: center;
    padding: 56px 20px;
  }

  .opt-shell { width: 100%; max-width: 460px; }

  .opt-masthead { display: flex; align-items: center; gap: 13px; margin-bottom: 22px; }
  .opt-masthead-icon {
    width: 40px; height: 40px; border-radius: 10px; flex-shrink: 0;
    background: var(--opt-gradient);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 18px rgba(34,211,238,0.22);
  }
  .opt-title { font-size: 18px; font-weight: 800; letter-spacing: -0.01em; }
  .opt-subtitle { font-size: 12.5px; color: var(--opt-text-muted); margin-top: 2px; }

  .opt-card {
    background: var(--opt-bg-surface);
    border: 1px solid var(--opt-border);
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 20px 50px -20px rgba(0,0,0,0.55);
  }

  .opt-card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 20px; padding-bottom: 18px; border-bottom: 1px solid var(--opt-border); }
  .opt-eyebrow { font-size: 10px; font-weight: 700; letter-spacing: 0.09em; color: var(--opt-text-muted); text-transform: uppercase; margin-bottom: 4px; }
  .opt-card-title { font-size: 15px; font-weight: 700; }

  .opt-badge {
    flex-shrink: 0; display: inline-flex; align-items: center; gap: 6px;
    font-size: 10.5px; font-weight: 650; letter-spacing: 0.01em;
    padding: 5px 10px; border-radius: 20px; white-space: nowrap;
    border: 1px solid transparent;
  }
  .opt-badge-dot { width: 6px; height: 6px; border-radius: 50%; }
  .opt-badge--local { background: rgba(59,130,246,0.12); color: #93C5FD; border-color: rgba(59,130,246,0.28); }
  .opt-badge--local .opt-badge-dot { background: var(--opt-blue); box-shadow: 0 0 0 3px rgba(59,130,246,0.18); }
  .opt-badge--remote { background: rgba(52,211,153,0.12); color: #6EE7B7; border-color: rgba(52,211,153,0.28); }
  .opt-badge--remote .opt-badge-dot { background: var(--opt-green); box-shadow: 0 0 0 3px rgba(52,211,153,0.18); }
  .opt-badge--warn { background: rgba(251,191,36,0.12); color: #FCD34D; border-color: rgba(251,191,36,0.28); }
  .opt-badge--warn .opt-badge-dot { background: var(--opt-amber); box-shadow: 0 0 0 3px rgba(251,191,36,0.18); }
  .opt-badge--invalid { background: rgba(251,113,133,0.12); color: #FDA4AF; border-color: rgba(251,113,133,0.28); }
  .opt-badge--invalid .opt-badge-dot { background: var(--opt-red); box-shadow: 0 0 0 3px rgba(251,113,133,0.18); }

  .opt-field { margin-bottom: 18px; }
  .opt-field--last { margin-bottom: 4px; }
  .opt-label { display: block; font-size: 11.5px; font-weight: 650; color: var(--opt-text-secondary); margin-bottom: 7px; letter-spacing: 0.01em; }

  .opt-input {
    width: 100%; background: rgba(255,255,255,0.04);
    border: 1px solid var(--opt-border); border-radius: 9px;
    color: var(--opt-text-primary); font-size: 13px;
    padding: 10px 12px; outline: none;
    transition: border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
  }
  .opt-input:hover { border-color: var(--opt-border-strong); }
  .opt-input:focus { border-color: var(--opt-cyan); background: rgba(34,211,238,0.045); box-shadow: 0 0 0 3px rgba(34,211,238,0.12); }
  .opt-input--mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }
  .opt-input--error { border-color: var(--opt-red); }
  .opt-input--error:focus { box-shadow: 0 0 0 3px rgba(251,113,133,0.14); }

  .opt-help { font-size: 11px; color: var(--opt-text-muted); line-height: 1.55; margin-top: 8px; }
  .opt-help code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--opt-text-secondary); }

  .opt-divider { height: 1px; background: var(--opt-border); margin: 4px 0 20px; }

  .opt-checkbox-row {
    display: flex; align-items: center; gap: 8px; font-size: 12px;
    color: var(--opt-text-secondary); margin: 14px 0 22px; cursor: pointer; user-select: none;
  }
  .opt-checkbox-row input { width: 14px; height: 14px; accent-color: var(--opt-cyan); cursor: pointer; }

  .opt-button-row { display: flex; gap: 10px; }
  .opt-btn {
    border-radius: 9px; padding: 10px 14px; font-size: 12.5px; font-weight: 650;
    cursor: pointer; border: 1px solid transparent; font-family: inherit;
    transition: filter 0.15s ease, background 0.15s ease, opacity 0.15s ease, transform 0.1s ease, border-color 0.15s ease;
  }
  .opt-btn:active:not(:disabled) { transform: scale(0.98); }
  .opt-btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .opt-btn--primary { flex: 1.3; background: var(--opt-gradient); color: #0A0A0C; box-shadow: 0 6px 18px -6px rgba(34,211,238,0.4); }
  .opt-btn--primary:hover:not(:disabled) { filter: brightness(1.08); }
  .opt-btn--secondary { flex: 1.3; background: rgba(255,255,255,0.06); color: var(--opt-text-primary); border-color: var(--opt-border-strong); }
  .opt-btn--secondary:hover:not(:disabled) { background: rgba(255,255,255,0.1); }
  .opt-btn--ghost { flex: 0.8; background: transparent; color: var(--opt-text-secondary); border-color: var(--opt-border); }
  .opt-btn--ghost:hover { background: rgba(255,255,255,0.04); color: var(--opt-text-primary); }

  .opt-status {
    display: flex; align-items: flex-start; gap: 9px;
    margin-top: 16px; padding: 11px 13px; border-radius: 10px; font-size: 12px; line-height: 1.5;
  }
  .opt-status--ok { background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.22); color: #6EE7B7; }
  .opt-status--fail { background: rgba(251,113,133,0.08); border: 1px solid rgba(251,113,133,0.22); color: #FDA4AF; }
  .opt-status-icon {
    flex-shrink: 0; width: 16px; height: 16px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 800; background: rgba(255,255,255,0.1);
  }

  .opt-hint { margin-top: 16px; font-size: 11px; color: var(--opt-text-muted); line-height: 1.6; }
  .opt-hint code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--opt-text-secondary); }

  .opt-footer {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    margin-top: 20px; font-size: 10.5px; color: var(--opt-text-muted); letter-spacing: 0.02em;
  }
  .opt-footer-dot { opacity: 0.5; }
`;
