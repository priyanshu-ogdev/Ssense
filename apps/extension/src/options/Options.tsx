import { useEffect, useMemo, useState } from 'react';
import { LOCAL_SERVER_URL, ONLINE_SERVER_URL, DEFAULT_SERVER_MODE, type ServerMode } from '../config';

// ═══════════════════════════════════════════════════════════════
// Ssense Options — configures the ONLY backend this extension talks
// to: the SLM server. The endpoint is no longer a free-text field —
// it is always one of two fixed, known-good addresses defined in
// config.ts. The user picks a *mode*:
//
//   Auto    probe localhost first, fall back to the hosted server
//   Local   force LOCAL_SERVER_URL
//   Online  force ONLINE_SERVER_URL
//
// This removes an entire class of support tickets and phishing risk
// (typo'd or malicious "server URL" values) at the cost of zero
// flexibility a normal user actually needs. api-client.ts resolves
// the mode into a URL at request time — see resolveMode().
// ═══════════════════════════════════════════════════════════════

type SavedState = 'idle' | 'saving' | 'saved' | 'error';
type TestState = 'idle' | 'testing' | 'ok' | 'fail';
type ProbeState = 'checking' | 'reachable' | 'unreachable';

const STORAGE_KEYS = ['ssense_server_mode', 'ssense_api_key', 'ssense_hmac_secret'] as const;
const PROBE_TIMEOUT_MS = 1200;

async function probeLocal(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
    const res = await fetch(`${LOCAL_SERVER_URL}/health`, { method: 'GET', signal: controller.signal });
    clearTimeout(t);
    return res.ok;
  } catch {
    return false;
  }
}

const MODES: { id: ServerMode; label: string; hint: string }[] = [
  { id: 'auto', label: 'Auto', hint: 'Detect automatically' },
  { id: 'local', label: 'Local', hint: LOCAL_SERVER_URL },
  { id: 'online', label: 'Online', hint: ONLINE_SERVER_URL },
];

export default function Options() {
  const [mode, setMode] = useState<ServerMode>(DEFAULT_SERVER_MODE);
  const [apiKey, setApiKey] = useState('');
  const [hmacSecret, setHmacSecret] = useState('');
  const [showSecrets, setShowSecrets] = useState(false);
  const [saveState, setSaveState] = useState<SavedState>('idle');
  const [testState, setTestState] = useState<TestState>('idle');
  const [testDetail, setTestDetail] = useState<string>('');
  const [probeState, setProbeState] = useState<ProbeState>('checking');

  useEffect(() => {
    const styleTag = document.createElement('style');
    styleTag.innerHTML = OPTIONS_CSS;
    document.head.appendChild(styleTag);
    return () => { document.head.removeChild(styleTag); };
  }, []);

  useEffect(() => {
    chrome.storage.local.get(STORAGE_KEYS, (data) => {
      if (data.ssense_server_mode) setMode(data.ssense_server_mode);
      if (data.ssense_api_key) setApiKey(data.ssense_api_key);
      if (data.ssense_hmac_secret) setHmacSecret(data.ssense_hmac_secret);
    });
  }, []);

  // Re-probe the local server whenever the mode changes to "auto" (or on mount),
  // purely to drive the resolved-endpoint indicator — the actual runtime
  // resolution happens independently inside api-client.ts.
  useEffect(() => {
    if (mode !== 'auto') return;
    let cancelled = false;
    setProbeState('checking');
    probeLocal().then((reachable) => {
      if (!cancelled) setProbeState(reachable ? 'reachable' : 'unreachable');
    });
    return () => { cancelled = true; };
  }, [mode]);

  const resolvedUrl = useMemo(() => {
    if (mode === 'local') return LOCAL_SERVER_URL;
    if (mode === 'online') return ONLINE_SERVER_URL;
    return probeState === 'reachable' ? LOCAL_SERVER_URL : ONLINE_SERVER_URL;
  }, [mode, probeState]);

  const resolvedKind: 'local' | 'online' = resolvedUrl === LOCAL_SERVER_URL ? 'local' : 'online';

  const isValid = apiKey.trim().length > 0 && hmacSecret.trim().length > 0;

  const persist = async () => {
    await chrome.storage.local.set({
      ssense_server_mode: mode,
      ssense_api_key: apiKey.trim(),
      ssense_hmac_secret: hmacSecret.trim(),
    });
  };

  const handleSave = async () => {
    setSaveState('saving');
    try {
      await persist();
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
    setMode(DEFAULT_SERVER_MODE);
    setSaveState('idle');
    setTestState('idle');
  };

  const handleTest = async () => {
    setTestState('testing');
    setTestDetail('');
    try {
      // Save first so the service worker reads the mode currently on screen,
      // not whatever was previously persisted.
      await persist();
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

  return (
    <div className="opt-page">
      <div className="opt-shell">
        <header className="opt-masthead">
          <button
            type="button"
            className="opt-back-btn"
            onClick={() => { if (window.history.length > 1) window.history.back(); else window.close(); }}
            title="Back"
            aria-label="Back"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
          </button>
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
            <span className={`opt-badge opt-badge--${resolvedKind}`}>
              <span className="opt-badge-dot" />
              {resolvedKind === 'local' ? 'Local · Dev' : 'Online · Production'}
            </span>
          </div>

          <div className="opt-field">
            <label className="opt-label">Server</label>

            <div className="opt-segmented" role="radiogroup" aria-label="Server mode">
              {MODES.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  role="radio"
                  aria-checked={mode === m.id}
                  className={`opt-segment ${mode === m.id ? 'opt-segment--active' : ''}`}
                  onClick={() => setMode(m.id)}
                >
                  {m.label}
                </button>
              ))}
            </div>

            <div className="opt-resolved">
              <span className="opt-resolved-label">Resolved endpoint</span>
              <code className="opt-resolved-url">{resolvedUrl}</code>
              {mode === 'auto' && (
                <span className={`opt-probe opt-probe--${probeState}`}>
                  {probeState === 'checking' && 'Checking local server…'}
                  {probeState === 'reachable' && 'Local server detected'}
                  {probeState === 'unreachable' && 'Local server not found — using Online'}
                </span>
              )}
            </div>

            <div className="opt-help">
              The endpoint is fixed and cannot be typed in — pick <strong>Auto</strong> to
              detect the local server automatically, or force <strong>Local</strong> /
              <strong> Online</strong> explicitly.
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
              API key and HMAC secret are required. They must match <code>SSENSE_API_KEYS</code> /
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
  .opt-back-btn {
    flex-shrink: 0; width: 30px; height: 30px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    background: transparent; border: 1px solid var(--opt-border);
    color: var(--opt-text-secondary); cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
  }
  .opt-back-btn:hover { background: rgba(255,255,255,0.06); color: var(--opt-text-primary); border-color: var(--opt-border-strong); }
  .opt-back-btn:active { transform: scale(0.94); }
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
  .opt-badge--online { background: rgba(52,211,153,0.12); color: #6EE7B7; border-color: rgba(52,211,153,0.28); }
  .opt-badge--online .opt-badge-dot { background: var(--opt-green); box-shadow: 0 0 0 3px rgba(52,211,153,0.18); }

  .opt-field { margin-bottom: 18px; }
  .opt-field--last { margin-bottom: 4px; }
  .opt-label { display: block; font-size: 11.5px; font-weight: 650; color: var(--opt-text-secondary); margin-bottom: 7px; letter-spacing: 0.01em; }

  .opt-segmented {
    display: flex; gap: 4px; padding: 4px;
    background: rgba(255,255,255,0.035); border: 1px solid var(--opt-border);
    border-radius: 11px;
  }
  .opt-segment {
    flex: 1; border: none; border-radius: 8px; padding: 9px 10px;
    font-family: inherit; font-size: 12.5px; font-weight: 650;
    color: var(--opt-text-secondary); background: transparent; cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease, box-shadow 0.15s ease;
  }
  .opt-segment:hover:not(.opt-segment--active) { color: var(--opt-text-primary); background: rgba(255,255,255,0.04); }
  .opt-segment--active {
    color: #0A0A0C; background: var(--opt-gradient);
    box-shadow: 0 4px 14px -4px rgba(34,211,238,0.4);
  }

  .opt-resolved {
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
    margin-top: 12px; padding: 10px 12px;
    background: rgba(255,255,255,0.03); border: 1px solid var(--opt-border);
    border-radius: 9px;
  }
  .opt-resolved-label { font-size: 10px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--opt-text-muted); }
  .opt-resolved-url { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 12px; color: var(--opt-text-primary); }
  .opt-probe { font-size: 11px; color: var(--opt-text-muted); margin-left: auto; }
  .opt-probe--reachable { color: #6EE7B7; }
  .opt-probe--unreachable { color: var(--opt-text-muted); }

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

  .opt-help { font-size: 11px; color: var(--opt-text-muted); line-height: 1.55; margin-top: 10px; }
  .opt-help code, .opt-hint code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--opt-text-secondary); }
  .opt-help strong { color: var(--opt-text-secondary); font-weight: 650; }

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

  .opt-footer {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    margin-top: 20px; font-size: 10.5px; color: var(--opt-text-muted); letter-spacing: 0.02em;
  }
  .opt-footer-dot { opacity: 0.5; }
`;
