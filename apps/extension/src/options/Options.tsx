import React, { useEffect, useState } from 'react';

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

export default function Options() {
  const [serverUrl, setServerUrl] = useState('http://localhost:8080');
  const [apiKey, setApiKey] = useState('');
  const [hmacSecret, setHmacSecret] = useState('');
  const [showSecrets, setShowSecrets] = useState(false);
  const [saveState, setSaveState] = useState<SavedState>('idle');
  const [testState, setTestState] = useState<TestState>('idle');
  const [testDetail, setTestDetail] = useState<string>('');

  useEffect(() => {
    chrome.storage.local.get(STORAGE_KEYS, (data) => {
      if (data.ssense_server_url) setServerUrl(data.ssense_server_url);
      if (data.ssense_api_key) setApiKey(data.ssense_api_key);
      if (data.ssense_hmac_secret) setHmacSecret(data.ssense_hmac_secret);
    });
  }, []);

  const isValid = serverUrl.trim().length > 0 && apiKey.trim().length > 0 && hmacSecret.trim().length > 0;

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

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <div style={styles.header}>
          <div style={styles.headerIcon}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          </div>
          <div>
            <div style={styles.title}>Ssense Settings</div>
            <div style={styles.subtitle}>Connect to your SLM server</div>
          </div>
        </div>

        <div style={styles.field}>
          <label style={styles.label}>Server URL</label>
          <input
            style={styles.input}
            type="text"
            value={serverUrl}
            onChange={(e) => setServerUrl(e.target.value)}
            placeholder="http://localhost:8080 (dev) or https://yourdomain.example.com (prod, via Nginx)"
            spellCheck={false}
          />
        </div>

        <div style={styles.field}>
          <label style={styles.label}>API Key</label>
          <input
            style={styles.input}
            type={showSecrets ? 'text' : 'password'}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Paste your SSENSE_API_KEYS value"
            autoComplete="off"
            spellCheck={false}
          />
        </div>

        <div style={styles.field}>
          <label style={styles.label}>HMAC Secret</label>
          <input
            style={styles.input}
            type={showSecrets ? 'text' : 'password'}
            value={hmacSecret}
            onChange={(e) => setHmacSecret(e.target.value)}
            placeholder="Paste your SSENSE_HMAC_SECRET value"
            autoComplete="off"
            spellCheck={false}
          />
        </div>

        <label style={styles.checkboxRow}>
          <input type="checkbox" checked={showSecrets} onChange={(e) => setShowSecrets(e.target.checked)} />
          Show values
        </label>

        <div style={styles.buttonRow}>
          <button style={{ ...styles.button, ...styles.buttonPrimary }} disabled={!isValid} onClick={handleSave}>
            {saveState === 'saving' ? 'Saving…' : saveState === 'saved' ? 'Saved ✓' : 'Save'}
          </button>
          <button style={{ ...styles.button, ...styles.buttonSecondary }} disabled={!isValid} onClick={handleTest}>
            {testState === 'testing' ? 'Testing…' : 'Test Connection'}
          </button>
          <button style={{ ...styles.button, ...styles.buttonGhost }} onClick={handleClear}>
            Clear
          </button>
        </div>

        {testState === 'ok' && <div style={styles.statusOk}>✅ {testDetail}</div>}
        {testState === 'fail' && <div style={styles.statusFail}>⚠️ {testDetail}</div>}
        {!isValid && (
          <div style={styles.hint}>
            All three fields are required. These must match the SSENSE_API_KEYS / SSENSE_HMAC_SECRET
            configured on the server's own <code>.env</code> — there is no default/dev key anymore.
          </div>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: '100vh',
    background: '#09090B',
    color: '#F4F4F5',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    display: 'flex',
    justifyContent: 'center',
    padding: '48px 16px',
  },
  card: {
    width: '100%',
    maxWidth: 420,
    background: 'rgba(255,255,255,0.03)',
    border: '1px solid rgba(255,255,255,0.08)',
    borderRadius: 12,
    padding: 24,
  },
  header: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 },
  headerIcon: {
    width: 36, height: 36, borderRadius: 8,
    background: 'linear-gradient(135deg, #06B6D4, #3B82F6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
  },
  title: { fontSize: 16, fontWeight: 700 },
  subtitle: { fontSize: 12, color: '#A1A1AA', marginTop: 2 },
  field: { marginBottom: 16 },
  label: { display: 'block', fontSize: 12, fontWeight: 600, color: '#A1A1AA', marginBottom: 6 },
  input: {
    width: '100%', boxSizing: 'border-box', background: 'rgba(255,255,255,0.05)',
    border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#F4F4F5',
    fontSize: 13, padding: '9px 10px', outline: 'none', fontFamily: 'monospace',
  },
  checkboxRow: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#A1A1AA', marginBottom: 20, cursor: 'pointer' },
  buttonRow: { display: 'flex', gap: 8 },
  button: {
    flex: 1, borderRadius: 8, padding: '9px 12px', fontSize: 13, fontWeight: 600,
    cursor: 'pointer', border: '1px solid transparent',
  },
  buttonPrimary: { background: '#06B6D4', color: '#09090B' },
  buttonSecondary: { background: 'rgba(255,255,255,0.06)', color: '#F4F4F5', borderColor: 'rgba(255,255,255,0.1)' },
  buttonGhost: { background: 'transparent', color: '#A1A1AA', borderColor: 'rgba(255,255,255,0.1)', flex: 0.6 },
  statusOk: { marginTop: 14, fontSize: 12, color: '#4ADE80' },
  statusFail: { marginTop: 14, fontSize: 12, color: '#F87171' },
  hint: { marginTop: 14, fontSize: 11, color: '#71717A', lineHeight: 1.5 },
};
