import React, { useEffect, useState } from 'react';

type Mode = 'cloud' | 'offline';
type DownloadState = 'idle' | 'downloading' | 'ready' | 'error';

interface Progress {
  file: string;
  pct: number;
  mbPerSec: number;
  updatedAt?: number;
}

const C = {
  bg: '#09090B',
  panel: '#111113',
  border: 'rgba(255,255,255,.09)',
  text: '#F4F4F5',
  muted: '#A1A1AA',
  faint: '#71717A',
  cyan: '#22D3EE',
  blue: '#3B82F6',
  green: '#4ADE80',
  amber: '#FBBF24',
  red: '#FB7185',
};

function requestId() {
  return crypto.randomUUID();
}

function pctText(n: number) {
  // -1 is the daemon's "indeterminate" signal (origin didn't send Content-Length),
  // not a literal negative percent.
  if (n < 0) return 'Downloading…';
  return `${Math.max(0, Math.min(100, n)).toFixed(0)}%`;
}

export default function Popup() {
  const [domain, setDomain] = useState('');
  const [mode, setMode] = useState<Mode>('cloud');
  const [downloadState, setDownloadState] = useState<DownloadState>('idle');
  const [progress, setProgress] = useState<Progress>({ file: '', pct: 0, mbPerSec: 0 });
  const [error, setError] = useState('');
  const [checking, setChecking] = useState(true);
  const [serviceStatus, setServiceStatus] = useState<'checking' | 'online' | 'offline' | 'not-configured'>('checking');
  const [lastError, setLastError] = useState('');

  useEffect(() => {
    let mounted = true;

    (async () => {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!mounted) return;
      try { setDomain(tab?.url?.startsWith('http') ? new URL(tab.url).hostname : ''); } catch { setDomain(''); }

      const data = await chrome.storage.local.get([
        'ssense_offline_mode',
        'ssense_download_progress',
        'ssense_offline_error',
      ]);
      if (!mounted) return;

      setMode(data.ssense_offline_mode ? 'offline' : 'cloud');
      if (data.ssense_offline_error) { setDownloadState('error'); setError(String(data.ssense_offline_error)); }
      if (data.ssense_download_progress) {
        const p = data.ssense_download_progress as Progress;
        setProgress(p);
        if (p.pct !== 0 && p.pct < 100) setDownloadState('downloading');
        if (p.pct >= 100) setDownloadState('ready');
      }
      setChecking(false);
      try {
        const health = await chrome.runtime.sendMessage({ type: 'HEALTH_CHECK', requestId: requestId() });
        if (data.ssense_offline_mode) setServiceStatus(health?.success ? 'online' : 'offline');
        else if (health?.errorKind === 'auth') setServiceStatus('not-configured');
        else setServiceStatus(health?.success ? 'online' : 'offline');
      } catch { setServiceStatus('offline'); }
    })();

    const listener = (message: any) => {
      if (message.type === 'OFFLINE_DOWNLOAD_PROGRESS') {
        setDownloadState('downloading');
        setProgress({
          file: message.file,
          pct: message.pct,
          mbPerSec: message.mbPerSec,
          updatedAt: Date.now(),
        });
      }
      if (message.type === 'OFFLINE_DOWNLOAD_ERROR') {
        setDownloadState('error');
        setError(message.error || 'Offline model download failed.');
        setLastError(message.error || 'Offline model download failed.');
      }
      if (message.type === 'OFFLINE_READY') {
        setMode('offline');
        setDownloadState('ready');
        setProgress({ file: '', pct: 100, mbPerSec: 0 });
      }
    };

    chrome.runtime.onMessage.addListener(listener);
    return () => {
      mounted = false;
      chrome.runtime.onMessage.removeListener(listener);
    };
  }, []);

  const enableOffline = async () => {
    setError('');
    setLastError('');
    setDownloadState('downloading');
    try {
      const response = await chrome.runtime.sendMessage({
        type: 'SET_OFFLINE_MODE',
        enabled: true,
        startDownload: true,
        requestId: requestId(),
      });

      if (response?.success && response?.offlineMode) {
        setMode('offline');
        setDownloadState('ready');
      } else if (response?.type === 'ERROR' || response?.success === false) {
        setDownloadState('error');
        setError(response?.error || 'Could not enable Offline Mode.');
      }
    } catch (err: any) {
      setDownloadState('error');
      setError(err?.message || 'The extension could not start Offline Mode.');
    }
  };

  const disableOffline = async () => {
    setError('');
    setLastError('');
    try {
      const response = await chrome.runtime.sendMessage({
        type: 'SET_OFFLINE_MODE',
        enabled: false,
      });
      if (response?.success) {
        setMode('cloud');
        setDownloadState('idle');
      } else {
        setError(response?.error || 'Could not switch to Cloud Mode.');
      }
    } catch (err: any) {
      setError(err?.message || 'The extension could not switch to Cloud Mode.');
    }
  };

  const openSidePanel = async (view: 'audit' | 'history' | 'privacy' = 'audit') => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    await chrome.storage.local.set({ ssense_sidepanel_view: view });
    if (tab?.id) await chrome.sidePanel.open({ tabId: tab.id });
    window.close();
  };

  const openFullReport = () => openSidePanel('audit');

  const openSettings = () => {
    chrome.runtime.openOptionsPage();
    window.close();
  };

  const offlineBusy = downloadState === 'downloading';

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <div style={styles.logo}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.4">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6-8 10-8 10" />
          </svg>
        </div>
        <div>
          <div style={styles.brand}>Ssense</div>
          <div style={styles.sub}>DPDP Compliance Shield</div>
        </div>
        <div style={styles.domain}>{domain || 'Browser page'}</div>
      </header>

      <main style={styles.body}>
        <section style={styles.modeCard}>
          <div style={styles.modeTop}>
            <div>
              <div style={styles.eyebrow}>AI ENGINE</div>
              <div style={styles.modeTitle}>
                {mode === 'offline' ? 'Private · Offline' : 'Cloud · Fast'}
              </div>
              <div style={styles.modeDescription}>
                {mode === 'offline'
                  ? 'Inference stays on this device. No policy text is sent to the cloud.'
                  : 'Fast server inference is enabled by default. Offline models are optional.'}
              </div>
            </div>

            <button
              aria-label={mode === 'offline' ? 'Disable Offline Mode' : 'Enable Offline Privacy Mode'}
              disabled={checking || offlineBusy}
              onClick={mode === 'offline' ? disableOffline : enableOffline}
              style={{
                ...styles.switch,
                ...(mode === 'offline' ? styles.switchOn : {}),
                opacity: offlineBusy ? 0.55 : 1,
              }}
            >
              <span style={{ ...styles.knob, ...(mode === 'offline' ? styles.knobOn : {}) }} />
            </button>
          </div>

          {offlineBusy && (
            <div style={styles.downloadBox}>
              <div style={styles.downloadHeader}>
                <span>Preparing Offline Mode</span>
                <strong>{pctText(progress.pct)}</strong>
              </div>
              <div style={styles.progressTrack}>
                <div style={{ ...styles.progressBar, width: `${Math.max(1, progress.pct)}%` }} />
              </div>
              <div style={styles.downloadMeta}>
                <span>{progress.file || 'Connecting to model storage…'}</span>
                <span>{progress.mbPerSec > 0 ? `${progress.mbPerSec.toFixed(1)} MB/s` : 'Starting…'}</span>
              </div>
              <div style={styles.note}>
                Large model files download once and resume automatically if interrupted.
              </div>
            </div>
          )}

          {downloadState === 'ready' && mode === 'offline' && (
            <div style={styles.ready}>
              <span style={styles.readyDot}>✓</span>
              Offline models are installed and ready.
            </div>
          )}

          {error && <div style={styles.error}>⚠ {error}{lastError && <div style={{ marginTop: 4, color: C.faint }}>Your Cloud/Offline selection was not changed.</div>}</div>}
        </section>

        <div style={styles.statusRow}>
          <span style={{ ...styles.statusDot, background: serviceStatus === 'online' ? C.green : serviceStatus === 'checking' ? C.amber : C.red }} />
          <span style={styles.statusLabel}>{serviceStatus === 'online' ? 'AI service ready' : serviceStatus === 'checking' ? 'Checking AI service…' : serviceStatus === 'not-configured' ? 'Cloud needs setup' : 'AI service unavailable'}</span>
          {mode === 'offline' && <span style={styles.statusMode}>LOCAL</span>}
        </div>

        <section style={styles.quickGrid}>
          <button style={styles.quickButton} onClick={() => openSidePanel('privacy')} disabled={!domain}>
            <span>🔎</span><span>Privacy</span>
          </button>
          <button style={styles.quickButton} onClick={() => openSidePanel('history')}>
            <span>◷</span><span>Site&nbsp;history</span>
          </button>
        </section>

        <section style={styles.privacy}>
          <div style={styles.privacyIcon}>↯</div>
          <div>
            <div style={styles.privacyTitle}>Privacy-first by choice</div>
            <div style={styles.privacyText}>
              Cloud Mode is the default for speed. Offline Mode is opt-in and requires an explicit model download.
            </div>
          </div>
        </section>

        {domain && (
          <button style={styles.primary} onClick={openFullReport}>
            View {domain} Report
            <span>→</span>
          </button>
        )}
        <button style={styles.secondary} onClick={openSettings}>Settings</button>
      </main>

      <footer style={styles.footer}>
        <span>SSENSE EDGE</span>
        <span>•</span>
        <span>{mode === 'offline' ? 'Local inference' : 'Cloud inference'}</span>
      </footer>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    width: '100%',
    maxWidth: 380,
    minHeight: 420,
    maxHeight: 600,
    overflowY: 'auto',
    background: C.bg,
    color: C.text,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    WebkitFontSmoothing: 'antialiased',
  },
  header: {
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '14px 16px', borderBottom: `1px solid ${C.border}`,
  },
  logo: {
    width: 28, height: 28, borderRadius: 8,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: `linear-gradient(135deg, ${C.cyan}, ${C.blue})`,
    flexShrink: 0,
  },
  brand: { fontSize: 14, fontWeight: 750 },
  sub: { fontSize: 9.5, color: C.faint, marginTop: 1, letterSpacing: .35 },
  domain: {
    marginLeft: 'auto', maxWidth: 130, overflow: 'hidden',
    textOverflow: 'ellipsis', whiteSpace: 'nowrap',
    fontSize: 10, color: C.muted, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  body: { padding: 16 },
  modeCard: {
    background: C.panel, border: `1px solid ${C.border}`,
    borderRadius: 13, padding: 15,
  },
  modeTop: { display: 'flex', gap: 12, alignItems: 'flex-start' },
  eyebrow: { fontSize: 9, letterSpacing: 1.3, color: C.faint, fontWeight: 700, marginBottom: 4 },
  modeTitle: { fontSize: 16, fontWeight: 750 },
  modeDescription: { fontSize: 11, color: C.muted, lineHeight: 1.45, marginTop: 5, maxWidth: 260 },
  switch: {
    marginLeft: 'auto', width: 44, height: 25, borderRadius: 20, border: 'none',
    padding: 3, background: '#27272A', cursor: 'pointer', flexShrink: 0,
  },
  switchOn: { background: 'linear-gradient(90deg, #0891B2, #2563EB)' },
  knob: {
    display: 'block', width: 19, height: 19, borderRadius: '50%',
    background: '#F4F4F5', transition: 'transform .18s ease',
  },
  knobOn: { transform: 'translateX(19px)' },
  downloadBox: {
    marginTop: 14, padding: 12, borderRadius: 10,
    background: 'rgba(255,255,255,.035)', border: `1px solid ${C.border}`,
  },
  downloadHeader: { display: 'flex', justifyContent: 'space-between', fontSize: 11, fontWeight: 650 },
  progressTrack: { height: 6, borderRadius: 6, background: '#27272A', marginTop: 9, overflow: 'hidden' },
  progressBar: {
    height: '100%', borderRadius: 6,
    background: `linear-gradient(90deg, ${C.cyan}, ${C.blue})`,
    transition: 'width .25s ease',
  },
  downloadMeta: {
    display: 'flex', justifyContent: 'space-between', gap: 8,
    marginTop: 7, fontSize: 9.5, color: C.muted,
  },
  note: { marginTop: 8, fontSize: 9.5, color: C.faint, lineHeight: 1.4 },
  ready: { marginTop: 12, fontSize: 10.5, color: C.green, display: 'flex', gap: 7, alignItems: 'center' },
  readyDot: {
    width: 18, height: 18, borderRadius: '50%', background: 'rgba(74,222,128,.12)',
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800,
  },
  error: { marginTop: 11, color: C.red, fontSize: 10.5, lineHeight: 1.4 },
  statusRow: {
    display: 'flex', alignItems: 'center', gap: 7,
    marginTop: 12, padding: '8px 11px', borderRadius: 9,
    background: 'rgba(255,255,255,.03)', border: `1px solid ${C.border}`,
  },
  statusDot: { width: 7, height: 7, borderRadius: '50%', flexShrink: 0 },
  statusLabel: { fontSize: 11, color: C.muted, fontWeight: 600, flex: 1 },
  statusMode: {
    fontSize: 8.5, fontWeight: 800, letterSpacing: .6, color: C.blue,
    background: 'rgba(59,130,246,.12)', padding: '2px 7px', borderRadius: 5,
  },
  quickGrid: {
    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10,
    marginTop: 12,
  },
  quickButton: {
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7,
    padding: '10px 12px', borderRadius: 9,
    background: 'rgba(255,255,255,.04)', border: `1px solid ${C.border}`,
    color: C.text, fontSize: 11.5, fontWeight: 650, cursor: 'pointer',
    transition: 'background 0.15s ease, border-color 0.15s ease',
  },
  privacy: {
    display: 'flex', gap: 10, marginTop: 12, padding: 12,
    border: `1px solid ${C.border}`, borderRadius: 11, background: 'rgba(34,211,238,.025)',
  },
  privacyIcon: {
    width: 25, height: 25, borderRadius: 7, background: 'rgba(34,211,238,.1)',
    color: C.cyan, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800,
  },
  privacyTitle: { fontSize: 10.5, fontWeight: 700 },
  privacyText: { fontSize: 9.5, color: C.faint, lineHeight: 1.4, marginTop: 2 },
  primary: {
    width: '100%', marginTop: 14, border: 'none', borderRadius: 9,
    padding: '10px 12px', background: C.cyan, color: C.bg,
    fontWeight: 750, fontSize: 11.5, cursor: 'pointer',
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
  },
  secondary: {
    width: '100%', marginTop: 8, border: `1px solid ${C.border}`, borderRadius: 9,
    padding: '9px 12px', background: 'rgba(255,255,255,.04)', color: C.text,
    fontWeight: 600, fontSize: 11, cursor: 'pointer',
  },
  footer: {
    padding: '9px 16px', borderTop: `1px solid ${C.border}`,
    display: 'flex', justifyContent: 'center', gap: 7,
    color: C.faint, fontSize: 8.5, letterSpacing: .7,
  },
};
