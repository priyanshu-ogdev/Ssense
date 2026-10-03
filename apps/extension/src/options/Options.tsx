// apps/extension/src/options/Options.tsx — Settings (public v1)
//
// Sections: Account · Scanning · Alerts & protection · Sync & devices · Appearance ·
// About. Every toggle saves immediately.

import React, { useEffect, useState } from 'react';
import { Avatar, Icon, Spinner, Switch, Toast, useToast } from '../ui/components';
import { GoogleButton, SignInError, SignInPromise, useGoogleSignIn } from '../ui/SignIn';
import { send, useAuth, usePrefs, useSyncState, useTheme } from '../ui/hooks';
import { normaliseDomain } from '../utils/domain';
import './options.css';

const NAV: [string, string, string][] = [
  ['account', 'Account', 'lock'],
  ['scanning', 'Scanning', 'scan'],
  ['alerts', 'Alerts & protection', 'bell'],
  ['server', 'Server', 'globe'],
  ['sync', 'Sync & devices', 'sync'],
  ['appearance', 'Appearance', 'sparkle'],
  ['about', 'About', 'check'],
];

const Item: React.FC<{ title: string; hint?: string; children: React.ReactNode }> = ({ title, hint, children }) => (
  <div className="op-item"><div style={{ minWidth: 0, flex: 1 }}><b>{title}</b>{hint && <small>{hint}</small>}</div>{children}</div>
);
const Section: React.FC<{ id: string; title: string; desc?: string; children: React.ReactNode }> = ({ id, title, desc, children }) => (
  <section className="op-sec" id={id}><h2 className="sx-display">{title}</h2>{desc && <p>{desc}</p>}<div className="sx-card op-list">{children}</div></section>
);

export default function Options({ onBack }: { onBack?: () => void } = {}) {
  const { auth, reload } = useAuth();
  const { prefs, update } = usePrefs();
  const { state: syncState, syncNow } = useSyncState();
  const { msg, show } = useToast();
  useTheme(prefs?.theme);
  const [section, setSection] = useState('account');
  const [ignoreInput, setIgnoreInput] = useState('');
  const { busy, error, signIn } = useGoogleSignIn(() => { void reload(); show('Signed in'); });
  const version = chrome.runtime.getManifest?.().version ?? '1.0.0';

  const [overrideEnabled, setOverrideEnabled] = useState(false);
  const [serverUrl, setServerUrl] = useState((import.meta.env.VITE_SSENSE_SERVER_URL as string) || 'http://localhost:8000');
  const [apiKey, setApiKey] = useState((import.meta.env.VITE_SSENSE_API_KEY as string) || '');
  const [hmacSecret, setHmacSecret] = useState((import.meta.env.VITE_SSENSE_HMAC_SECRET as string) || '');
  const [testingConnection, setTestingConnection] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);

  useEffect(() => {
    chrome.storage.local.get([
      'ssense_override_enabled',
      'ssense_server_url',
      'ssense_api_key',
      'ssense_hmac_secret',
    ]).then((d) => {
      if (d.ssense_override_enabled !== undefined) setOverrideEnabled(Boolean(d.ssense_override_enabled));
      if (d.ssense_server_url) setServerUrl(d.ssense_server_url);
      if (d.ssense_api_key) setApiKey(d.ssense_api_key);
      if (d.ssense_hmac_secret) setHmacSecret(d.ssense_hmac_secret);
    }).catch(() => {});
  }, []);

  const saveServerConfig = async (override?: boolean) => {
    const isOverride = override !== undefined ? override : overrideEnabled;
    await chrome.storage.local.set({
      ssense_override_enabled: isOverride,
      ssense_server_url: serverUrl.trim().replace(/\/$/, ''),
      ssense_api_key: apiKey.trim(),
      ssense_hmac_secret: hmacSecret.trim(),
    });
    show('Server configuration saved');
  };

  const handleTestConnection = async () => {
    setTestingConnection(true);
    setTestResult(null);
    const target = serverUrl.trim().replace(/\/$/, '');
    try {
      const res = await send<any>({ type: 'GET_SERVER_PING', url: target });
      if (res?.success || res?.online) {
        setTestResult({
          success: true,
          message: `Connected to Ssense SLM Server${res.version ? ` (v${res.version})` : ''}`,
        });
      } else {
        const direct = await fetch(`${target}/health`, { method: 'GET' }).catch(() => null);
        if (direct && direct.ok) {
          setTestResult({ success: true, message: 'Connected to Ssense SLM Server' });
        } else {
          setTestResult({
            success: false,
            message: res?.error || 'Cannot reach server. Verify host IP, port 8000, and firewall.',
          });
        }
      }
    } catch (e: any) {
      setTestResult({ success: false, message: e?.message || 'Connection failed' });
    } finally {
      setTestingConnection(false);
    }
  };

  const set = (patch: Parameters<typeof update>[0]) => { void update(patch); show('Saved'); };


  useEffect(() => {
    const ids = NAV.map(([id]) => id);
    const io = new IntersectionObserver((es) => {
      const v = es.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (v) setSection(v.target.id);
    }, { rootMargin: '-10% 0px -70% 0px' });
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (el) io.observe(el);
    });
    return () => io.disconnect();
  }, [prefs === null]);

  const addIgnore = () => {
    const d = normaliseDomain(ignoreInput);
    if (!d || !d.includes('.')) { show('Enter a domain like example.com'); return; }
    if (prefs && !prefs.ignoredDomains.includes(d)) set({ ignoredDomains: [...prefs.ignoredDomains, d] });
    setIgnoreInput('');
  };

  const handleBackToBrowsing = async () => {
    if (onBack) {
      onBack();
      return;
    }
    try {
      const tabs = await chrome.tabs.query({ currentWindow: true });
      const browsingTab = tabs.find((t) => t.url && t.url.startsWith('http') && !t.url.includes('chrome-extension://'));
      if (browsingTab?.id) {
        await chrome.tabs.update(browsingTab.id, { active: true });
        const curr = await chrome.tabs.getCurrent();
        if (curr?.id) await chrome.tabs.remove(curr.id);
        return;
      }
    } catch {}
    if (window.history.length > 1) {
      window.history.back();
    } else {
      window.close();
    }
  };

  if (!prefs || !auth) return <div style={{ minHeight: '100vh', display: 'grid', placeItems: 'center' }}><Spinner size={24} /></div>;

  return (
    <div className="op">
      <nav className="op-nav" aria-label="Settings sections">
        <button
          className="sx-btn sx-btn--ghost"
          onClick={handleBackToBrowsing}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 7,
            marginBottom: 16,
            padding: '8px 12px',
            fontSize: 12.5,
            fontWeight: 650,
            color: 'var(--ssense-accent-cyan)',
            background: 'var(--ssense-bg-elevated)',
            border: '1px solid var(--ssense-border)',
            borderRadius: 8,
            cursor: 'pointer',
            width: '100%',
            justifyContent: 'flex-start',
            transition: 'all 0.15s ease'
          }}
          title={onBack ? 'Return to chat' : 'Return to your active webpage / browsing tab'}
        >
          <Icon name="arrowLeft" size={14} />
          <span>{onBack ? 'Back to Chat' : 'Back to Browsing'}</span>
        </button>

        <div className="op-brand" style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 18 }}>
          <div style={{
            width: 32,
            height: 32,
            borderRadius: 9,
            background: 'var(--ssense-gradient-ai)',
            display: 'grid',
            placeItems: 'center',
            color: '#ffffff',
            boxShadow: '0 2px 8px rgba(6, 182, 212, 0.25)',
            flexShrink: 0
          }}>
            <Icon name="settings" size={18} />
          </div>
          <div className="sx-display" style={{ fontSize: 18 }}>Settings</div>
        </div>
        {NAV.map(([id, label, icon]) => (
          <a
            key={id}
            href={`#${id}`}
            aria-current={section === id}
            onClick={(e) => {
              e.preventDefault();
              document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
              setSection(id);
            }}
          >
            <Icon name={icon} size={15} />{label}
          </a>
        ))}
      </nav>

      <main className="op-main">

        <Section id="account" title="Account" desc="Signing in with Google keeps your history and settings on every device you use.">
          {auth.signedIn ? (
            <>
              <div className="op-item">
                <div style={{ display: 'flex', gap: 12, alignItems: 'center', minWidth: 0 }}>
                  <Avatar name={auth.name} email={auth.email} url={auth.avatarUrl} size={44} />
                  <div style={{ minWidth: 0 }}><b>{auth.name || 'Signed in'}</b><small className="sx-trunc">{auth.email}</small></div>
                </div>
                <span className="sx-pill sx-tone-ok"><i />Google account</span>
              </div>
              <Item title="Sign out" hint="Removes this device’s credentials.">
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <button className="sx-btn sx-btn--sm" onClick={async () => { await send({ type: 'SIGN_OUT', keepLocalData: true }); await reload(); }}><Icon name="logout" size={14} /> Sign out</button>
                </div>
              </Item>
            </>
          ) : (
            <div style={{ padding: '16px 0', display: 'grid', gap: 12, maxWidth: 420 }}>
              <GoogleButton busy={busy} onClick={signIn} /><SignInError message={error} /><SignInPromise />
            </div>
          )}
        </Section>

        <Section id="scanning" title="Scanning & Layout" desc="Ssense reads a site’s public privacy policy — never the pages you browse.">
          <Item title="Extension icon click" hint="Choose what opens when you click the Ssense icon in your browser toolbar.">
            <select
              className="sx-input sx-select"
              style={{ width: 'auto' }}
              value={prefs.toolbarAction || 'sidepanel'}
              onChange={(e) => set({ toolbarAction: e.target.value as any })}
              aria-label="Extension icon click behavior"
            >
              <option value="sidepanel">Side Panel (Docked, Stretchable &amp; Commands) [Default]</option>
              <option value="tab">Full Widescreen Dashboard (New Tab)</option>
              <option value="popup">Fixed Popup Window (480px)</option>
            </select>
          </Item>
          <Item title="Scan sites automatically" hint="Audit a site when you open it."><Switch label="Scan automatically" checked={prefs.autoScan} onChange={(v) => set({ autoScan: v })} /></Item>
          <Item title="Re-scan after" hint="Sites audited more recently than this are served instantly from your saved results.">
            <select className="sx-input sx-select" style={{ width: 'auto' }} value={prefs.rescanAfterDays} onChange={(e) => set({ rescanAfterDays: Number(e.target.value) })} aria-label="Re-scan interval">
              {[7, 14, 30, 60, 90].map((d) => <option key={d} value={d}>{d} days</option>)}
            </select>
          </Item>
          <div style={{ padding: '14px 0', display: 'grid', gap: 10 }}>
            <div><b style={{ fontSize: 14 }}>Never scan these sites</b><small style={{ display: 'block', fontSize: 12.5, color: 'var(--ssense-text-muted)', marginTop: 2 }}>Banks, work tools, anything you’d rather skip. Subdomains are included.</small></div>
            <div style={{ display: 'flex', gap: 8 }}>
              <input className="sx-input" value={ignoreInput} onChange={(e) => setIgnoreInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && addIgnore()} placeholder="example.com" aria-label="Domain to ignore" />
              <button className="sx-btn" onClick={addIgnore}>Add</button>
            </div>
            <div className="op-chips">
              {prefs.ignoredDomains.length === 0 && <span className="sx-muted" style={{ fontSize: 12.5 }}>No ignored sites.</span>}
              {prefs.ignoredDomains.map((d) => (
                <span key={d} className="sx-tag" style={{ display: 'inline-flex', gap: 6, alignItems: 'center', padding: '4px 6px 4px 10px' }}>{d}
                  <button className="sx-icon-btn" style={{ width: 18, height: 18 }} aria-label={`Stop ignoring ${d}`} onClick={() => set({ ignoredDomains: prefs.ignoredDomains.filter((x) => x !== d) })}><Icon name="x" size={11} /></button>
                </span>
              ))}
            </div>
          </div>
        </Section>

        <Section id="alerts" title="Alerts & protection">
          <Item title="Low-score notifications" hint="A desktop notification when a site drops below your threshold."><Switch label="Low-score notifications" checked={prefs.notifyOnLowScore} onChange={(v) => set({ notifyOnLowScore: v })} /></Item>
          <Item title={`Alert below ${prefs.lowScoreThreshold}`} hint="Trust score that counts as risky.">
            <input type="range" min={10} max={90} step={5} value={prefs.lowScoreThreshold} disabled={!prefs.notifyOnLowScore} onChange={(e) => set({ lowScoreThreshold: Number(e.target.value) })} aria-label="Alert threshold" style={{ accentColor: 'var(--ssense-accent)', width: 160 }} />
          </Item>
          <Item title="Apply recommended protections" hint="Block third-party trackers, send GPC and mask fingerprinting on sites whose audit calls for it."><Switch label="Apply protections" checked={prefs.enforceProtections} onChange={(v) => set({ enforceProtections: v })} /></Item>
          <Item title="Score on toolbar icon" hint="Show the trust score as a badge."><Switch label="Toolbar badge" checked={prefs.showBadge} onChange={(v) => set({ showBadge: v })} /></Item>
        </Section>



        <Section id="server" title="Server" desc="Configure your SLM inference server connection or connect across PCs on the same Wi-Fi / LAN.">
          <Item title="Custom server override" hint="Enable custom server override for local cluster or self-hosted deployment.">
            <Switch
              label="Custom server override"
              checked={overrideEnabled}
              onChange={(v) => {
                setOverrideEnabled(v);
                void saveServerConfig(v);
              }}
            />
          </Item>

          {overrideEnabled && (
            <div style={{ padding: '16px 0', display: 'grid', gap: 14 }}>
              <div>
                <b style={{ fontSize: 13.5, display: 'block', marginBottom: 4 }}>Server URL</b>
                <input
                  className="sx-input"
                  style={{ width: '100%', fontFamily: 'monospace', fontSize: 13 }}
                  value={serverUrl}
                  onChange={(e) => setServerUrl(e.target.value)}
                  placeholder="https://api.example.com or http://localhost:8000"
                  aria-label="Server URL"
                />
              </div>

              <div>
                <b style={{ fontSize: 13.5, display: 'block', marginBottom: 4 }}>API Key</b>
                <input
                  className="sx-input"
                  style={{ width: '100%', fontFamily: 'monospace', fontSize: 13 }}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="API Key"
                  aria-label="API Key"
                />
              </div>

              <div>
                <b style={{ fontSize: 13.5, display: 'block', marginBottom: 4 }}>HMAC Secret</b>
                <input
                  className="sx-input"
                  type="password"
                  style={{ width: '100%', fontFamily: 'monospace', fontSize: 13 }}
                  value={hmacSecret}
                  onChange={(e) => setHmacSecret(e.target.value)}
                  placeholder="HMAC Secret"
                  aria-label="HMAC Secret"
                />
              </div>

              <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 4, flexWrap: 'wrap' }}>
                <button
                  className="sx-btn sx-btn--primary"
                  onClick={() => void saveServerConfig()}
                  style={{ padding: '7px 16px', fontSize: 13 }}
                >
                  Save
                </button>
                <button
                  className="sx-btn"
                  onClick={() => void handleTestConnection()}
                  disabled={testingConnection}
                  style={{ padding: '7px 16px', fontSize: 13, display: 'inline-flex', alignItems: 'center', gap: 6 }}
                >
                  {testingConnection ? <Spinner size={14} /> : <Icon name="refresh" size={14} />}
                  <span>{testingConnection ? 'Testing…' : 'Test connection'}</span>
                </button>
              </div>

              {testResult && (
                <div
                  style={{
                    padding: '10px 14px',
                    borderRadius: 8,
                    fontSize: 13,
                    fontWeight: 600,
                    marginTop: 6,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    background: testResult.success ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)',
                    color: testResult.success ? '#10b981' : '#ef4444',
                    border: `1px solid ${testResult.success ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                  }}
                >
                  <Icon name={testResult.success ? 'check' : 'alert'} size={16} />
                  <span>{testResult.success ? `✅ ${testResult.message}` : `❌ ${testResult.message}`}</span>
                </div>
              )}
            </div>
          )}
        </Section>

        <Section id="sync" title="Sync & devices" desc="Your audit history and site settings sync automatically across signed-in devices.">
          <Item title="Cloud sync status" hint={syncState?.status === 'syncing' ? 'Sync in progress…' : syncState?.lastSyncAt ? `Last synced: ${new Date(syncState.lastSyncAt).toLocaleTimeString()}` : 'Not synced yet'}>
            <button className="sx-btn sx-btn--sm" onClick={() => void syncNow()} disabled={syncState?.status === 'syncing'}>
              <Icon name="sync" size={14} />
              <span>{syncState?.status === 'syncing' ? 'Syncing…' : 'Sync now'}</span>
            </button>
          </Item>
        </Section>

        <Section id="appearance" title="Appearance">
          <Item title="Theme" hint="Follow your system or choose one.">
            <div style={{ display: 'flex', gap: 4, padding: 3, borderRadius: 10, background: 'var(--ssense-bg-elevated)' }} role="radiogroup" aria-label="Theme">
              {(['system', 'light', 'dark'] as const).map((t) => (
                <button key={t} role="radio" aria-checked={prefs.theme === t} className="sx-btn sx-btn--sm" style={{ border: 0, background: prefs.theme === t ? 'var(--ssense-bg-surface)' : 'transparent' }} onClick={() => set({ theme: t })}>{t[0].toUpperCase() + t.slice(1)}</button>
              ))}
            </div>
          </Item>
        </Section>

        <Section id="about" title="About">
          <Item title="Ssense — DPDP Privacy Shield" hint="Automated compliance auditing against the Digital Personal Data Protection Act, 2023."><span className="sx-stamp">v{version}</span></Item>
          <Item title="Severity labels" hint="High / medium / low impact groupings are a display aid derived from the violation type; they don’t change the trust score."><span /></Item>
        </Section>
      </main>
      <Toast message={msg} />
    </div>
  );
}
