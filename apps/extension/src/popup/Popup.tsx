// apps/extension/src/popup/Popup.tsx
//
// Toolbar popup. Three states:
//   1. Signed out  → Google sign-in
//   2. Signed in   → "This site" (live status + findings) / "All sites" (portfolio, collapsible)
// Always shows sync state and links to Settings / the full report panel.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Avatar, BrandMark, Collapsible, DomainTile, Icon, ScoreRing, Spinner, StatusPill, ViolationGroups, useOpenSet } from '../ui/components';
import { GoogleButton, SignInError, SignInPromise, useGoogleSignIn } from '../ui/SignIn';
import { highlightOnPage, openSidePanel, send, useActiveTab, useAuth, usePrefs, useSites, useSyncState, useTheme } from '../ui/hooks';
import { byAttention, summarize, type SiteRow } from '../utils/rows';
import { STATUS_META, formatRelative, isScannableHost } from '../utils/status';
import { normaliseDomain } from '../utils/domain';

type Tab = 'site' | 'all';

const SEG_COLORS: Record<string, string> = {
  compliant: 'var(--ssense-accent-emerald)', review: 'var(--ssense-accent-amber)', risk: 'var(--ssense-accent-rose)',
  scanning: 'var(--ssense-info)',
};

export const Popup: React.FC = () => {
  const { auth, reload: reloadAuth } = useAuth();
  const { prefs, update } = usePrefs();
  useTheme(prefs?.theme);

  if (!auth) return <div className="pp" style={{ minHeight: 560, height: 580, placeItems: 'center', display: 'grid' }}><Spinner size={22} /></div>;
  return <Main auth={auth} prefs={prefs} update={update} onDoneAuth={reloadAuth} />;
};

// ─── Signed-out ───────────────────────────────────────────────────────────────
export const SignedOut: React.FC<{ onDone: () => void }> = ({ onDone }) => {
  const { busy, error, signIn } = useGoogleSignIn(onDone);
  return (
    <div className="pp" style={{ padding: 18, gap: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <BrandMark size={34} />
        <div><div className="sx-display" style={{ fontSize: 17 }}>Ssense</div><div className="sx-eyebrow">DPDP privacy shield</div></div>
      </div>
      <div>
        <h1 className="sx-display" style={{ fontSize: 26, lineHeight: 1.1, margin: '0 0 8px' }}>See what every site does with your data.</h1>
        <p style={{ margin: 0, fontSize: 13, lineHeight: 1.55, color: 'var(--ssense-text-secondary)' }}>
          Sign in once. Ssense then audits each site’s privacy policy against India’s DPDP Act as you browse.
        </p>
      </div>
      <GoogleButton busy={busy} onClick={signIn} />
      <SignInError message={error} />
      <SignInPromise />
    </div>
  );
};

// ─── Main shell ───────────────────────────────────────────────────────────────
const Main: React.FC<{
  auth: NonNullable<ReturnType<typeof useAuth>['auth']>;
  prefs: ReturnType<typeof usePrefs>['prefs'];
  update: ReturnType<typeof usePrefs>['update'];
  onDoneAuth?: () => void;
}> = ({ auth, prefs, update }) => {
  const [tab, setTab] = useState<Tab>('site');
  const [guestNoticeDismissed, setGuestNoticeDismissed] = useState(false);
  const { rows, loading } = useSites();
  const { state: sync, syncNow } = useSyncState();
  const active = useActiveTab();
  const summary = useMemo(() => summarize(rows), [rows]);

  // Freshen data when the popup opens (cheap no-op if synced recently).
  const kicked = useRef(false);
  useEffect(() => {
    if (kicked.current || !sync || !auth.signedIn) return;
    kicked.current = true;
    if (!sync.lastSyncAt || Date.now() - sync.lastSyncAt > 2 * 60_000) void syncNow();
  }, [sync, syncNow, auth.signedIn]);

  const openFullTab = () => {
    chrome.tabs.create({ url: chrome.runtime.getURL('sidepanel.html') });
  };

  return (
    <div className="pp">
      <div className="pp-head">
        <BrandMark size={30} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="sx-display" style={{ fontSize: 15, lineHeight: 1.1 }}>Ssense</div>
          <div className="sx-muted sx-trunc" style={{ fontSize: 11 }} title={auth.signedIn ? (auth.email || auth.name) : 'Guest Mode (Local)'}>
            {auth.signedIn ? (auth.email || auth.name) : 'Guest Mode (Local)'}
          </div>
        </div>
        {prefs && (
          <button className="sx-icon-btn" onClick={() => update({ autoScan: !prefs.autoScan })} title={prefs.autoScan ? 'Auto-scan is on — click to pause' : 'Auto-scan is paused — click to resume'} aria-pressed={prefs.autoScan}>
            <Icon name={prefs.autoScan ? 'scan' : 'pause'} size={16} style={{ color: prefs.autoScan ? 'var(--ssense-accent)' : 'var(--ssense-accent-amber)' }} />
          </button>
        )}
        <button
          className="sx-btn sx-btn--ghost sx-btn--sm pp-max-btn"
          onClick={openFullTab}
          title="Stretch & Open full widescreen dashboard in a new tab"
          style={{ gap: 5, padding: '4px 8px', fontSize: 11.5 }}
        >
          <Icon name="maximize" size={13} />
          <span>Widescreen Tab</span>
        </button>
        <button
          className="sx-btn sx-btn--ghost sx-btn--sm"
          onClick={async () => {
            try {
              const currentWindow = await chrome.windows.getCurrent();
              if (currentWindow.id !== undefined && chrome.sidePanel?.open) {
                await chrome.sidePanel.open({ windowId: currentWindow.id });
                window.close();
                return;
              }
            } catch {}
            openFullTab();
          }}
          title="Open stretchable Side Panel on the side of your window"
          style={{ gap: 4, padding: '4px 8px', fontSize: 11.5 }}
        >
          <Icon name="sparkle" size={13} />
          <span>Side Panel</span>
        </button>
        <button className="sx-icon-btn" onClick={() => chrome.runtime.openOptionsPage()} title="Settings"><Icon name="settings" size={16} /></button>
        {auth.signedIn ? (
          <button className="sx-icon-btn" onClick={() => chrome.runtime.openOptionsPage()} title={auth.name} style={{ padding: 0 }}><Avatar name={auth.name} email={auth.email} url={auth.avatarUrl} size={26} /></button>
        ) : (
          <button className="sx-btn sx-btn--ghost sx-btn--sm" onClick={() => chrome.runtime.openOptionsPage()} title="Sign in with Google to sync across devices" style={{ padding: '3px 8px', fontSize: 11 }}>Sign in</button>
        )}
      </div>

      <div style={{ margin: '8px 14px 2px', display: 'flex', gap: 6, alignItems: 'center' }}>
        <div style={{ position: 'relative', flex: 1, display: 'flex', alignItems: 'center' }}>
          <Icon name="search" size={13} style={{ position: 'absolute', left: 10, color: 'var(--ssense-text-muted)', pointerEvents: 'none' }} />
          <input
            className="sx-input sx-mono"
            style={{ fontSize: 11.5, padding: '6px 10px 6px 30px', height: 32 }}
            placeholder="Type query or command (e.g. audit, retention)..."
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                const val = (e.target as HTMLInputElement).value;
                if (val.trim()) {
                  chrome.tabs.create({ url: chrome.runtime.getURL(`sidepanel.html?q=${encodeURIComponent(val.trim())}`) });
                }
              }
            }}
          />
        </div>
        <button
          className="sx-btn sx-btn--primary sx-btn--sm"
          style={{ height: 32, padding: '0 12px', fontSize: 11, display: 'inline-flex', alignItems: 'center', gap: 5 }}
          onClick={(e) => {
            const inputEl = e.currentTarget.parentElement?.querySelector('input') as HTMLInputElement | null;
            chrome.tabs.create({ url: chrome.runtime.getURL(`sidepanel.html?q=${encodeURIComponent(inputEl?.value || '')}`) });
          }}
          title="Send query to DPDP AI Co-pilot"
        >
          <Icon name="sparkle" size={12} />
          <span>Ask AI</span>
        </button>
      </div>

      {!auth.signedIn && !guestNoticeDismissed && (
        <div style={{ margin: '6px 14px 4px', padding: '6px 10px', borderRadius: 8, background: 'var(--ssense-bg-elevated)', border: '1px solid var(--ssense-border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11 }}>
          <span style={{ color: 'var(--ssense-text-secondary)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <Icon name="shield" size={12} style={{ color: 'var(--ssense-accent)' }} />
            <span>Auditing active locally · <a href="#" onClick={(e) => { e.preventDefault(); chrome.runtime.openOptionsPage(); }} style={{ color: 'var(--ssense-accent)', textDecoration: 'underline' }}>Sign in</a> to sync</span>
          </span>
          <button onClick={() => setGuestNoticeDismissed(true)} style={{ background: 'none', border: 'none', color: 'var(--ssense-text-muted)', cursor: 'pointer', padding: '0 2px' }} title="Dismiss">✕</button>
        </div>
      )}

      <div className="pp-tabs" role="tablist">
        <button className="pp-tab" role="tab" aria-selected={tab === 'site'} onClick={() => setTab('site')}><Icon name="globe" size={13} /> This site</button>
        <button className="pp-tab" role="tab" aria-selected={tab === 'all'} onClick={() => setTab('all')}>
          <Icon name="history" size={13} /> All sites <span className="sx-muted" style={{ fontWeight: 600 }}>{summary.total}</span>
        </button>
      </div>

      <div className="pp-body sx-scroll" role="tabpanel">
        {tab === 'site'
          ? <ThisSite host={active.host} rows={rows} loading={loading} autoScan={prefs?.autoScan ?? true} />
          : <AllSites rows={rows} summary={summary} loading={loading} />}
      </div>

      <div className="pp-foot">
        <Icon name={!auth.signedIn ? 'shield' : sync?.status === 'error' ? 'cloudOff' : 'sync'} size={13} className={sync?.status === 'syncing' ? 'sx-ring-spin' : undefined}
          style={{ color: sync?.status === 'error' ? 'var(--ssense-accent-rose)' : undefined }} />
        <span className="sx-trunc" style={{ flex: 1 }} title={sync?.lastError || ''}>
          {!auth.signedIn ? 'Guest Mode (Local)'
            : !prefs?.syncEnabled ? 'Sync is off'
            : sync?.status === 'syncing' ? 'Syncing…'
            : sync?.status === 'error' ? 'Sync problem — will retry'
            : sync?.lastSyncAt ? `Synced ${formatRelative(sync.lastSyncAt)}` : 'Not synced yet'}
        </span>
        {auth.signedIn && prefs?.syncEnabled && <button className="sx-btn sx-btn--ghost sx-btn--sm" onClick={syncNow} disabled={sync?.status === 'syncing'}>Sync now</button>}
        <button className="sx-btn sx-btn--primary sx-btn--sm" onClick={async () => { await openSidePanel('audit'); window.close(); }}>Open panel</button>
      </div>
    </div>
  );
};

// ─── This site ────────────────────────────────────────────────────────────────
const ThisSite: React.FC<{ host: string | null; rows: SiteRow[]; loading: boolean; autoScan: boolean }> = ({ host, rows, loading, autoScan }) => {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  const norm = host ? normaliseDomain(host) : null;
  const row = norm ? rows.find((r) => r.domain === norm) : undefined;

  if (!host || !isScannableHost(host)) {
    return <Empty icon="shield" title="Nothing to scan here" text="Ssense audits public websites. Open a site in this tab and it will be scanned automatically." />;
  }
  if (loading && !row) return <div className="sx-skel" style={{ height: 120 }} />;

  const scanNow = async () => {
    setBusy(true); setNote('');
    const r = await send<any>({ type: 'SCAN_NOW', domain: norm });
    setBusy(false);
    if (r?.success === false && r.error) setNote(r.error);
  };
  const toggleIgnore = () => send({ type: 'IGNORE_SITE', domain: norm, ignore: !row?.ignored });

  const status = row?.status ?? (autoScan ? 'scanning' : 'paused');
  const scanning = status === 'scanning';
  const meta = STATUS_META[status];
  const heroGlow = scanning ? 'sx-glow-cyan' : typeof row?.score === 'number' ? (row.score >= 80 ? 'sx-glow-emerald' : row.score >= 50 ? 'sx-glow-amber' : 'sx-glow-rose') : '';

  return (
    <div className="pp-site-layout">
      <div className="pp-site-left">
        <div className={`sx-card pp-hero ${heroGlow}`}>
          <ScoreRing score={row?.score ?? null} size={76} scanning={scanning} />
          <div style={{ minWidth: 0, display: 'grid', gap: 6 }}>
            <div className="sx-trunc" style={{ fontSize: 15, fontWeight: 700 }} title={host}>{host}</div>
            <div><StatusPill status={status} /></div>
            <div className="sx-muted" style={{ fontSize: 11.5, lineHeight: 1.4 }}>
              {scanning ? 'Reading the privacy policy…'
                : row?.entry.lastAuditAt ? <>
                    {row.counts.total === 0 ? 'No violations' : `${row.counts.total} issue${row.counts.total === 1 ? '' : 's'}${row.counts.high ? ` · ${row.counts.high} high impact` : ''}`}
                    {' · '}audited {formatRelative(row.entry.lastAuditAt)}
                  </> : meta.hint}
            </div>
          </div>
        </div>

        {(status === 'error' || status === 'nopolicy') && row?.error && (
          <div style={{ fontSize: 12, lineHeight: 1.5, padding: '9px 11px', borderRadius: 9, background: 'var(--ssense-bg-elevated)', color: 'var(--ssense-text-secondary)' }}>{row.error}</div>
        )}
        {note && (
          <div role="alert" style={{ fontSize: 11.5, padding: '9px 11px', borderRadius: 9, background: 'var(--ssense-bad-soft)', color: 'var(--ssense-accent-rose)', display: 'flex', alignItems: 'flex-start', gap: 7, lineHeight: 1.45 }}>
            <Icon name="alert" size={14} style={{ marginTop: 1, flexShrink: 0 }} />
            <div style={{ flex: 1 }}>
              <span>{note}</span>
              {(note.toLowerCase().includes('server') || note.toLowerCase().includes('sign in') || note.toLowerCase().includes('connection')) && (
                <div style={{ marginTop: 4 }}>
                  <a href="#" onClick={(e) => { e.preventDefault(); chrome.runtime.openOptionsPage(); }} style={{ color: 'var(--ssense-accent)', textDecoration: 'underline', fontWeight: 600 }}>
                    {note.toLowerCase().includes('sign in') ? 'Sign in in Settings →' : 'Configure server in Settings →'}
                  </a>
                </div>
              )}
            </div>
          </div>
        )}

        <div style={{ display: 'flex', gap: 8 }}>
          <button className="sx-btn sx-btn--primary" style={{ flex: 1 }} onClick={scanNow} disabled={busy || scanning || row?.ignored}>
            {busy || scanning ? <Spinner size={14} /> : <Icon name="refresh" size={14} />}
            {row?.entry.lastAuditAt ? 'Re-scan now' : 'Scan now'}
          </button>
          <button className="sx-btn" onClick={async () => { await openSidePanel('audit'); window.close(); }} title="Ask the co-pilot about this site"><Icon name="chat" size={14} /> Ask AI</button>
          <button className="sx-btn" onClick={toggleIgnore} title={row?.ignored ? 'Resume scanning this site' : 'Never scan this site'} aria-pressed={row?.ignored}>
            <Icon name={row?.ignored ? 'scan' : 'eyeOff'} size={14} />
          </button>
        </div>
      </div>

      <div className="pp-site-right">
        {row && row.entry.lastReport ? (
          <div style={{ display: 'grid', gap: 8 }}>
            <div className="sx-eyebrow">DPDP Act Compliance Findings</div>
            <ViolationGroups violations={row.violations} compact onHighlight={highlightOnPage} />
          </div>
        ) : (
          <div className="sx-card" style={{ padding: 24, textAlign: 'center', color: 'var(--ssense-text-secondary)', fontSize: 12.5, display: 'grid', gap: 8, placeItems: 'center' }}>
            <Icon name="shield" size={24} style={{ color: 'var(--ssense-accent)' }} />
            <div>No DPDP compliance findings recorded yet. Click <b>Scan now</b> to audit this site.</div>
          </div>
        )}
      </div>
    </div>
  );
};

// ─── All sites ────────────────────────────────────────────────────────────────
const AllSites: React.FC<{ rows: SiteRow[]; summary: ReturnType<typeof summarize>; loading: boolean }> = ({ rows, summary, loading }) => {
  const openSet = useOpenSet();
  const sorted = useMemo(() => [...rows].sort(byAttention).slice(0, 60), [rows]);
  if (loading) return <><div className="sx-skel" style={{ height: 64 }} /><div className="sx-skel" style={{ height: 48 }} /><div className="sx-skel" style={{ height: 48 }} /></>;
  if (!rows.length) return <Empty icon="history" title="No sites yet" text="Browse as usual — every site you open shows up here with its compliance status." />;

  const segs = (['risk', 'review', 'compliant', 'scanning'] as const).filter((k) => summary.byStatus[k] > 0);
  const others = summary.total - segs.reduce((n, k) => n + summary.byStatus[k], 0);
  return (
    <>
      <div className="sx-card sx-card--pad" style={{ display: 'grid', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
          <div><span className="sx-display" style={{ fontSize: 28 }}>{summary.avgScore ?? '–'}</span><span className="sx-muted" style={{ fontSize: 12 }}> avg score</span></div>
          <div className="sx-muted" style={{ fontSize: 12 }}>{summary.total} sites · {summary.violations} issues</div>
        </div>
        <div className="pp-seg" role="img" aria-label="Sites by status">
          {segs.map((k) => <span key={k} style={{ flex: summary.byStatus[k], background: SEG_COLORS[k] }} />)}
          {others > 0 && <span style={{ flex: others, background: 'var(--ssense-border-strong)' }} />}
        </div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 11.5 }}>
          {([['risk', 'Non-compliant'], ['review', 'Review'], ['compliant', 'Compliant'], ['scanning', 'Scanning']] as const).map(([k, l]) => (
            <span key={k} style={{ display: 'inline-flex', gap: 5, alignItems: 'center', color: 'var(--ssense-text-secondary)' }}>
              <i style={{ width: 7, height: 7, borderRadius: '50%', background: SEG_COLORS[k] }} />{l} <b style={{ color: 'var(--ssense-text-primary)' }}>{summary.byStatus[k]}</b>
            </span>
          ))}
        </div>
      </div>

      <div className="pp-sites-grid">
        {sorted.map((r) => (
          <Collapsible key={r.domain} open={openSet.isOpen(r.domain)} onToggle={() => openSet.toggle(r.domain)} className="sx-card"
            headClassName="pp-site-head"
            header={
              <>
                <DomainTile domain={r.domain} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="sx-trunc" style={{ fontSize: 13, fontWeight: 650 }}>{r.domain}</div>
                  <div style={{ marginTop: 3 }}><StatusPill status={r.status} /></div>
                </div>
                <ScoreRing score={r.score} size={34} scanning={r.status === 'scanning'} />
              </>
            }>
            <div style={{ padding: '2px 12px 12px', display: 'grid', gap: 9 }}>
              {r.counts.total > 0 ? (
                <>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {r.counts.high > 0 && <span className="sx-pill sx-tone-bad"><i />{r.counts.high} high</span>}
                    {r.counts.medium > 0 && <span className="sx-pill sx-tone-warn"><i />{r.counts.medium} medium</span>}
                    {r.counts.low > 0 && <span className="sx-pill sx-tone-info"><i />{r.counts.low} low</span>}
                  </div>
                  <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, lineHeight: 1.6, color: 'var(--ssense-text-secondary)' }}>
                    {r.violations.slice(0, 3).map((v, i) => <li key={i}>{v.violation_type.replace(/_/g, ' ').toLowerCase()}</li>)}
                    {r.violations.length > 3 && <li className="sx-muted">+{r.violations.length - 3} more</li>}
                  </ul>
                </>
              ) : <div className="sx-muted" style={{ fontSize: 12 }}>{r.error || STATUS_META[r.status].hint}</div>}
              <button className="sx-btn sx-btn--sm" onClick={async () => { await openSidePanel('history', r.domain); window.close(); }}>Full details <Icon name="external" size={12} /></button>
            </div>
          </Collapsible>
        ))}
      </div>
      {rows.length > sorted.length && <div className="sx-muted" style={{ fontSize: 11.5, textAlign: 'center', marginTop: 8 }}>Showing 60 of {rows.length} — open the panel for everything.</div>}
    </>
  );
};

const Empty: React.FC<{ icon: string; title: string; text: string }> = ({ icon, title, text }) => (
  <div style={{ textAlign: 'center', padding: '26px 10px', display: 'grid', gap: 8, justifyItems: 'center' }}>
    <div style={{ width: 44, height: 44, borderRadius: 14, display: 'grid', placeItems: 'center', background: 'var(--ssense-accent-soft)', color: 'var(--ssense-accent)' }}><Icon name={icon} size={22} /></div>
    <div className="sx-display" style={{ fontSize: 17 }}>{title}</div>
    <div style={{ fontSize: 12.5, lineHeight: 1.5, color: 'var(--ssense-text-secondary)', maxWidth: 270 }}>{text}</div>
  </div>
);
