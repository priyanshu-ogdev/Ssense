import React, { useEffect, useState } from 'react';

// ═══════════════════════════════════════════════════════════════
// Ssense Popup — quick-glance compliance status for the active tab.
//
// DESIGN NOTE: this is a statutory audit tool, not a generic dashboard.
// The signature element is a stamped "audit seal" — a notched ring with
// tick marks, like an official document stamp — around the trust score,
// rather than a generic circular progress bar. Tick density and arc fill
// both encode the score, so the shape itself communicates "certified /
// under review / flagged" before any number is read.
//
// Palette matches the existing sidebar tokens (deep near-black, cyan/blue
// accent) for brand consistency across popup/options/sidebar — one product,
// one identity — with two additions specific to this surface: an amber for
// "under review" and a stamp-ink red for statutory violations, distinct
// from a generic UI error red.
// ═══════════════════════════════════════════════════════════════

type Status = 'loading' | 'unconfigured' | 'no-report' | 'ready' | 'error';

const COLORS = {
  bg: '#09090B',
  surface: 'rgba(255,255,255,0.03)',
  border: 'rgba(255,255,255,0.08)',
  textPrimary: '#F4F4F5',
  textMuted: '#A1A1AA',
  textFaint: '#71717A',
  cyan: '#06B6D4',
  blue: '#3B82F6',
  amber: '#F5A623',
  stampRed: '#D64545',
  success: '#4ADE80',
};

function tierFor(score: number): { label: string; color: string } {
  if (score >= 80) return { label: 'CERTIFIED COMPLIANT', color: COLORS.success };
  if (score >= 50) return { label: 'UNDER REVIEW', color: COLORS.amber };
  return { label: 'VIOLATIONS FLAGGED', color: COLORS.stampRed };
}

// The seal: a ring of 40 tick marks. Ticks fill clockwise proportional to
// score, colored by tier — reads like a dial being certified, not a loading
// spinner.
function AuditSeal({ score, color }: { score: number; color: string }) {
  const TICKS = 40;
  const filled = Math.round((score / 100) * TICKS);
  const radius = 44;
  const center = 52;

  const ticks = Array.from({ length: TICKS }, (_, i) => {
    const angle = (i / TICKS) * 2 * Math.PI - Math.PI / 2;
    const isFilled = i < filled;
    const len = isFilled ? 9 : 6;
    const x1 = center + (radius - len) * Math.cos(angle);
    const y1 = center + (radius - len) * Math.sin(angle);
    const x2 = center + radius * Math.cos(angle);
    const y2 = center + radius * Math.sin(angle);
    return (
      <line
        key={i}
        x1={x1} y1={y1} x2={x2} y2={y2}
        stroke={isFilled ? color : 'rgba(255,255,255,0.12)'}
        strokeWidth={isFilled ? 2.4 : 1.6}
        strokeLinecap="round"
      />
    );
  });

  return (
    <svg width={104} height={104} viewBox="0 0 104 104">
      {ticks}
      <circle cx={center} cy={center} r={radius - 16} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
      <text
        x={center} y={center - 2} textAnchor="middle"
        fontSize={26} fontWeight={700} fill={COLORS.textPrimary}
        fontFamily="ui-monospace, 'SF Mono', Menlo, monospace"
      >
        {score}
      </text>
      <text
        x={center} y={center + 16} textAnchor="middle"
        fontSize={8} fontWeight={600} fill={COLORS.textFaint}
        fontFamily="ui-monospace, 'SF Mono', Menlo, monospace"
        letterSpacing={1}
      >
        / 100
      </text>
    </svg>
  );
}

export default function Popup() {
  const [status, setStatus] = useState<Status>('loading');
  const [domain, setDomain] = useState<string>('');
  const [score, setScore] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState('');

  useEffect(() => {
    (async () => {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const url = tab?.url || '';
      let host = '';
      try { host = new URL(url).hostname; } catch { /* system page (chrome://, etc.) */ }

      if (!host) {
        setStatus('no-report');
        return;
      }
      setDomain(host);

      const config = await chrome.runtime.sendMessage({ type: 'GET_ENGINE_CONFIG' });
      if (!config?.configured) {
        setStatus('unconfigured');
        return;
      }

      try {
        const res = await chrome.runtime.sendMessage({ type: 'GET_TRUST_SCORE', domain: host });
        if (res?.success && typeof res.score === 'number') {
          setScore(res.score);
          setStatus('ready');
        } else {
          setStatus('error');
          setErrorMsg(res?.error || 'No audit on record for this site yet.');
        }
      } catch (err: any) {
        setStatus('error');
        setErrorMsg(err?.message || 'Could not reach the audit service.');
      }
    })();
  }, []);

  const openFullReport = async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) await chrome.sidePanel.open({ tabId: tab.id });
    window.close();
  };

  const openSettings = () => {
    chrome.runtime.openOptionsPage();
    window.close();
  };

  return (
    <div style={styles.root}>
      <div style={styles.header}>
        <div style={styles.headerIcon}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          </svg>
        </div>
        <div style={styles.headerText}>Ssense</div>
        <div style={styles.domainChip}>{domain || '—'}</div>
      </div>

      <div style={styles.body}>
        {status === 'loading' && <div style={styles.centerNote}>Checking this site…</div>}

        {status === 'unconfigured' && (
          <div style={styles.centerBlock}>
            <div style={styles.centerNote}>Set up your server connection to start auditing sites.</div>
            <button style={styles.primaryBtn} onClick={openSettings}>Open Settings</button>
          </div>
        )}

        {status === 'no-report' && (
          <div style={styles.centerNote}>Ssense doesn't audit browser or system pages.</div>
        )}

        {status === 'error' && (
          <div style={styles.centerBlock}>
            <div style={{ ...styles.centerNote, color: COLORS.amber }}>{errorMsg}</div>
            <button style={styles.secondaryBtn} onClick={openFullReport}>Run audit in full panel</button>
          </div>
        )}

        {status === 'ready' && score !== null && (
          <>
            <div style={styles.sealRow}>
              <AuditSeal score={score} color={tierFor(score).color} />
            </div>
            <div style={{ ...styles.tierLabel, color: tierFor(score).color }}>
              {tierFor(score).label}
            </div>
            <div style={styles.actions}>
              <button style={styles.primaryBtn} onClick={openFullReport}>View Full Report</button>
              <button style={styles.secondaryBtn} onClick={openSettings}>Settings</button>
            </div>
          </>
        )}
      </div>

      <div style={styles.footer}>Audited locally via your SLM server</div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    background: COLORS.bg,
    color: COLORS.textPrimary,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '12px 14px',
    borderBottom: `1px solid ${COLORS.border}`,
  },
  headerIcon: {
    width: 22, height: 22, borderRadius: 6,
    background: `linear-gradient(135deg, ${COLORS.cyan}, ${COLORS.blue})`,
    display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
  },
  headerText: { fontSize: 13, fontWeight: 700, flexShrink: 0 },
  domainChip: {
    marginLeft: 'auto', fontSize: 10.5, color: COLORS.textMuted,
    fontFamily: 'ui-monospace, SF Mono, Menlo, monospace',
    maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  body: {
    padding: '20px 18px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    minHeight: 150,
    justifyContent: 'center',
  },
  sealRow: { display: 'flex', justifyContent: 'center', marginBottom: 10 },
  tierLabel: {
    fontSize: 10.5, fontWeight: 700, letterSpacing: 1.2,
    fontFamily: 'ui-monospace, SF Mono, Menlo, monospace',
    marginBottom: 16,
  },
  actions: { display: 'flex', gap: 8, width: '100%' },
  centerBlock: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 },
  centerNote: { fontSize: 12, color: COLORS.textMuted, textAlign: 'center', lineHeight: 1.5, maxWidth: 240 },
  primaryBtn: {
    flex: 1, background: COLORS.cyan, color: COLORS.bg, border: 'none',
    borderRadius: 7, padding: '8px 12px', fontSize: 12, fontWeight: 700, cursor: 'pointer',
  },
  secondaryBtn: {
    flex: 1, background: 'rgba(255,255,255,0.06)', color: COLORS.textPrimary,
    border: `1px solid ${COLORS.border}`, borderRadius: 7, padding: '8px 12px',
    fontSize: 12, fontWeight: 600, cursor: 'pointer',
  },
  footer: {
    padding: '9px 14px', borderTop: `1px solid ${COLORS.border}`,
    fontSize: 10, color: COLORS.textFaint, textAlign: 'center',
  },
};
