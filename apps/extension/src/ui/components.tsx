// apps/extension/src/ui/components.tsx — shared presentational components.

import React, { useEffect, useRef, useState } from 'react';
import type { Violation } from '../types/server-protocol';
import { SEVERITY_LABEL, SEVERITY_ORDER, groupBySeverity, prettyAction, prettyViolationType, severityOf, type Severity } from '../utils/severity';
import { STATUS_META, scoreTone, type SiteStatus } from '../utils/status';

// ─── Icons (inline, stroke-based) ─────────────────────────────────────────────
const PATHS: Record<string, React.ReactNode> = {
  shield: <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />,
  check: <path d="M20 6 9 17l-5-5" />,
  alert: <><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /><path d="M12 9v4M12 17h.01" /></>,
  x: <path d="M18 6 6 18M6 6l12 12" />,
  chevron: <path d="m9 6 6 6-6 6" />,
  refresh: <><path d="M21 12a9 9 0 0 1-15.5 6.2L3 16" /><path d="M3 21v-5h5M3 12A9 9 0 0 1 18.5 5.8L21 8" /><path d="M21 3v5h-5" /></>,
  settings: <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" /></>,
  history: <><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5M12 7v5l3 2" /></>,
  sync: <><path d="M4 12a8 8 0 0 1 13.7-5.6L20 8" /><path d="M20 4v4h-4M20 12a8 8 0 0 1-13.7 5.6L4 16" /><path d="M4 20v-4h4" /></>,
  cloudOff: <><path d="m2 2 20 20M8.4 4.6A7 7 0 0 1 19 10h.5a3.5 3.5 0 0 1 1.9 6.4M5.6 6.6A5 5 0 0 0 6 17h10" /></>,
  search: <><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" /></>,
  external: <><path d="M15 3h6v6M10 14 21 3" /><path d="M18 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h5" /></>,
  download: <><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" /></>,
  trash: <><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6" /></>,
  pause: <path d="M6 4h4v16H6zM14 4h4v16h-4z" />,
  globe: <><circle cx="12" cy="12" r="10" /><path d="M2 12h20M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20" /></>,
  lock: <><rect x="4" y="11" width="16" height="10" rx="2" /><path d="M8 11V7a4 4 0 0 1 8 0v4" /></>,
  bell: <><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9M10.3 21a1.9 1.9 0 0 0 3.4 0" /></>,
  logout: <><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" /></>,
  scan: <><path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2" /><path d="M7 12h10" /></>,
  clock: <><circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" /></>,
  device: <><rect x="2" y="4" width="14" height="10" rx="1.5" /><path d="M0 18h18M18 8h4a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1h-4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z" /></>,
  chat: <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />,
  eyeOff: <><path d="M17.9 17.9A10.9 10.9 0 0 1 12 20c-7 0-11-8-11-8a18.5 18.5 0 0 1 5.1-5.9M9.9 4.2A9.1 9.1 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.2 3.2M1 1l22 22" /></>,
  expand: <><path d="m7 15 5 5 5-5M7 9l5-5 5 5" /></>,
  collapse: <><path d="m7 20 5-5 5 5M7 4l5 5 5-5" /></>,
  maximize: <><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" /></>,
  arrowLeft: <path d="m12 19-7-7 7-7M5 12h14" />,
  sparkle: <path d="M12 3l2.1 5.9L20 11l-5.9 2.1L12 19l-2.1-5.9L4 11l5.9-2.1z" />,
  brain: <><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 11 18V7a2 2 0 0 0-2-2" /><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 13 18V7a2 2 0 0 1 2-2" /></>,
  zap: <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />,
  copy: <><rect width="14" height="14" x="8" y="8" rx="2" ry="2" /><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" /></>,
  checkCopy: <path d="M20 6 9 17l-5-5" />,
  fileText: <><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" /><path d="M14 2v4a2 2 0 0 0 2 2h4" /><path d="M10 9H8M16 13H8M16 17H8" /></>,
  pin: <><line x1="12" y1="17" x2="12" y2="22" /><path d="M5 17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1v4.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24Z" /></>,
  pinOff: <><line x1="2" y1="2" x2="22" y2="22" /><path d="M12 17v5M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V17h12M15 9.34V6h1a2 2 0 0 0 0-4H7.89" /></>,
  quote: <><path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V20c0 1 0 1 1 1z" /><path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V20c0 1 0 1 1 1z" /></>,
  scale: <><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" /><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" /><path d="M7 21h10M12 3v18M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2" /></>,
  shieldCheck: <><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /><path d="m9 12 2 2 4-4" /></>,
  shieldAlert: <><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></>,
  infoCircle: <><circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" /></>,
};

export const Icon: React.FC<{ name: keyof typeof PATHS | string; size?: number; className?: string; style?: React.CSSProperties }> = ({ name, size = 16, className, style }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className={className} style={style} aria-hidden="true">
    {PATHS[name] ?? null}
  </svg>
);

export const GoogleG: React.FC<{ size?: number }> = ({ size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
    <path fill="#4285F4" d="M23.745 12.27c0-.79-.07-1.54-.19-2.27H12v4.51h6.47c-.29 1.48-1.14 2.73-2.4 3.58v3h3.86c2.26-2.09 3.56-5.17 3.56-8.82z" />
    <path fill="#34A853" d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.86-3c-1.08.72-2.45 1.16-4.07 1.16-3.13 0-5.78-2.11-6.73-4.96H1.29v3.09C3.26 21.3 7.31 24 12 24z" />
    <path fill="#FBBC05" d="M5.27 14.29A7.2 7.2 0 0 1 4.9 12c0-.8.14-1.57.37-2.29V6.62H1.29A11.99 11.99 0 0 0 0 12c0 1.94.46 3.77 1.29 5.38l3.98-3.09z" />
    <path fill="#EA4335" d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.31 0 3.26 2.7 1.29 6.62l3.98 3.09C6.22 6.86 8.87 4.75 12 4.75z" />
  </svg>
);

export const BrandMark: React.FC<{ size?: number }> = ({ size = 30 }) => (
  <div style={{ width: size, height: size, borderRadius: size * 0.3, background: 'var(--ssense-gradient-ai)', display: 'grid', placeItems: 'center', color: 'var(--ssense-accent-ink)', flexShrink: 0 }}>
    <Icon name="settings" size={size * 0.55} />
  </div>
);

// ─── Score ring ───────────────────────────────────────────────────────────────
const TONE_VAR: Record<string, string> = {
  ok: 'var(--ssense-accent-emerald)', warn: 'var(--ssense-accent-amber)', bad: 'var(--ssense-accent-rose)',
  info: 'var(--ssense-info)', muted: 'var(--ssense-text-muted)',
};

export const ScoreRing: React.FC<{ score: number | null; size?: number; scanning?: boolean }> = ({ score, size = 64, scanning }) => {
  const stroke = Math.max(4, Math.round(size * 0.085));
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const tone = scanning ? 'info' : scoreTone(score);
  const pct = scanning ? 0.28 : typeof score === 'number' ? Math.max(0.02, Math.min(1, score / 100)) : 0;
  return (
    <div className="sx-ring" style={{ width: size, height: size }} role="img" aria-label={typeof score === 'number' ? `Trust score ${score} out of 100` : 'No score yet'}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className={scanning ? 'sx-ring-spin' : undefined}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--ssense-border-strong)" strokeWidth={stroke} strokeDasharray="1.5 4" opacity={0.7} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={TONE_VAR[tone]} strokeWidth={stroke} strokeLinecap="round"
          strokeDasharray={`${c * pct} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} style={{ transition: 'stroke-dasharray .6s ease' }} />
      </svg>
      {!scanning && (
        <div className="sx-ring-num" style={{ fontSize: size * 0.36, color: typeof score === 'number' ? TONE_VAR[tone] : 'var(--ssense-text-muted)' }}>
          {typeof score === 'number' ? score : '–'}
        </div>
      )}
    </div>
  );
};

export const StatusPill: React.FC<{ status: SiteStatus; label?: string }> = ({ status, label }) => {
  const m = STATUS_META[status];
  return <span className={`sx-pill sx-tone-${m.tone}`} title={m.hint}><i />{label ?? m.label}</span>;
};

export const DomainTile: React.FC<{ domain: string }> = ({ domain }) => (
  <div className="sx-tile" aria-hidden="true">{(domain || '?').charAt(0).toUpperCase()}</div>
);

export const Avatar: React.FC<{ name?: string; email?: string; url?: string; size?: number }> = ({ name, email, url, size = 30 }) => {
  const [broken, setBroken] = useState(false);
  const initial = (name || email || 'S').trim().charAt(0).toUpperCase();
  return (
    <div style={{ width: size, height: size, borderRadius: '50%', overflow: 'hidden', flexShrink: 0, display: 'grid', placeItems: 'center', background: 'var(--ssense-gradient-ai)', color: 'var(--ssense-accent-ink)', font: `700 ${size * 0.42}px var(--ssense-font-display)` }}>
      {url && !broken ? <img src={url} alt="" referrerPolicy="no-referrer" onError={() => setBroken(true)} style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : initial}
    </div>
  );
};

// ─── Collapsible ──────────────────────────────────────────────────────────────
export const Collapsible: React.FC<{
  open: boolean; onToggle: () => void; header: React.ReactNode; children: React.ReactNode;
  id?: string; className?: string; headClassName?: string; style?: React.CSSProperties;
}> = ({ open, onToggle, header, children, id, className, headClassName, style }) => {
  const bodyId = useRef(`c-${Math.random().toString(36).slice(2, 8)}`).current;
  return (
    <div className={`sx-coll ${className || ''}`} data-open={open} id={id} style={style}>
      <button type="button" className={`sx-coll-head ${headClassName || ''}`} onClick={onToggle} aria-expanded={open} aria-controls={bodyId}>
        <Icon name="chevron" size={15} className="sx-chev" />
        {header}
      </button>
      <div className="sx-coll-body" id={bodyId} role="region" hidden={undefined} aria-hidden={!open}>
        <div className="sx-coll-inner" {...(!open ? { inert: '' as any } : {})}>{children}</div>
      </div>
    </div>
  );
};

/** Set of open ids with toggle / expandAll / collapseAll helpers. */
export function useOpenSet(initial: string[] = []) {
  const [open, setOpen] = useState<Set<string>>(new Set(initial));
  return {
    open,
    isOpen: (k: string) => open.has(k),
    toggle: (k: string) => setOpen((p) => { const n = new Set(p); n.has(k) ? n.delete(k) : n.add(k); return n; }),
    add: (k: string) => setOpen((p) => new Set(p).add(k)),
    set: (keys: string[]) => setOpen(new Set(keys)),
    clear: () => setOpen(new Set()),
  };
}

// ─── Violations ───────────────────────────────────────────────────────────────
const SEV_TONE: Record<Severity, string> = { high: 'bad', medium: 'warn', low: 'info' };

export const ViolationCard: React.FC<{
  v: Violation; compact?: boolean; onHighlight?: (quote: string) => void; defaultOpen?: boolean;
}> = ({ v, compact, onHighlight, defaultOpen }) => {
  const [open, setOpen] = useState(Boolean(defaultOpen));
  const sev = severityOf(v);
  return (
    <div className="sx-card sx-coll" data-open={open} style={{ display: 'flex', gap: 10, padding: '9px 11px 9px 9px', background: 'var(--ssense-bg-elevated)' }}>
      <span className={`sx-sev sx-sev--${sev}`} aria-hidden />
      <div style={{ flex: 1, minWidth: 0 }}>
        <button type="button" className="sx-coll-head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: compact ? 12 : 13, fontWeight: 650, lineHeight: 1.3 }}>{prettyViolationType(v.violation_type)}</div>
            <div style={{ display: 'flex', gap: 6, marginTop: 5, flexWrap: 'wrap', alignItems: 'center' }}>
              {v.statute_reference && <span className="sx-stamp" title="DPDP Act / Rules reference">§ {v.statute_reference.replace(/^section\s*/i, '')}</span>}
              <span className={`sx-pill sx-tone-${SEV_TONE[sev]}`} style={{ padding: '1px 8px', fontSize: 10.5 }}>{prettyAction(v.network_action)}</span>
            </div>
          </div>
          <Icon name="chevron" size={14} className="sx-chev" />
        </button>
        <div className="sx-coll-body"><div className="sx-coll-inner">
          <div style={{ display: 'grid', gap: 9, paddingTop: 10 }}>
            {v.evidence_quote && (
              <blockquote className="sx-quote">
                “{v.evidence_quote}”
                {onHighlight && (
                  <div><button type="button" className="sx-btn sx-btn--ghost sx-btn--sm" style={{ marginTop: 7, padding: '4px 8px', fontFamily: 'var(--ssense-font-ui)', fontStyle: 'normal' }} onClick={() => onHighlight(v.evidence_quote)}>
                    <Icon name="search" size={12} /> Show on page
                  </button></div>
                )}
              </blockquote>
            )}
            {v.step_3_semantic_justification && <p style={{ margin: 0, fontSize: 12, lineHeight: 1.55, color: 'var(--ssense-text-secondary)' }}>{v.step_3_semantic_justification}</p>}
            {v.offending_entities?.length > 0 && (
              <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                {v.offending_entities.map((e, i) => <span key={i} className="sx-tag">{e}</span>)}
              </div>
            )}
          </div>
        </div></div>
      </div>
    </div>
  );
};

/** Violations grouped by impact; each group is itself collapsible. */
export const ViolationGroups: React.FC<{
  violations: Violation[]; onHighlight?: (q: string) => void; compact?: boolean; defaultOpenHigh?: boolean;
}> = ({ violations, onHighlight, compact, defaultOpenHigh = true }) => {
  const groups = groupBySeverity(violations);
  const openSet = useOpenSet(defaultOpenHigh ? ['high'] : []);
  if (!violations.length) {
    return (
      <div style={{ display: 'flex', gap: 9, alignItems: 'center', padding: '10px 12px', borderRadius: 10, background: 'var(--ssense-ok-soft)', color: 'var(--ssense-accent-emerald)', fontSize: 12.5, fontWeight: 600 }}>
        <Icon name="check" size={16} /> No DPDP violations found in this policy.
      </div>
    );
  }
  return (
    <div style={{ display: 'grid', gap: 8 }}>
      {SEVERITY_ORDER.filter((s) => groups[s].length).map((s) => (
        <Collapsible key={s} open={openSet.isOpen(s)} onToggle={() => openSet.toggle(s)}
          header={
            <span style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 0 }}>
              <span className={`sx-pill sx-tone-${SEV_TONE[s]}`}><i />{SEVERITY_LABEL[s]}</span>
              <span className="sx-muted" style={{ fontSize: 12, fontWeight: 600 }}>{groups[s].length}</span>
            </span>
          }>
          <div style={{ display: 'grid', gap: 7, paddingTop: 8 }}>
            {groups[s].map((v, i) => <ViolationCard key={`${v.violation_type}-${i}`} v={v} compact={compact} onHighlight={onHighlight} />)}
          </div>
        </Collapsible>
      ))}
    </div>
  );
};

export const Spinner: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" className="sx-ring-spin" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" aria-label="Loading">
    <circle cx="12" cy="12" r="9" opacity=".25" /><path d="M21 12a9 9 0 0 0-9-9" />
  </svg>
);

export const Toast: React.FC<{ message: string | null }> = ({ message }) => (message ? <div className="sx-toast" role="status">{message}</div> : null);
export function useToast(ms = 2200) {
  const [msg, setMsg] = useState<string | null>(null);
  const t = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(t.current), []);
  return { msg, show: (m: string) => { setMsg(m); window.clearTimeout(t.current); t.current = window.setTimeout(() => setMsg(null), ms); } };
}

export const Switch: React.FC<{ checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean }> = ({ checked, onChange, label, disabled }) => (
  <label className="sx-switch"><input type="checkbox" role="switch" aria-label={label} checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} /><span /></label>
);

/** Two-step destructive button — replaces window.confirm(), which extension popups often suppress. */
export const ConfirmButton: React.FC<{ label: string; confirmLabel?: string; onConfirm: () => void | Promise<void>; disabled?: boolean; icon?: string; size?: 'sm' }> = ({ label, confirmLabel = 'Confirm', onConfirm, disabled, icon, size }) => {
  const [armed, setArmed] = useState(false);
  useEffect(() => { if (!armed) return; const t = setTimeout(() => setArmed(false), 4000); return () => clearTimeout(t); }, [armed]);
  return (
    <button type="button" className={`sx-btn sx-btn--danger ${size === 'sm' ? 'sx-btn--sm' : ''}`} disabled={disabled}
      onClick={() => { if (armed) { setArmed(false); void onConfirm(); } else setArmed(true); }}>
      {icon && <Icon name={icon} size={14} />}{armed ? `${confirmLabel}?` : label}
    </button>
  );
};
