// apps/extension/src/sidebar/components/ChatInterface.tsx

import React, { useState, useEffect, useLayoutEffect, useRef, useCallback } from 'react';
import { Icon } from '../../ui/components';
import type { AuditReport, RateLimitInfo } from '../../types/server-protocol';
import { normaliseDomain } from '../../utils/domain';

function formatDuration(ms: number): string {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

// ═══════════════════════════════════════════════════════════════
// DESIGN SYSTEM
// ═══════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════
// DESIGN SYSTEM
// ═══════════════════════════════════════════════════════════════
export const DESIGN_SYSTEM_CSS = `
  /* Tokens + fonts now come from ../../ui/ui.css (shared design system). */

  * { box-sizing: border-box; }

  .ssense-root {
    font-family: var(--ssense-font-ui);
    background: var(--ssense-bg-deep);
    color: var(--ssense-text-primary);
    height: 100vh;
    height: 100dvh;
    max-height: 100dvh;
    width: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
    -webkit-font-smoothing: antialiased;
    transition: background-color 0.2s ease, color 0.2s ease;
  }

  @media (min-width: 768px) {
    .ssense-header-top,
    .ssense-toolbar,
    .ssense-audit-card,
    .ssense-stream,
    .ssense-input-dock,
    .ssense-input-box {
      max-width: 1400px;
      margin-left: auto;
      margin-right: auto;
      width: 100%;
    }
  }

  .ssense-scroll::-webkit-scrollbar { width: 6px; }
  .ssense-scroll::-webkit-scrollbar-track { background: transparent; }
  .ssense-scroll::-webkit-scrollbar-thumb { background: rgba(120,120,128,0.22); border-radius: 3px; }
  .ssense-scroll::-webkit-scrollbar-thumb:hover { background: rgba(120,120,128,0.38); }

  @keyframes ssense-fade-in-up { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  @keyframes ssense-pulse { 0%,100%{opacity:.4;transform:scale(.8)} 50%{opacity:1;transform:scale(1.2)} }
  .ssense-animate-in { animation: ssense-fade-in-up 0.25s cubic-bezier(0.16,1,0.3,1) forwards; }
  .ssense-gradient-text { background: var(--ssense-gradient-ai); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .ssense-thinking-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--ssense-accent-cyan); animation: ssense-pulse 1.4s infinite ease-in-out; }

  .ssense-header {
    padding: 12px 16px 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
    border-bottom: 1px solid var(--ssense-border);
    z-index: 10;
    position: relative;
    background: var(--ssense-header-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
  }
  .ssense-header-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .ssense-header-left { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
  .ssense-header-icon {
    width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0;
    background: var(--ssense-gradient-ai);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2px 10px rgba(6,182,212,0.3);
  }
  .ssense-header-info { flex: 1; min-width: 0; }
  .ssense-domain { font-size: 13px; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ssense-badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 12px; background: var(--ssense-bg-elevated); border: 1px solid var(--ssense-border); margin-top: 4px; }
  .ssense-badge-dot { width: 6px; height: 6px; border-radius: 50%; }

  .ssense-toolbar {
    display: flex;
    align-items: center;
    gap: 6px;
    padding-bottom: 10px;
    overflow-x: auto;
    scrollbar-width: none;
    -ms-overflow-style: none;
    flex-wrap: nowrap;
  }
  .ssense-toolbar::-webkit-scrollbar { display: none; }
  .ssense-toolbar-btn {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--ssense-bg-elevated);
    border: 1px solid var(--ssense-border);
    color: var(--ssense-text-secondary);
    font-size: 11px; font-weight: 500; padding: 5px 9px;
    border-radius: 7px; cursor: pointer;
    transition: background .15s, color .15s, border-color .15s;
    white-space: nowrap; flex-shrink: 0;
  }
  .ssense-toolbar-btn:hover {
    background: var(--ssense-bg-surface);
    color: var(--ssense-text-primary);
    border-color: var(--ssense-border-strong);
  }
  .ssense-toolbar-btn:disabled { opacity: .4; cursor: default; }
  .ssense-toolbar-btn--active {
    background: rgba(6,182,212,0.16);
    color: var(--ssense-accent-cyan);
    border-color: rgba(6,182,212,0.35);
  }
  .ssense-toolbar-spacer { flex: 1 1 auto; min-width: 4px; }

  .ssense-audit-card {
    margin: 14px 16px 0;
    border: 1px solid var(--ssense-border);
    border-radius: 12px;
    background: var(--ssense-bg-surface);
    box-shadow: var(--ssense-shadow-ambient);
    overflow: hidden;
    flex-shrink: 0;
    z-index: 10;
    position: relative;
  }
  .ssense-audit-header {
    padding: 12px 14px;
    display: flex; align-items: center; justify-content: space-between;
    cursor: pointer; user-select: none;
    transition: background 0.15s ease;
  }
  .ssense-audit-header:hover { background: var(--ssense-bg-elevated); }
  .ssense-audit-body {
    padding: 0 14px 14px;
    border-top: 1px solid var(--ssense-border);
    animation: ssense-fade-in-up .2s ease;
    display: flex; flex-direction: column; gap: 12px;
    max-height: min(40vh, 280px);
    overflow-y: auto;
  }
  .ssense-audit-body::-webkit-scrollbar { width: 4px; }
  .ssense-audit-body::-webkit-scrollbar-thumb { background: rgba(120,120,128,0.25); border-radius: 2px; }
  .ssense-audit-reasoning { font-size: 12px; line-height: 1.55; color: var(--ssense-text-secondary); margin: 10px 0 0; font-style: italic; }

  .ssense-violation-card {
    background: var(--ssense-bg-elevated);
    border: 1px solid var(--ssense-border);
    border-radius: 10px; padding: 12px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .ssense-violation-top { display: flex; justify-content: space-between; align-items: center; }
  .ssense-violation-type {
    color: var(--ssense-accent-rose); font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .05em; padding: 3px 7px;
    background: rgba(244,63,94,0.12); border-radius: 6px;
  }
  .ssense-violation-action { color: var(--ssense-text-secondary); font-size: 10.5px; font-weight: 600; }
  .ssense-evidence {
    margin: 0; padding: 8px 12px; border-left: 2px solid var(--ssense-accent-rose);
    background: rgba(244,63,94,0.05); border-radius: 0 6px 6px 0;
    font-size: 11.5px; line-height: 1.5; font-style: italic; cursor: pointer;
  }
  .ssense-entities-list { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }
  .ssense-entity-tag {
    font-size: 10px; font-family: var(--ssense-font-mono); padding: 2px 6px;
    border-radius: 4px; background: var(--ssense-bg-surface);
    color: var(--ssense-text-secondary); border: 1px solid var(--ssense-border);
  }

  .ssense-stream {
    flex: 1; overflow-y: auto; padding: 20px 16px;
    display: flex; flex-direction: column; gap: 20px;
    z-index: 10; position: relative;
  }
  .ssense-empty-state { text-align: center; margin-top: 10%; opacity: .95; }
  .ssense-quick-prompts { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 20px; }
  .ssense-quick-prompt {
    padding: 7px 12px; border-radius: 8px; font-size: 11.5px; font-weight: 500;
    color: var(--ssense-text-secondary); cursor: pointer; transition: all .15s;
    white-space: nowrap; border: 1px solid var(--ssense-border);
    background: var(--ssense-bg-surface); font-family: inherit;
  }
  .ssense-quick-prompt:hover {
    border-color: var(--ssense-accent-cyan);
    color: var(--ssense-text-primary);
    background: rgba(6,182,212,0.06);
  }

  .ssense-msg { display: flex; max-width: 100%; }
  .ssense-msg-user { justify-content: flex-end; }
  .ssense-msg-bubble {
    padding: 10px 14px; font-size: 13px; line-height: 1.6;
    max-width: 88%; word-break: break-word;
  }
  .ssense-msg-bubble.user {
    border-radius: 16px 16px 4px 16px;
    background: var(--ssense-bg-elevated);
    border: 1px solid var(--ssense-border);
  }
  .ssense-msg-bubble.ai {
    border-radius: 16px 16px 16px 4px;
    background: var(--ssense-bg-surface);
    border: 1px solid var(--ssense-border);
    box-shadow: var(--ssense-shadow-ambient);
  }
  .ssense-msg-header {
    display: flex; align-items: center; gap: 6px;
    margin-bottom: 8px; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .05em;
  }
  .ssense-msg-header-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--ssense-accent-cyan); }
  .ssense-inline-code {
    background: rgba(120,120,128,0.15); padding: 2px 5px; border-radius: 4px;
    font-family: var(--ssense-font-mono); font-size: 11.5px; color: var(--ssense-accent-cyan);
  }

  .ssense-input-dock {
    padding: 14px 16px 20px;
    border-top: 1px solid var(--ssense-border);
    z-index: 10; position: relative;
    background: var(--ssense-dock-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
  }
  .ssense-input-container {
    display: flex; align-items: center; padding: 4px 4px 4px 16px;
    border-radius: 14px; border: 1px solid var(--ssense-border);
    background: var(--ssense-bg-surface);
    transition: border-color .2s, box-shadow .2s;
  }
  .ssense-input-container:focus-within {
    border-color: rgba(6,182,212,0.5);
    box-shadow: 0 0 0 2px rgba(6,182,212,0.15);
  }
  .ssense-input-field {
    flex: 1; background: transparent; border: none; outline: none;
    color: var(--ssense-text-primary); font-family: inherit; font-size: 13.5px;
    padding: 10px 0; min-width: 0;
  }
  .ssense-input-field::placeholder { color: var(--ssense-text-muted); }
  .ssense-send-btn {
    width: 34px; height: 34px; border-radius: 10px; border: none;
    flex-shrink: 0; background: transparent; color: var(--ssense-text-muted);
    cursor: not-allowed; display: flex; align-items: center; justify-content: center;
    transition: all .2s; transform: scale(0.9);
  }
  .ssense-send-btn.active {
    background: var(--ssense-gradient-ai); color: #fff;
    cursor: pointer; transform: scale(1);
    box-shadow: 0 2px 8px rgba(6,182,212,0.3);
  }

  .ssense-service-banner {
    border-bottom: 1px solid rgba(244,63,94,0.3); padding: 8px 16px;
    display: flex; align-items: center; gap: 8px; color: var(--ssense-accent-rose);
    font-size: 11.5px; font-weight: 500; z-index: 20; background: rgba(244,63,94,0.08);
  }

  .ssense-shield-panel {
    border-bottom: 1px solid var(--ssense-border); padding: 12px 16px;
    font-size: 11px; display: flex; flex-direction: column; gap: 8px;
    background: var(--ssense-bg-surface); z-index: 10;
  }
  .ssense-shield-row {
    display: flex; justify-content: space-between; align-items: center;
    cursor: pointer; color: var(--ssense-text-secondary);
  }
  .ssense-shield-row input { accent-color: var(--ssense-accent-cyan); }
  .ssense-branding {
    text-align: center; margin-top: 10px; font-size: 10px;
    color: var(--ssense-text-muted); letter-spacing: .03em;
  }

  .ssense-mode-switch {
    display: inline-flex; align-items: center;
    background: var(--ssense-bg-elevated);
    border: 1px solid var(--ssense-border);
    border-radius: 9px; padding: 2px; gap: 2px; flex-shrink: 0;
  }
  .ssense-mode-option {
    display: inline-flex; align-items: center; gap: 4px; border: none;
    background: transparent; color: var(--ssense-text-muted);
    font-size: 10.5px; font-weight: 600; padding: 4px 8px; border-radius: 7px;
    cursor: pointer; transition: background .15s, color .15s; white-space: nowrap;
  }
  .ssense-mode-option:hover { color: var(--ssense-text-secondary); }
  .ssense-mode-option--active {
    background: var(--ssense-bg-surface);
    color: var(--ssense-text-primary);
    box-shadow: 0 1px 3px rgba(0,0,0,0.15);
  }
  .ssense-mode-option--active.ssense-mode-thinking { color: var(--ssense-accent-violet); }
  .ssense-mode-option--active.ssense-mode-concise { color: var(--ssense-accent-cyan); }

  .ssense-quota-chip {
    font-size: 9.5px; font-weight: 600; color: var(--ssense-text-muted);
    padding: 3px 8px; border-radius: 10px; background: var(--ssense-bg-elevated);
    border: 1px solid var(--ssense-border); white-space: nowrap;
  }
  .ssense-quota-chip--low {
    color: var(--ssense-accent-amber);
    background: rgba(245,158,11,0.1);
    border-color: rgba(245,158,11,0.3);
  }

  .ssense-cooldown-banner {
    margin: 10px 16px 0; padding: 10px 12px; border-radius: 10px;
    border: 1px solid rgba(245,158,11,0.25); background: rgba(245,158,11,0.06);
    display: flex; flex-direction: column; gap: 4px;
  }
  .ssense-cooldown-title { display: flex; align-items: center; gap: 8px; font-size: 11.5px; font-weight: 600; color: var(--ssense-accent-amber); }
  .ssense-cooldown-sub { font-size: 10.5px; color: var(--ssense-text-secondary); line-height: 1.5; }

  /* Responsive tweaks for narrow panels (<=360px) and mobile viewports */
  @media (max-width: 360px) {
    .ssense-header { padding: 10px 12px 0; }
    .ssense-toolbar { gap: 4px; padding-bottom: 8px; }
    .ssense-toolbar-btn { font-size: 10px; padding: 4px 7px; }
    .ssense-mode-option { font-size: 10px; padding: 3px 6px; }
    .ssense-stream { padding: 14px 12px; gap: 14px; }
    .ssense-msg-bubble { max-width: 92%; font-size: 12.5px; padding: 8px 12px; }
    .ssense-input-dock { padding: 10px 12px 16px; }
    .ssense-audit-card { margin: 10px 12px 0; }
    .ssense-audit-body { max-height: min(38vh, 250px); }
  }
`;

// ─── Markdown tokenizer (CSP-safe, line-aware, no dangerouslySetInnerHTML) ──
const parseInlineMarkdown = (line: string, lineKey: string | number): React.ReactNode[] => {
  return line.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g).map((part, i) => {
    const key = `${lineKey}-${i}`;
    if (part.startsWith('`') && part.endsWith('`') && part.length >= 2) {
      return <code key={key} className="ssense-inline-code">{part.slice(1, -1)}</code>;
    }
    if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith('*') && part.endsWith('*') && part.length >= 2) {
      return <em key={key}>{part.slice(1, -1)}</em>;
    }
    return <React.Fragment key={key}>{part}</React.Fragment>;
  });
};

const parseMarkdown = (text: string): React.ReactNode[] => {
  const lines = text.split('\n');
  return lines.map((line, idx) => {
    if (!line.trim()) {
      return <div key={idx} style={{ height: 6 }} />;
    }
    if (line.trim() === '---' || line.trim() === '***') {
      return <hr key={idx} style={{ border: 'none', borderTop: '1px solid var(--ssense-border)', margin: '8px 0' }} />;
    }
    if (line.trim().toLowerCase().startsWith('statutory analysis under dpdp act 2023:')) {
      return (
        <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 7, margin: '8px 0 6px', padding: '6px 10px', borderRadius: 7, background: 'rgba(169, 155, 232, 0.1)', border: '1px solid rgba(169, 155, 232, 0.28)', color: 'var(--ssense-accent-violet)', fontSize: 11.5, fontWeight: 700 }}>
          <Icon name="brain" size={14} />
          <span>Statutory Analysis · DPDP Act 2023</span>
        </div>
      );
    }
    if (line.startsWith('### ')) {
      return (
        <div key={idx} style={{ fontSize: 13, fontWeight: 700, margin: '6px 0 2px', color: 'var(--ssense-text-primary)' }}>
          {parseInlineMarkdown(line.slice(4), idx)}
        </div>
      );
    }
    if (line.startsWith('## ')) {
      return (
        <div key={idx} style={{ fontSize: 14, fontWeight: 700, margin: '8px 0 3px', color: 'var(--ssense-text-primary)' }}>
          {parseInlineMarkdown(line.slice(3), idx)}
        </div>
      );
    }
    if (line.startsWith('# ')) {
      return (
        <div key={idx} style={{ fontSize: 15, fontWeight: 800, margin: '10px 0 4px', color: 'var(--ssense-text-primary)' }}>
          {parseInlineMarkdown(line.slice(2), idx)}
        </div>
      );
    }
    if (line.startsWith('> ')) {
      return (
        <blockquote key={idx} style={{ margin: '4px 0', padding: '6px 12px', borderLeft: '2px solid var(--ssense-accent-cyan)', background: 'var(--ssense-accent-soft)', borderRadius: '0 7px 7px 0', fontSize: 12.5, fontStyle: 'italic', color: 'var(--ssense-text-primary)' }}>
          {parseInlineMarkdown(line.slice(2), idx)}
        </blockquote>
      );
    }
    if (/^[\*\-]\s+/.test(line)) {
      const content = line.replace(/^[\*\-]\s+/, '');
      return (
        <div key={idx} style={{ display: 'flex', gap: 6, margin: '2px 0', paddingLeft: 4 }}>
          <span style={{ color: 'var(--ssense-accent-cyan)', fontWeight: 700, flexShrink: 0 }}>•</span>
          <span style={{ flex: 1 }}>{parseInlineMarkdown(content, idx)}</span>
        </div>
      );
    }
    const numMatch = line.match(/^(\d+)\.\s+(.*)$/);
    if (numMatch) {
      return (
        <div key={idx} className="sx-reasoning-trail" style={{ display: 'flex', gap: 8, margin: '4px 0', padding: '6px 10px', borderRadius: 8 }}>
          <span style={{ color: 'var(--ssense-accent-violet)', fontWeight: 700, fontSize: 11.5, minWidth: 16, flexShrink: 0 }}>{numMatch[1]}.</span>
          <span style={{ flex: 1, fontSize: 12.5 }}>{parseInlineMarkdown(numMatch[2], idx)}</span>
        </div>
      );
    }
    return (
      <div key={idx} style={{ margin: '1px 0' }}>
        {parseInlineMarkdown(line, idx)}
      </div>
    );
  });
};

// ─── Sub-components ───────────────────────────────────────────────────────────
const ComplianceBadge = ({ score, delta }: { score: number | null; delta?: number | null }) => {
  if (score === null) return (
    <div className="ssense-badge">
      <div className="ssense-badge-dot" style={{ background: 'var(--ssense-accent-amber)', animation: 'ssense-pulse 1.5s infinite' }} />
      <span style={{ fontSize:11, fontWeight:500, color:'var(--ssense-text-secondary)' }}>Scanning</span>
    </div>
  );
  const color = score >= 80 ? 'var(--ssense-accent-emerald)' : score >= 50 ? 'var(--ssense-accent-amber)' : 'var(--ssense-accent-rose)';
  const label = score >= 80 ? 'Compliant' : score >= 50 ? 'Caution' : 'Violations';
  return (
    <div className="ssense-badge">
      <div className="ssense-badge-dot" style={{ background: color }} />
      <span style={{ fontSize:11, fontWeight:600, color }}>{score}</span>
      <span style={{ fontSize:11, fontWeight:500, color:'var(--ssense-text-muted)' }}>{label}</span>
      {typeof delta === 'number' && delta !== 0 && (
        <span style={{
          fontSize: 10,
          fontWeight: 700,
          color: delta > 0 ? 'var(--ssense-accent-emerald)' : 'var(--ssense-accent-rose)',
          background: delta > 0 ? 'rgba(16,185,129,0.12)' : 'rgba(244,63,94,0.12)',
          padding: '1px 5px',
          borderRadius: 4,
          marginLeft: 2,
        }}>
          {delta > 0 ? `+${delta}` : delta}
        </span>
      )}
    </div>
  );
};

const MessageBubble = React.memo(({ msg }: { msg: { role: 'user' | 'ai'; text: string } }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(msg.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className={`ssense-msg ssense-animate-in ${msg.role === 'user' ? 'ssense-msg-user' : 'ssense-msg-ai'}`}>
      <div className={`ssense-msg-bubble ${msg.role}`} style={{ position: 'relative' }}>
        {msg.role === 'ai' && (
          <div className="ssense-msg-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div className="ssense-msg-header-dot" />
              <span className="ssense-gradient-text" style={{ fontWeight: 700, letterSpacing: '0.02em' }}>Ssense AI</span>
              <span style={{ fontSize: 9.5, opacity: 0.7, letterSpacing: '0.04em', background: 'var(--ssense-bg-elevated)', padding: '1px 6px', borderRadius: 4, border: '1px solid var(--ssense-border)' }}>
                DPDP 2023 Verified
              </span>
            </div>
            <button
              onClick={handleCopy}
              className="sx-icon-btn"
              style={{ width: 22, height: 22, padding: 0 }}
              title={copied ? 'Copied to clipboard' : 'Copy answer'}
            >
              <Icon name={copied ? 'checkCopy' : 'copy'} size={12} style={{ color: copied ? 'var(--ssense-accent-emerald)' : 'var(--ssense-text-muted)' }} />
            </button>
          </div>
        )}
        {msg.role === 'ai' ? parseMarkdown(msg.text) : msg.text}
      </div>
    </div>
  );
});

// ═══════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════
export const ChatInterface: React.FC<{ onOpenHistory?: () => void; onOpenPrivacy?: () => void; onOpenSettings?: () => void }> = ({ onOpenHistory, onOpenPrivacy, onOpenSettings }) => {
  const [domain, setDomain]               = useState<string | null>(null);
  const [isSystemPage, setIsSystemPage]   = useState(false);
  const [trustScore, setTrustScore]       = useState<number | null>(null);
  const [scoreDelta, setScoreDelta]       = useState<number | null>(null);
  const [auditReport, setAuditReport]     = useState<AuditReport | null>(null);
  const [auditError, setAuditError]       = useState('');
  const [showAuditDetails, setShowAuditDetails] = useState(false);
  const [showExplainability, setShowExplainability] = useState(false);
  const [showShield, setShowShield]       = useState(false);
  const [messages, setMessages]           = useState<{ role: 'user' | 'ai'; text: string }[]>([]);
  const [input, setInput]                 = useState(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      return params.get('q') || '';
    } catch {
      return '';
    }
  });
  const [isAuditing, setIsAuditing]       = useState(false);
  // ── Site-thread queue: chat identity is DECOUPLED from "which tab is
  // active". `domain` below still tracks the viewed tab (for the audit
  // panel, which is shared/global and fine to auto-follow). `activeSite`
  // is the one thread the user has explicitly selected for chat — never
  // changed automatically by tab navigation.
  const [siteQueue, setSiteQueue]         = useState<{ domain:string; lastActiveAt:number; pinned:boolean; unread:boolean }[]>([]);
  const [activeSite, setActiveSite]       = useState<string | null>(null);
  const [queueHint, setQueueHint]         = useState<string | null>(null);

  // `chattingFor` names the thread a request is in flight for — NOT a
  // bare boolean. `isChatting` (composer-facing) is derived from it in
  // combination with `activeSite`, so switching threads mid-stream
  // unlocks the composer for the newly-active thread immediately instead
  // of staying locked for a request that belongs to a different site.
  const [chattingFor, setChattingFor]     = useState<string | null>(null);
  const isChatting = chattingFor !== null && chattingFor === activeSite;
  const [responseMode, setResponseMode]   = useState<'concise' | 'thinking'>('concise');
  const [loadingText, setLoadingText]     = useState('Connecting to Ssense AI...');
  const [serviceAvailable, setServiceAvailable] = useState(true);
  const [serviceError, setServiceError]   = useState('');
  const [siteHistory, setSiteHistory]     = useState<any | null>(null);
  const [shieldSettings, setShieldSettings] = useState({ blockTrackers: true, spoofHardware: true, injectGPC: true });
  const [cacheSource, setCacheSource]     = useState<string>('inference');
  const [cacheAgeDays, setCacheAgeDays]   = useState<number>(0);
  const [isOffline, setIsOffline]         = useState(typeof navigator !== 'undefined' ? !navigator.onLine : false);
  const [chatQuota, setChatQuota]         = useState<RateLimitInfo | null>(null);
  const [cooldownUntil, setCooldownUntil] = useState<number | null>(null);
  const [cooldownLeft, setCooldownLeft]   = useState(0);
  const [isMaximized, setIsMaximized]     = useState(false);

  useEffect(() => {
    const updateMaxState = () => {
      if (typeof document !== 'undefined' && document.fullscreenElement) {
        setIsMaximized(true);
        return;
      }
      try {
        chrome.windows?.getCurrent?.().then(w => {
          setIsMaximized(w.state === 'maximized' || w.state === 'fullscreen');
        }).catch(() => {});
      } catch {}
    };

    updateMaxState();
    document.addEventListener('fullscreenchange', updateMaxState);
    window.addEventListener('resize', updateMaxState);
    return () => {
      document.removeEventListener('fullscreenchange', updateMaxState);
      window.removeEventListener('resize', updateMaxState);
    };
  }, []);

  const handleMaximize = async () => {
    if (document.fullscreenElement) {
      try {
        await document.exitFullscreen();
        setIsMaximized(false);
        return;
      } catch {}
    }

    if (window.innerWidth >= 600) {
      try {
        const curr = await chrome.windows.getCurrent();
        if (curr?.id !== undefined) {
          const nextState = (curr.state === 'maximized' || curr.state === 'fullscreen') ? 'normal' : 'maximized';
          await chrome.windows.update(curr.id, { state: nextState });
          setIsMaximized(nextState === 'maximized');
          return;
        }
      } catch {}
      try {
        await document.documentElement.requestFullscreen();
        setIsMaximized(true);
        return;
      } catch {}
    }

    // Open sidepanel in a full tab for maximum space and standard window behavior
    chrome.tabs.create({ url: chrome.runtime.getURL('sidepanel.html') });
  };

  const scrollRef = useRef<HTMLDivElement>(null);
  const currentDomainRef = useRef<string | null>(null);
  const activeSiteRef = useRef<string | null>(null);
  useEffect(() => { activeSiteRef.current = activeSite; }, [activeSite]);

  // ── Load persisted shield settings ──────────────────────────
  useEffect(() => {
    chrome.storage.local.get('ssense_shield_settings').then(d => {
      if (d.ssense_shield_settings) setShieldSettings(d.ssense_shield_settings);
    }).catch(() => {});
  }, []);

  const toggleShield = (key: 'blockTrackers' | 'spoofHardware' | 'injectGPC') => {
    setShieldSettings(prev => {
      const next = { ...prev, [key]: !prev[key] };
      chrome.storage.local.set({ ssense_shield_settings: next }).catch(() => {});
      return next;
    });
  };

  // ── Export report ──────────────────────────────────────────
  const exportAuditReport = () => {
    if (!auditReport || !domain) return;
    const lines = [
      `# Ssense DPDP Audit Report`,
      `**Domain:** \`${domain}\`  |  **Score:** \`${auditReport.dpdp_trust_score}/100\`  |  **Date:** \`${new Date().toISOString()}\``,
      `## Global Reasoning`,
      auditReport.global_legal_reasoning,
      `## Violations (${auditReport.violations.length})`,
      ...auditReport.violations.map((v, i) => [
        `### ${i+1}. ${v.violation_type.replace(/_/g,' ')}`,
        `- **Statute:** ${v.statute_reference}`,
        `- **Action:** \`${v.network_action}\``,
        `- **Evidence:** "${v.evidence_quote}"`,
        v.offending_entities?.length ? `- **Entities:** ${v.offending_entities.join(', ')}` : '',
      ].filter(Boolean).join('\n')),
    ];
    const blob = new Blob([lines.join('\n\n')], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement('a'), { href: url, download: `ssense_${domain.replace(/\W/g,'_')}.md` });
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Adaptive Health check (visibility-aware + 2-min backoff) ─
  useEffect(() => {
    const ping = () => {
      if (document.hidden) return; // Skip if tab/panel is hidden
      chrome.runtime.sendMessage({ type: 'HEALTH_CHECK', requestId: crypto.randomUUID() })
        .then(res => {
          const ok = Boolean(res?.success) && res?.modelLoaded !== false;
          setServiceAvailable(ok);
          setServiceError(ok ? '' : res?.error || 'AI service unavailable. Check Settings.');
        })
        .catch(err => { setServiceAvailable(false); setServiceError(err?.message || 'Cannot reach Ssense server.'); });
    };

    ping();
    const id = setInterval(ping, 120_000); // 2-min backoff: reduces health traffic by 87%

    const onVis = () => { if (!document.hidden) ping(); };
    document.addEventListener('visibilitychange', onVis);

    const onOn = () => setIsOffline(false);
    const onOff = () => setIsOffline(true);
    window.addEventListener('online', onOn);
    window.addEventListener('offline', onOff);

    return () => {
      clearInterval(id);
      document.removeEventListener('visibilitychange', onVis);
      window.removeEventListener('online', onOn);
      window.removeEventListener('offline', onOff);
    };
  }, []);

  // ── Tab tracking & History loading on domain switch ────────
  useEffect(() => {
    const handleUrl = async (url: string | undefined) => {
      if (!url?.startsWith('http')) {
        setIsSystemPage(true); setDomain(null); currentDomainRef.current = null;
        setAuditReport(null); setAuditError('');
        return;
      }
      setIsSystemPage(false);
      const newDomain = new URL(url).hostname;
      if (newDomain === currentDomainRef.current) return;
      setDomain(newDomain); currentDomainRef.current = newDomain;
      setTrustScore(null); setScoreDelta(null); setAuditReport(null); setAuditError('');
      setShowAuditDetails(false); setSiteHistory(null);
      // Note: chat is intentionally NOT touched here. Navigating tabs
      // must never load, clear, or otherwise reach into any chat thread —
      // that only happens via an explicit SELECT_SITE_THREAD action (see
      // the site-queue effect below). This is what makes it structurally
      // impossible for one site's chat to bleed into another's.

      try {
        // Audit + browsing-history metadata still auto-follow the tab —
        // these are global/shared, not per-thread, so that's fine.
        const [localAudit, hist] = await Promise.all([
          chrome.runtime.sendMessage({ type: 'GET_LOCAL_AUDIT_WITH_META', domain: newDomain }),
          chrome.runtime.sendMessage({ type: 'GET_SITE_HISTORY', domain: newDomain }),
        ]);

        if (localAudit?.success && localAudit.entry) {
          const e = localAudit.entry;
          setTrustScore(e.trust_score);
          setAuditReport({
            dpdp_trust_score: e.trust_score,
            subtlety_score: e.subtlety_score,
            violations: e.violations,
            global_legal_reasoning: e.global_legal_reasoning,
          });
          setCacheSource(localAudit.source || e.source || 'local_cache');
          setCacheAgeDays(localAudit.ageDays || e.age_days || 0);
          setShowAuditDetails(e.violation_count > 0);
        }

        if (hist?.success && hist.entry) {
          setSiteHistory(hist.entry);
          const historyArr = hist.entry.scoreHistory || [];
          if (historyArr.length >= 2) {
            const last = historyArr[historyArr.length - 1];
            const prev = historyArr[historyArr.length - 2];
            setScoreDelta(last.score - prev.score);
          }
        }
      } catch { /* non-fatal */ }
    };

    chrome.tabs.query({ active: true, currentWindow: true }, tabs => handleUrl(tabs[0]?.url));
    const onUpdate = (_: number, ci: chrome.tabs.TabChangeInfo, tab: chrome.tabs.Tab) => {
      if (tab.active && (ci.status === 'complete' || ci.url)) handleUrl(tab.url);
    };
    const onActivate = async (info: chrome.tabs.TabActiveInfo) => handleUrl((await chrome.tabs.get(info.tabId)).url);
    const onMsg = (msg: any) => {
      if (msg.type === 'AUDIT_COMPLETE' && msg.domain === currentDomainRef.current) {
        setTrustScore(msg.score); setAuditError(''); setServiceAvailable(true); setServiceError('');
        setIsAuditing(false);
        if (msg.previousScore != null) {
          setScoreDelta(msg.score - msg.previousScore);
        }
        if (msg.source) setCacheSource(msg.source);
        if (msg.report) { setAuditReport(msg.report); setShowAuditDetails(msg.report.violations.length > 0); }
        chrome.runtime.sendMessage({ type: 'GET_SITE_HISTORY', domain: msg.domain })
          .then(r => { if (r?.success) setSiteHistory(r.entry || null); }).catch(() => {});
        // A freshly-audited site auto-joins the pickable queue (unselected)
        // — refresh so it shows up without forcing it active.
        chrome.runtime.sendMessage({ type: 'GET_SITE_QUEUE' })
          .then(r => { if (r?.success) { setSiteQueue(r.queue || []); setActiveSite(r.active ?? null); } }).catch(() => {});
      }
      if (msg.type === 'AUDIT_ERROR' && msg.domain === currentDomainRef.current) {
        setAuditError(msg.error || 'Audit could not be completed.');
        setIsAuditing(false);
      }
    };
    chrome.tabs.onUpdated.addListener(onUpdate);
    chrome.tabs.onActivated.addListener(onActivate);
    chrome.runtime.onMessage.addListener(onMsg);
    return () => {
      chrome.tabs.onUpdated.removeListener(onUpdate);
      chrome.tabs.onActivated.removeListener(onActivate);
      chrome.runtime.onMessage.removeListener(onMsg);
    };
  }, []);

  // ── Site-thread queue: load on mount, refresh after any local mutation ──
  const refreshQueue = useCallback(async () => {
    try {
      const r = await chrome.runtime.sendMessage({ type: 'GET_SITE_QUEUE' });
      if (r?.success) { setSiteQueue(r.queue || []); setActiveSite(r.active ?? null); }
    } catch { /* non-fatal */ }
  }, []);
  useEffect(() => { refreshQueue(); }, [refreshQueue]);

  // Load this thread's own message history whenever the ACTIVE site
  // changes — never when the viewed tab changes.
  useEffect(() => {
    let cancelled = false;
    if (!activeSite) { setMessages([]); return; }
    chrome.runtime.sendMessage({ type: 'GET_CHAT_HISTORY', domain: activeSite })
      .then(r => {
        if (cancelled) return;
        setMessages(r?.success && Array.isArray(r.messages) ? r.messages.map((m: any) => ({ role: m.role, text: m.text })) : []);
      })
      .catch(() => { if (!cancelled) setMessages([]); });
    return () => { cancelled = true; };
  }, [activeSite]);

  const selectThread = useCallback(async (target: string) => {
    setQueueHint(null);
    if (activeSiteRef.current && activeSiteRef.current !== target) {
      setQueueHint(`Deselect "${activeSiteRef.current}" first — only one site can be active at a time.`);
      setTimeout(() => setQueueHint(null), 3500);
      return;
    }
    const r = await chrome.runtime.sendMessage({ type: 'SELECT_SITE_THREAD', domain: target });
    if (r?.success) await refreshQueue();
    else setQueueHint(r?.error || 'Could not select that site.');
  }, [refreshQueue]);

  const deselectThread = useCallback(async () => {
    await chrome.runtime.sendMessage({ type: 'DESELECT_SITE_THREAD' });
    await refreshQueue();
  }, [refreshQueue]);

  const removeThread = useCallback(async (target: string) => {
    await chrome.runtime.sendMessage({ type: 'REMOVE_SITE_THREAD', domain: target });
    await refreshQueue();
  }, [refreshQueue]);

  const togglePinThread = useCallback(async (target: string) => {
    await chrome.runtime.sendMessage({ type: 'TOGGLE_PIN_SITE_THREAD', domain: target });
    await refreshQueue();
  }, [refreshQueue]);

  useLayoutEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, isChatting]);

  useEffect(() => {
    if (!isAuditing && !isChatting) return;
    const stages = isAuditing
      ? ['Scanning page DOM...', 'Checking DPDP compliance cache...', 'Forensic legal analysis...', 'Finalizing report...']
      : ['Connecting to Ssense AI...', 'Retrieving context...', 'Reasoning over DPDP Act...', 'Generating response...'];
    let i = 0; setLoadingText(stages[0]);
    const id = setInterval(() => { i = (i+1) % stages.length; setLoadingText(stages[i]); }, 3000);
    return () => clearInterval(id);
  }, [isAuditing, isChatting]);

  // ── Cooldown countdown (chat rate limit only — audits are unaffected) ──
  useEffect(() => {
    if (!cooldownUntil) { setCooldownLeft(0); return; }
    const tick = () => {
      const left = Math.max(0, Math.ceil((cooldownUntil - Date.now()) / 1000));
      setCooldownLeft(left);
      if (left <= 0) setCooldownUntil(null);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [cooldownUntil]);

  // ── Manual audit: re-inject extractor → finds URL → server does everything ─
  const runAudit = useCallback(async (forceRefresh = false) => {
    if (!domain || isAuditing) return;
    setIsAuditing(true); setAuditError('');
    try {
      if (!forceRefresh) {
        const local = await chrome.runtime.sendMessage({ type: 'GET_LOCAL_AUDIT_WITH_META', domain });
        if (local?.success && local.entry) {
          setTrustScore(local.entry.trust_score);
          setAuditReport({
            dpdp_trust_score: local.entry.trust_score,
            subtlety_score: local.entry.subtlety_score,
            violations: local.entry.violations,
            global_legal_reasoning: local.entry.global_legal_reasoning,
          });
          setCacheSource(local.source || 'local_cache');
          setCacheAgeDays(local.ageDays || 0);
          setShowAuditDetails(local.entry.violation_count > 0);
          setIsAuditing(false);
          return;
        }
      }
      const retry = await chrome.runtime.sendMessage({ type: 'RETRY_EXTRACTION' });
      if (!retry?.success) throw new Error(retry?.error || 'Could not scan this page for a privacy policy link.');
    } catch (err: any) {
      setAuditError(err.message || 'Audit failed.');
      setIsAuditing(false);
    }
  }, [domain, isAuditing]);

  // ── Chat (Streaming relay via port with fallback) ───────────
  // Composes against `activeSite` — the explicitly-selected thread — never
  // against the tab-followed `domain`. The background double-checks this
  // (see service-worker.ts's CHAT/START_CHAT handlers), so even a stale
  // closure here can't misroute a message; worst case the background
  // rejects it with `stale_thread`.
  const livePortRef = useRef<chrome.runtime.Port | null>(null);

  const handleSend = useCallback(async (text?: string) => {
    const prompt = text || input;
    const threadDomain = activeSite;
    if (!prompt.trim() || isChatting || !threadDomain || isSystemPage || !serviceAvailable || cooldownLeft > 0 || isOffline) return;

    // A previous stream should never still be open when a new send fires
    // (isChatting guards this), but belt-and-suspenders: never let two
    // live ports write into the same message list at once.
    if (livePortRef.current) { try { livePortRef.current.disconnect(); } catch {} livePortRef.current = null; }

    setMessages(prev => [...prev, { role: 'user', text: prompt }]);
    setInput('');
    setChattingFor(threadDomain);

    setLoadingText(`Evaluating ${threadDomain} policy records...`);
    const stageTimer1 = setTimeout(() => {
      setLoadingText('Cross-referencing DPDP Act provisions...');
    }, 6000);
    const stageTimer2 = setTimeout(() => {
      setLoadingText(responseMode === 'thinking' ? 'Synthesizing comprehensive legal reasoning...' : 'Formulating concise response...');
    }, 15000);

    const clearStageTimers = () => {
      clearTimeout(stageTimer1);
      clearTimeout(stageTimer2);
    };

    try {
      const port = chrome.runtime.connect({ name: 'ssense-chat-stream' });
      livePortRef.current = port;
      let accumulatedAiText = '';
      let messageAppended = false;

      const isStillTheActiveThread = () => activeSiteRef.current === threadDomain;

      port.onMessage.addListener((streamMsg) => {
        // The user may have deselected/switched threads while this
        // request was in flight. The background still saves the reply
        // under `threadDomain` in storage either way — this check only
        // controls whether it renders into the CURRENTLY VISIBLE thread,
        // which is what prevents siteA's answer from appearing under
        // siteB's conversation.
        if (!isStillTheActiveThread()) {
          if (streamMsg.type === 'DONE' || streamMsg.type === 'ERROR') {
            clearStageTimers();
            setLoadingText('Connecting to Ssense AI...');
            setChattingFor(null);
            if (livePortRef.current === port) livePortRef.current = null;
            port.disconnect();
            refreshQueue(); // picks up the unread flag the background set
          }
          return;
        }
        if (streamMsg.type === 'CHUNK') {
          clearStageTimers();
          accumulatedAiText += streamMsg.delta || '';
          setMessages(prev => {
            if (!messageAppended) {
              messageAppended = true;
              return [...prev, { role: 'ai', text: accumulatedAiText }];
            }
            const updated = [...prev];
            updated[updated.length - 1] = { role: 'ai', text: accumulatedAiText };
            return updated;
          });
        } else if (streamMsg.type === 'DONE') {
          clearStageTimers();
          setLoadingText('Connecting to Ssense AI...');
          if (streamMsg.rateLimit) setChatQuota(streamMsg.rateLimit);
          setChattingFor(null);
          if (livePortRef.current === port) livePortRef.current = null;
          port.disconnect();
        } else if (streamMsg.type === 'ERROR') {
          clearStageTimers();
          setLoadingText('Connecting to Ssense AI...');
          const isRateLimited = streamMsg.errorKind === 'server' && /rate limit/i.test(streamMsg.error || '');
          if (isRateLimited) {
            setCooldownUntil(Date.now() + (streamMsg.rateLimit?.windowSeconds ?? 60) * 1000);
          } else {
            setMessages(prev => [...prev, { role: 'ai', text: `⚠️ ${streamMsg.error || 'Request failed.'}` }]);
          }
          setChattingFor(null);
          if (livePortRef.current === port) livePortRef.current = null;
          port.disconnect();
        }
      });

      // 120s silence guard: allows sufficient headroom for CPU prompt prefill
      // on multi-site queries without premature cutoff. Reset on every message.
      let silenceTimer: ReturnType<typeof setTimeout>;
      const armSilenceTimer = () => {
        clearTimeout(silenceTimer);
        silenceTimer = setTimeout(() => {
          if (livePortRef.current === port) {
            clearStageTimers();
            setLoadingText('Connecting to Ssense AI...');
            setMessages(prev => [...prev, { role: 'ai', text: '⚠️ The response timed out. Please try again.' }]);
            setChattingFor(null);
            livePortRef.current = null;
            try { port.disconnect(); } catch {}
          }
        }, 120_000);
      };
      port.onMessage.addListener(armSilenceTimer);
      armSilenceTimer();
      port.onDisconnect.addListener(() => {
        clearStageTimers();
        clearTimeout(silenceTimer);
        if (livePortRef.current === port) {
          livePortRef.current = null;
          setChattingFor(null);
          setLoadingText('Connecting to Ssense AI...');
        }
      });

      port.postMessage({
        type: 'START_CHAT',
        domain: threadDomain,
        userPrompt: prompt,
        responseMode,
        requestId: crypto.randomUUID(),
      });
    } catch {
      // Fallback to standard message passing
      try {
        const res = await chrome.runtime.sendMessage({ type: 'CHAT', domain: threadDomain, userPrompt: prompt, responseMode });
        if (res?.rateLimit) setChatQuota(res.rateLimit);
        if (activeSiteRef.current === threadDomain) {
          if (res?.success) {
            setMessages(prev => [...prev, { role: 'ai', text: res.message }]);
          } else {
            setMessages(prev => [...prev, { role: 'ai', text: `⚠️ ${res?.error || 'Request failed.'}` }]);
          }
        }
      } catch (err: any) {
        if (activeSiteRef.current === threadDomain) {
          setMessages(prev => [...prev, { role: 'ai', text: `⚠️ ${err?.message || 'Could not reach Ssense AI.'}` }]);
        }
      } finally {
        setChattingFor(null);
      }
    }
  }, [input, isChatting, activeSite, isSystemPage, serviceAvailable, responseMode, cooldownLeft, isOffline, refreshQueue]);

  // Deselecting or switching threads while a stream is live: let the
  // in-flight request finish in the background (it still saves correctly
  // and flags the thread unread) rather than aborting it destructively —
  // but the composer must unlock immediately for the newly-selected
  // thread, so `isChatting` is only ever read together with
  // `activeSite === <the domain the stream was opened for>` (see
  // `isStillTheActiveThread` above), not as a bare global flag.

  const quickPrompts = activeSite
    ? [`Is ${activeSite} selling my data?`, 'Where is my data stored?', 'Explain the data retention policy.']
    : [];

  // Is the tab currently being viewed already a thread the user can chat
  // in? Used to offer "Start chat about this site" instead of silently
  // doing nothing when the viewed tab isn't the active chat thread.
  const viewedIsQueued = domain ? siteQueue.some(s => s.domain === normaliseDomain(domain)) : false;
  const viewedIsActive = domain != null && activeSite === normaliseDomain(domain);

  return (
    <div className="ssense-root">
      {/* Background glow */}
      <div style={{ position:'absolute', top:'-30%', left:'50%', transform:'translateX(-50%)', width:'120%', height:'60%', background:`radial-gradient(circle,${trustScore !== null && trustScore < 50 ? 'rgba(244,63,94,0.06)' : 'rgba(6,182,212,0.04)'} 0%,transparent 70%)`, pointerEvents:'none', zIndex:0, filter:'blur(40px)' }} />

      {/* Service unavailable banner */}
      {!serviceAvailable && (
        <div className="ssense-service-banner" style={{ background: 'rgba(240, 100, 95, 0.09)', borderBottom: '1px solid rgba(240, 100, 95, 0.25)', padding: '9px 16px' }}>
          <Icon name="shieldAlert" size={16} style={{ color: 'var(--ssense-accent-rose)', flexShrink: 0 }} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontWeight: 650, fontSize: 11.5, color: 'var(--ssense-accent-rose)' }}>AI Server Offline · Local Protection Active</div>
            <div style={{ fontSize: 10.5, color: 'var(--ssense-text-secondary)', marginTop: 1 }}>{serviceError || 'Cannot reach Ssense SLM Server. Local tracker blocking remains active.'}</div>
          </div>
          <button onClick={() => onOpenSettings ? onOpenSettings() : chrome.runtime.openOptionsPage()} style={{ marginLeft: 8, background: 'var(--ssense-bg-elevated)', border: '1px solid var(--ssense-border)', color: 'var(--ssense-text-primary)', borderRadius: 6, padding: '4px 9px', fontSize: 11, cursor: 'pointer', flexShrink: 0, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            <Icon name="settings" size={11} />
            <span>Settings</span>
          </button>
        </div>
      )}

      {/* Header */}
      <header className="ssense-header">
        <div className="ssense-header-top">
          <div className="ssense-header-left">
            <div className="ssense-header-icon">
              <Icon name="settings" size={16} style={{ color: 'white', strokeWidth: 2.5 }} />
            </div>
            <div className="ssense-header-info">
              <div className="ssense-domain">{isSystemPage ? 'System Page' : (domain || 'Detecting…')}</div>
              {!isSystemPage && <ComplianceBadge score={trustScore} delta={scoreDelta} />}
            </div>
          </div>
          <button
            className="ssense-toolbar-btn"
            onClick={handleMaximize}
            title={isMaximized ? 'Restore window size' : 'Maximize full screen in Chrome (all edges extendable)'}
            style={{ padding: '4px 9px', fontSize: 11, display: 'inline-flex', alignItems: 'center', gap: 5 }}
          >
            <Icon name={isMaximized ? 'collapse' : 'maximize'} size={13} />
            <span>{isMaximized ? 'Restore' : 'Maximize'}</span>
          </button>
        </div>

        <nav className="ssense-toolbar">
          <button className="ssense-toolbar-btn" onClick={onOpenHistory} title="Past audits and browsing history">
            <Icon name="history" size={13} />
            <span>History</span>
          </button>
          <button className="ssense-toolbar-btn" onClick={onOpenPrivacy} title="View retrieved privacy policy text">
            <Icon name="fileText" size={13} />
            <span>Policy</span>
          </button>
          <button className="ssense-toolbar-btn" onClick={() => onOpenSettings ? onOpenSettings() : chrome.runtime.openOptionsPage()} title="Settings">
            <Icon name="settings" size={13} />
            <span>Settings</span>
          </button>
          <span className="ssense-toolbar-spacer" />
          {chatQuota && (
            <span
              className={`ssense-quota-chip${chatQuota.remaining <= 15 ? ' ssense-quota-chip--low' : ''}`}
              title="Chat messages remaining this minute. Audits are never limited."
              style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
            >
              <Icon name="chat" size={11} />
              <span>{chatQuota.remaining}/{chatQuota.limit}</span>
            </span>
          )}
          <div className="ssense-mode-switch" role="radiogroup" aria-label="Response length">
            <button
              role="radio" aria-checked={responseMode === 'concise'}
              className={`ssense-mode-option ssense-mode-concise${responseMode === 'concise' ? ' ssense-mode-option--active' : ''}`}
              onClick={() => setResponseMode('concise')}
              title="Regular: short, direct answers (<45 words)"
            >
              <Icon name="zap" size={12} />
              <span>Regular</span>
            </button>
            <button
              role="radio" aria-checked={responseMode === 'thinking'}
              className={`ssense-mode-option ssense-mode-thinking${responseMode === 'thinking' ? ' ssense-mode-option--active' : ''}`}
              onClick={() => setResponseMode('thinking')}
              title="Thinking: step-by-step statutory reasoning under DPDP Act"
            >
              <Icon name="brain" size={12} />
              <span>Thinking</span>
            </button>
          </div>
          <button
            className="ssense-toolbar-btn"
            onClick={() => runAudit(true)}
            disabled={!domain || isAuditing || isSystemPage}
            title="Run a fresh audit of this site's privacy policy — audits are never rate-limited"
          >
            <Icon name={isAuditing ? "sync" : "scan"} size={13} className={isAuditing ? "sx-ring-spin" : undefined} />
            <span>{isAuditing ? 'Auditing…' : 'Audit'}</span>
          </button>
          <button
            className={`ssense-toolbar-btn${showShield ? ' ssense-toolbar-btn--active' : ''}`}
            onClick={() => setShowShield(v => !v)}
            title="Active protection controls"
          >
            <Icon name="shield" size={13} />
            <span>Shield</span>
          </button>
        </nav>
      </header>

      {/* Site-thread queue — one consistent memory per user, tagged per
          site, queued rather than overwritten. Only one thread can be the
          active/selected one; switching requires explicitly deselecting
          the current thread first (enforced both here and in the
          background — see selectThread / service-worker.ts). */}
      {siteQueue.length > 0 && (
        <div style={{ display:'flex', flexWrap:'wrap', gap:6, alignItems:'center', padding:'10px 20px 0' }}>
          {siteQueue.map(s => {
            const active = s.domain === activeSite;
            return (
              <div
                key={s.domain}
                onClick={() => (active ? undefined : selectThread(s.domain))}
                title={active ? 'Active thread — click ✕ to deselect' : (activeSite ? `Deselect "${activeSite}" first` : `Select ${s.domain}`)}
                style={{
                  display:'flex', alignItems:'center', gap:6, padding:'5px 9px', borderRadius:8,
                  fontSize:11, fontWeight:600, cursor: active ? 'default' : 'pointer',
                  background: active ? 'rgba(63, 203, 176, 0.16)' : 'var(--ssense-bg-elevated)',
                  color: active ? 'var(--ssense-accent-cyan)' : 'var(--ssense-text-secondary)',
                  border: `1px solid ${active ? 'rgba(63, 203, 176, 0.35)' : 'var(--ssense-border)'}`,
                  opacity: !active && activeSite ? 0.6 : 1,
                  maxWidth: 160,
                }}
              >
                {s.pinned && <Icon name="pin" size={10} style={{ color: 'var(--ssense-accent-cyan)' }} />}
                {s.unread && !active && <span style={{ width:6, height:6, borderRadius:'50%', background:'var(--ssense-accent-rose)', flexShrink:0 }} />}
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.domain}</span>
                <span
                  onClick={e => { e.stopPropagation(); togglePinThread(s.domain); }}
                  style={{ opacity:0.6, display: 'inline-flex', alignItems: 'center', cursor:'pointer' }}
                  title={s.pinned ? 'Unpin' : 'Pin'}
                >
                  <Icon name={s.pinned ? "pinOff" : "pin"} size={10} />
                </span>
                {active ? (
                  <span onClick={e => { e.stopPropagation(); deselectThread(); }} style={{ opacity:0.8, cursor:'pointer', fontWeight:700 }} title="Deselect">✕</span>
                ) : (
                  <span onClick={e => { e.stopPropagation(); removeThread(s.domain); }} style={{ opacity:0.4, display: 'inline-flex', alignItems: 'center', cursor:'pointer' }} title="Remove thread">
                    <Icon name="trash" size={11} />
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
      {queueHint && (
        <div style={{ margin:'8px 20px 0', padding:'8px 12px', borderRadius:8, background:'rgba(245,158,11,0.08)', border:'1px solid rgba(245,158,11,0.25)', color:'var(--ssense-accent-amber)', fontSize:11, fontWeight:500 }}>
          {queueHint}
        </div>
      )}

      {/* Offer to start/switch to a thread for the tab actually being viewed,
          when it isn't already the active chat thread — explicit action,
          never automatic. */}
      {!isSystemPage && domain && !viewedIsActive && (
        <div style={{ margin:'8px 20px 0', display:'flex', alignItems:'center', gap:8 }}>
          <button
            onClick={() => selectThread(domain)}
            style={{ fontSize:11, fontWeight:600, padding:'6px 10px', borderRadius:8, border:'1px solid var(--ssense-border)', background:'var(--ssense-bg-elevated)', color:'var(--ssense-text-secondary)', cursor:'pointer' }}
          >
            {viewedIsQueued ? `↳ Switch chat to ${domain}` : `＋ Start chat about ${domain}`}
          </button>
          {activeSite && <span style={{ fontSize:10.5, color:'var(--ssense-text-muted)' }}>Currently chatting about {activeSite}</span>}
        </div>
      )}

      {/* Chat cooldown — chat only; audits above remain fully available */}
      {cooldownLeft > 0 && (
        <div className="ssense-cooldown-banner ssense-animate-in">
          <div className="ssense-cooldown-title"><span>⏳</span><span>Chat is briefly cooling down — back in {cooldownLeft}s</span></div>
          <div className="ssense-cooldown-sub">
            You've reached the chat message limit for this minute. This only affects chat — running an Audit above works as normal.
          </div>
        </div>
      )}

      {/* Shield panel */}
      {showShield && (
        <div className="ssense-shield-panel">
          <div style={{ display:'flex', justifyContent:'space-between', fontWeight:600, color:'var(--ssense-text-primary)' }}>
            <span>Active Protection</span>
            <span style={{ fontSize:10, color:'var(--ssense-accent-cyan)', cursor:'pointer' }} onClick={() => setShowShield(false)}>✕</span>
          </div>
          {([
            ['blockTrackers',  'Block Third-Party Trackers'],
            ['spoofHardware',  'Spoof Hardware APIs (Canvas / Audio)'],
            ['injectGPC',      'Inject Global Privacy Control (GPC)'],
          ] as const).map(([key, label]) => (
            <label key={key} className="ssense-shield-row">
              <span>{label}</span>
              <input type="checkbox" checked={shieldSettings[key]} onChange={() => toggleShield(key)} />
            </label>
          ))}
        </div>
      )}

      {/* Audit error */}
      {auditError && !isSystemPage && (
        <div style={{ margin:'10px 20px 0', padding:'10px 12px', borderRadius:9, border:'1px solid rgba(245,158,11,.25)', background:'rgba(245,158,11,.07)', color:'var(--ssense-accent-amber)', fontSize:10.5, lineHeight:1.5 }}>
          <strong>Audit unavailable.</strong> {auditError}
          <div style={{ marginTop:6, display:'flex', gap:8 }}>
            <button onClick={() => runAudit(false)} disabled={isAuditing} style={{ border:'1px solid rgba(245,158,11,.3)', background:'transparent', color:'var(--ssense-accent-amber)', borderRadius:6, padding:'4px 8px', fontSize:10, cursor:'pointer' }}>Retry</button>
            <button onClick={() => chrome.runtime.sendMessage({ type: 'RETRY_EXTRACTION' })} style={{ border:'1px solid rgba(245,158,11,.2)', background:'transparent', color:'var(--ssense-text-muted)', borderRadius:6, padding:'4px 8px', fontSize:10, cursor:'pointer' }}>Re-scan page</button>
          </div>
        </div>
      )}

      {/* Audit card */}
      {auditReport && !isSystemPage && (
        <div className={`ssense-audit-card ${auditReport.dpdp_trust_score >= 80 ? 'sx-glow-emerald' : auditReport.dpdp_trust_score >= 50 ? 'sx-glow-amber' : 'sx-glow-rose'}`}>
          <div className="ssense-audit-header" onClick={() => setShowAuditDetails(v => !v)}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Icon name={auditReport.violations.length === 0 ? "shieldCheck" : "shieldAlert"} size={16} style={{ color: auditReport.violations.length === 0 ? 'var(--ssense-accent-emerald)' : 'var(--ssense-accent-rose)' }} />
              <span className="ssense-gradient-text" style={{ fontWeight: 700, fontSize: 12.5 }}>
                {auditReport.violations.length === 0 ? 'DPDP Statutory Compliant' : `${auditReport.violations.length} Statutory Violation${auditReport.violations.length !== 1 ? 's' : ''} Detected`}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--ssense-text-secondary)', background: 'var(--ssense-bg-elevated)', padding: '2px 7px', borderRadius: 6, border: '1px solid var(--ssense-border)' }}>
                {auditReport.dpdp_trust_score}/100
              </span>
              <Icon name="chevron" size={14} style={{ transform: showAuditDetails ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s', color: 'var(--ssense-text-muted)' }} />
            </div>
          </div>

          {showAuditDetails && (
            <div className="ssense-audit-body">
              {/* Cache & Offline metadata badges */}
              {cacheSource === 'offline_cache' && (
                <div style={{ fontSize:10, color:'var(--ssense-accent-amber)', padding:'4px 8px', background:'rgba(245,158,11,0.08)', borderRadius:6, border:'1px solid rgba(245,158,11,0.2)', display: 'flex', alignItems: 'center', gap: 5 }}>
                  <Icon name="cloudOff" size={12} />
                  <span>Offline Mode — Displaying cached audit ({cacheAgeDays}d old)</span>
                </div>
              )}
              {cacheSource === 'persistent_cache' && cacheAgeDays > 0 && (
                <div style={{ fontSize:10, color:'var(--ssense-text-muted)', padding:'2px 4px', display: 'flex', alignItems: 'center', gap: 4 }}>
                  <Icon name="history" size={11} />
                  <span>From cache · {cacheAgeDays} day{cacheAgeDays !== 1 ? 's' : ''} ago</span>
                </div>
              )}

              {/* Score row */}
              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', background:'var(--ssense-bg-elevated)', padding:8, borderRadius:6, border:'1px solid var(--ssense-border)' }}>
                <div><div style={{ fontSize:10, color:'var(--ssense-text-muted)' }}>Trust Score</div><div style={{ fontSize:14, fontWeight:700, color:'var(--ssense-accent-cyan)' }}>{auditReport.dpdp_trust_score} / 100</div></div>
                <div><div style={{ fontSize:10, color:'var(--ssense-text-muted)' }}>Subtlety</div><div style={{ fontSize:14, fontWeight:700, color:'var(--ssense-accent-violet)' }} title="Legal obfuscation score — higher means more complex evasive language">{auditReport.subtlety_score} / 100</div></div>
                <button onClick={e => { e.stopPropagation(); exportAuditReport(); }} style={{ background:'var(--ssense-gradient-ai)', border:'none', color:'#fff', fontWeight:600, fontSize:10, padding:'4px 10px', borderRadius:6, cursor:'pointer', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <Icon name="download" size={11} />
                  <span>Export</span>
                </button>
              </div>

              <p className="ssense-audit-reasoning">{auditReport.global_legal_reasoning}</p>

              {/* Explainability / history accordion */}
              {(auditReport.explainability || siteHistory) && (
                <div style={{ border:'1px solid var(--ssense-border)', borderRadius:8, padding:9, background:'rgba(6,182,212,0.025)' }}>
                  <button onClick={e => { e.stopPropagation(); setShowExplainability(v => !v); }} style={{ width:'100%', background:'transparent', border:0, color:'var(--ssense-text-primary)', cursor:'pointer', display:'flex', justifyContent:'space-between', fontSize:10.5, fontWeight:700, padding:0 }}>
                    <span>Why this score?</span><span>{showExplainability ? '−' : '+'}</span>
                  </button>
                  {showExplainability && (
                    <div style={{ marginTop:9 }}>
                      {siteHistory ? (
                        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:7 }}>
                          {([['Current score', siteHistory.lastScore != null ? `${siteHistory.lastScore}/100` : '—'],['Visits', String(siteHistory.visitCount)],['Time on site', formatDuration(siteHistory.totalTimeMs)],['Violations', String(siteHistory.lastReport?.violations?.length ?? 0)],['Last audit', siteHistory.lastAuditAt ? new Date(siteHistory.lastAuditAt).toLocaleString() : 'Never'],['First seen', new Date(siteHistory.firstVisit).toLocaleDateString()]] as [string,string][]).map(([label, val]) => (
                            <div key={label} style={{ padding:7, borderRadius:6, background:'rgba(255,255,255,0.035)', border:'1px solid rgba(255,255,255,0.05)' }}>
                              <div style={{ fontSize:8.5, color:'var(--ssense-text-muted)' }}>{label}</div>
                              <div style={{ fontSize:10, fontWeight:650, marginTop:2 }}>{val}</div>
                            </div>
                          ))}
                        </div>
                      ) : <div style={{ fontSize:9.5, color:'var(--ssense-text-muted)' }}>No local history for this site yet.</div>}
                    </div>
                  )}
                </div>
              )}

              {/* Violations */}
              {auditReport.violations.map((v, i) => (
                <div key={i} className="ssense-violation-card" style={{ borderLeft: `3px solid ${v.network_action === 'BLOCK_THIRD_PARTY' ? 'var(--ssense-accent-rose)' : 'var(--ssense-accent-amber)'}` }}>
                  <div className="ssense-violation-top" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                      <span className="ssense-violation-type">{v.violation_type.replace(/_/g,' ')}</span>
                      {v.statute_reference && (
                        <span className="sx-badge-statute" title="DPDP Statutory Reference">
                          § {v.statute_reference}
                        </span>
                      )}
                    </div>
                    <span className={`sx-action-badge ${v.network_action === 'BLOCK_THIRD_PARTY' ? 'sx-action-badge--block' : 'sx-action-badge--warn'}`}>
                      {v.network_action === 'BLOCK_THIRD_PARTY' ? 'BLOCKED' : 'WARNING'}
                    </span>
                  </div>
                  {v.evidence_quote && (
                    <blockquote className="ssense-evidence" title="Click to highlight in page" onClick={() => chrome.tabs.query({ active:true, currentWindow:true }, tabs => { if (tabs[0]?.id) chrome.tabs.sendMessage(tabs[0].id, { type:'HIGHLIGHT_IN_DOM', quote:v.evidence_quote }).catch(()=>{}); })}>
                      "{v.evidence_quote}"
                      <div style={{ fontSize:10, color:'var(--ssense-accent-cyan)', marginTop:5, fontStyle:'normal', display: 'flex', alignItems: 'center', gap: 4, fontWeight: 600 }}>
                        <Icon name="search" size={11} />
                        <span>Click to highlight in page</span>
                      </div>
                    </blockquote>
                  )}
                  {v.offending_entities?.length > 0 && (
                    <div className="ssense-entities-list">
                      {v.offending_entities.map((e, j) => <span key={j} className="ssense-entity-tag">{e}</span>)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Message stream */}
      <div ref={scrollRef} className="ssense-stream">
        {isSystemPage
          ? <div className="ssense-empty-state" style={{ marginTop:'30%', color:'var(--ssense-text-muted)' }}>Ssense AI is disabled on browser system pages.</div>
          : <>
              {!activeSite && (
                <div className="ssense-empty-state">
                  <h2 className="ssense-gradient-text" style={{ fontSize:22, fontWeight:700, margin:0, letterSpacing:'-0.02em' }}>Ssense Co-Pilot</h2>
                  <p style={{ color:'var(--ssense-text-secondary)', fontSize:13, marginTop:8, lineHeight:1.5 }}>
                    {siteQueue.length > 0
                      ? 'Select a site above to continue that conversation.'
                      : domain ? `Start a chat about ${domain} to begin.` : 'Visit a site to start a privacy chat.'}
                  </p>
                </div>
              )}
              {activeSite && messages.length === 0 && !isChatting && (
                <div className="ssense-empty-state">
                  <h2 className="ssense-gradient-text" style={{ fontSize:22, fontWeight:700, margin:0, letterSpacing:'-0.02em' }}>Ssense Co-Pilot</h2>
                  <p style={{ color:'var(--ssense-text-secondary)', fontSize:13, marginTop:8, lineHeight:1.5 }}>Ask anything about {activeSite}'s data practices.</p>
                  <div className="ssense-quick-prompts">
                    {quickPrompts.map((p, i) => (
                      <button key={i} className="sx-chip-interactive" onClick={() => handleSend(p)}>
                        <Icon name={i === 0 ? "shieldAlert" : i === 1 ? "lock" : "clock"} size={12} style={{ color: 'var(--ssense-accent-cyan)' }} />
                        <span>{p}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {!serviceAvailable && activeSite && (
                <div className="sx-card" style={{ padding: '12px 14px', margin: '0 4px', background: 'var(--ssense-bg-surface)', border: '1px solid var(--ssense-border)', display: 'grid', gap: 6 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 7, color: 'var(--ssense-accent-emerald)', fontWeight: 650, fontSize: 12 }}>
                    <Icon name="shieldCheck" size={15} />
                    <span>Real-Time Local Protection Active</span>
                  </div>
                  <div style={{ fontSize: 11.5, color: 'var(--ssense-text-secondary)', lineHeight: 1.45 }}>
                    Local tracking protection, dark pattern blocking, and Global Privacy Control (GPC) continue working locally. AI conversational answers will resume when the server reconnects.
                  </div>
                </div>
              )}
              {messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
              {isChatting && (
                <div className="ssense-animate-in" style={{ display:'flex', alignItems:'center', gap:10, paddingLeft:4 }}>
                  <div className="ssense-thinking-dot" />
                  <div className="ssense-thinking-dot" style={{ animationDelay:'0.2s' }} />
                  <div className="ssense-thinking-dot" style={{ animationDelay:'0.4s' }} />
                  <span style={{ fontSize:12, color:'var(--ssense-text-muted)', marginLeft:4, fontWeight:500 }}>{loadingText}</span>
                </div>
              )}
            </>
        }
      </div>

      {/* Input dock */}
      <div className="ssense-input-dock">
        <div className="ssense-input-container">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            placeholder={
              isSystemPage ? 'Disabled on system pages'
              : !serviceAvailable ? 'AI server offline · Local protection & cached audits active'
              : !activeSite ? 'Select a site above to chat…'
              : isOffline ? 'Chat requires an internet connection'
              : cooldownLeft > 0 ? `Chat resumes in ${cooldownLeft}s…`
              : `Ask about ${activeSite}'s privacy practices…`
            }
            className="ssense-input-field"
            disabled={isChatting || !activeSite || isSystemPage || !serviceAvailable || cooldownLeft > 0 || isOffline}
          />
          <button onClick={() => handleSend()} disabled={!input.trim() || isChatting || !activeSite || isSystemPage || !serviceAvailable || cooldownLeft > 0 || isOffline} className={`ssense-send-btn${input.trim() && !isChatting && cooldownLeft === 0 && !isOffline && serviceAvailable ? ' active' : ''}`}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </div>
        <div className="ssense-branding">Powered by Ssense AI · DPDP Act 2023</div>
      </div>
    </div>
  );
};
