// apps/extension/src/sidebar/components/ChatInterface.tsx

import React, { useState, useEffect, useLayoutEffect, useRef, useCallback } from 'react';
import type { ChatResponse, DpdpAuditReport } from '../../types/native-protocol';

function formatDuration(ms: number): string {
  const totalSec = Math.round(ms / 1000);
  if (totalSec < 60) return `${totalSec}s`;
  const min = Math.floor(totalSec / 60);
  if (min < 60) return `${min}m`;
  const hr = Math.floor(min / 60);
  return `${hr}h ${min % 60}m`;
}

// ═══════════════════════════════════════════════════════════════
// 1. DESIGN SYSTEM
// ═══════════════════════════════════════════════════════════════
export const DESIGN_SYSTEM_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --ssense-bg-deep: #09090B;
    --ssense-bg-surface: #18181B;
    --ssense-bg-elevated: #27272A;
    --ssense-border: rgba(255, 255, 255, 0.06);
    --ssense-text-primary: #FAFAFA;
    --ssense-text-secondary: #A1A1AA;
    --ssense-text-muted: #71717A;
    --ssense-accent-cyan: #06B6D4;
    --ssense-accent-violet: #8B5CF6;
    --ssense-accent-emerald: #10B981;
    --ssense-accent-rose: #F43F5E;
    --ssense-accent-amber: #F59E0B;
    --ssense-gradient-ai: linear-gradient(135deg, var(--ssense-accent-cyan) 0%, var(--ssense-accent-violet) 100%);
    --ssense-glass: rgba(255, 255, 255, 0.02);
  }

  .ssense-root { font-family: 'Inter', sans-serif; background: var(--ssense-bg-deep); color: var(--ssense-text-primary); height: 100vh; width: 100%; display: flex; flex-direction: column; overflow: hidden; position: relative; -webkit-font-smoothing: antialiased; }
  .ssense-scroll::-webkit-scrollbar { width: 6px; } .ssense-scroll::-webkit-scrollbar-track { background: transparent; } .ssense-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
  @keyframes ssense-fade-in-up { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes ssense-pulse { 0%, 100% { opacity: 0.4; transform: scale(0.8); } 50% { opacity: 1; transform: scale(1.2); } }
  .ssense-animate-in { animation: ssense-fade-in-up 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
  .ssense-gradient-text { background: var(--ssense-gradient-ai); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .ssense-thinking-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--ssense-accent-cyan); animation: ssense-pulse 1.4s infinite ease-in-out; }

  .ssense-header { padding: 12px 16px 0; display: flex; flex-direction: column; gap: 10px; border-bottom: 1px solid var(--ssense-border); z-index: 10; position: relative; background: rgba(9, 9, 11, 0.8); backdrop-filter: blur(12px); }
  .ssense-header-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .ssense-header-left { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
  .ssense-header-icon { width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0; background: var(--ssense-gradient-ai); display: flex; align-items: center; justify-content: center; }
  .ssense-header-info { flex: 1; min-width: 0; }
  .ssense-domain { font-size: 13px; font-weight: 600; color: var(--ssense-text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ssense-badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 12px; background: rgba(255,255,255,0.05); margin-top: 4px; }
  .ssense-badge-dot { width: 6px; height: 6px; border-radius: 50%; }
  .ssense-tps-pill { flex-shrink: 0; display: inline-flex; align-items: center; gap: 4px; font-size: 10px; font-weight: 600; padding: 4px 8px; border-radius: 6px; background: rgba(6, 182, 212, 0.1); color: var(--ssense-accent-cyan); white-space: nowrap; }

  .ssense-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; padding-bottom: 10px; }
  .ssense-toolbar-btn { display: inline-flex; align-items: center; gap: 5px; background: rgba(255,255,255,0.05); border: 1px solid var(--ssense-border); color: var(--ssense-text-secondary); font-size: 11px; font-weight: 500; padding: 5px 9px; border-radius: 7px; cursor: pointer; transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease; white-space: nowrap; flex-shrink: 0; }
  .ssense-toolbar-btn:hover { background: rgba(255,255,255,0.09); color: var(--ssense-text-primary); border-color: rgba(255,255,255,0.14); }
  .ssense-toolbar-btn--active { background: rgba(6, 182, 212, 0.16); color: var(--ssense-accent-cyan); border-color: rgba(6, 182, 212, 0.35); }
  .ssense-toolbar-spacer { flex: 1 1 auto; min-width: 4px; }

  .ssense-audit-card { margin: 16px 20px 0; border: 1px solid var(--ssense-border); border-radius: 12px; background: var(--ssense-glass); overflow: hidden; transition: all 0.3s ease; flex-shrink: 0; z-index: 10; position: relative;}
  .ssense-audit-header { padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; cursor: pointer; user-select: none; }
  .ssense-audit-header:hover { background: rgba(255,255,255,0.02); }
  .ssense-audit-body { padding: 0 16px 16px; border-top: 1px solid var(--ssense-border); animation: ssense-fade-in-up 0.2s ease; display: flex; flex-direction: column; gap: 12px; max-height: 40vh; overflow-y: auto; }
  .ssense-audit-body::-webkit-scrollbar { width: 4px; } .ssense-audit-body::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
  .ssense-audit-reasoning { font-size: 12px; line-height: 1.5; color: var(--ssense-text-secondary); margin: 12px 0 0 0; font-style: italic; }
  
  .ssense-violation-card { background: var(--ssense-bg-surface); border: 1px solid var(--ssense-border); border-radius: 10px; padding: 12px; display: flex; flex-direction: column; gap: 8px; }
  .ssense-violation-top { display: flex; justify-content: space-between; align-items: center; }
  .ssense-violation-type { color: var(--ssense-accent-rose); font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; padding: 4px 8px; background: rgba(244, 63, 94, 0.1); border-radius: 6px; }
  .ssense-violation-action { color: var(--ssense-text-secondary); font-size: 11px; font-weight: 600; }
  .ssense-evidence { margin: 0; padding: 8px 12px; border-left: 2px solid var(--ssense-accent-rose); background: rgba(244, 63, 94, 0.04); border-radius: 0 6px 6px 0; font-size: 11.5px; line-height: 1.5; color: var(--ssense-text-primary); font-style: italic; }
  .ssense-entities-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
  .ssense-entity-tag { font-size: 10px; font-family: 'JetBrains Mono', monospace; padding: 2px 6px; border-radius: 4px; background: var(--ssense-bg-elevated); color: var(--ssense-text-secondary); border: 1px solid var(--ssense-border); }

  .ssense-stream { flex: 1; overflow-y: auto; padding: 24px 20px; display: flex; flex-direction: column; gap: 24px; z-index: 10; position: relative; }
  .ssense-empty-state { text-align: center; margin-top: 15%; opacity: 0.9; }
  .ssense-quick-prompts { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 24px; }
  .ssense-quick-prompt { padding: 8px 14px; border-radius: 8px; font-size: 12px; font-weight: 500; color: var(--ssense-text-secondary); cursor: pointer; transition: all 0.2s; white-space: nowrap; border: 1px solid var(--ssense-border); background: transparent; font-family: inherit; }
  .ssense-quick-prompt:hover { border-color: var(--ssense-accent-cyan); color: var(--ssense-text-primary); background: rgba(6, 182, 212, 0.05); }

  .ssense-msg { display: flex; max-width: 100%; }
  .ssense-msg-user { justify-content: flex-end; }
  .ssense-msg-ai { justify-content: flex-start; }
  .ssense-msg-bubble { padding: 10px 16px; font-size: 13.5px; line-height: 1.6; max-width: 85%; white-space: pre-wrap; word-break: break-word; color: var(--ssense-text-primary); }
  .ssense-msg-bubble.user { border-radius: 16px 16px 4px 16px; background: var(--ssense-bg-elevated); }
  .ssense-msg-bubble.ai { border-radius: 16px 16px 16px 4px; background: var(--ssense-glass); border: 1px solid var(--ssense-border); }
  .ssense-msg-header { display: flex; align-items: center; gap: 6px; margin-bottom: 10px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  .ssense-msg-header-dot { width: 4px; height: 4px; border-radius: 50%; background: var(--ssense-accent-cyan); }

  .ssense-inline-code { background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--ssense-accent-cyan); }
  .ssense-msg-bubble strong { font-weight: 600; color: #fff; }
  .ssense-msg-bubble em { font-style: italic; color: var(--ssense-text-secondary); }

  .ssense-input-dock { padding: 16px 20px 24px; border-top: 1px solid var(--ssense-border); z-index: 10; position: relative; background: rgba(9, 9, 11, 0.9); backdrop-filter: blur(12px); }
  .ssense-input-container { display: flex; align-items: center; padding: 4px 4px 4px 20px; border-radius: 16px; border: 1px solid var(--ssense-border); background: var(--ssense-bg-surface); transition: border-color 0.2s, box-shadow 0.2s; }
  .ssense-input-container:focus-within { border-color: rgba(6, 182, 212, 0.5); box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.15); }
  .ssense-input-field { flex: 1; background: transparent; border: none; outline: none; color: var(--ssense-text-primary); font-family: inherit; font-size: 14px; padding: 12px 0; }
  .ssense-input-field::placeholder { color: var(--ssense-text-muted); }
  .ssense-send-btn { width: 36px; height: 36px; border-radius: 12px; border: none; flex-shrink: 0; background: transparent; color: var(--ssense-text-muted); cursor: not-allowed; display: flex; align-items: center; justify-content: center; transition: all 0.2s; transform: scale(0.9); }
  .ssense-send-btn.active { background: var(--ssense-gradient-ai); color: #fff; cursor: pointer; transform: scale(1); }

  .ssense-offline-banner { background: rgba(244, 63, 94, 0.1); border-bottom: 1px solid rgba(244, 63, 94, 0.3); padding: 10px 20px; display: flex; align-items: center; gap: 10px; color: var(--ssense-accent-rose); font-size: 12px; font-weight: 500; z-index: 20; position: relative; }
`;

// ═══════════════════════════════════════════════════════════════
// 2. NATIVE REACT TOKENIZER (SOTA FIX 1: CSP Compliant Markdown)
// ═══════════════════════════════════════════════════════════════
const parseMarkdownNodes = (text: string): React.ReactNode[] => {
  // Regex splits by: `code`, **bold**, *italic*, and \n newlines
  const parts = text.split(/(`.*?`|\*\*.*?\*\*|\*.*?\*|\n)/g);
  
  return parts.map((part, i) => {
    if (part === '\n') return <br key={i} />;
    if (part.startsWith('`') && part.endsWith('`')) return <code key={i} className="ssense-inline-code">{part.slice(1, -1)}</code>;
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith('*') && part.endsWith('*')) return <em key={i}>{part.slice(1, -1)}</em>;
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
};

// ═══════════════════════════════════════════════════════════════
// 3. SUB-COMPONENTS
// ═══════════════════════════════════════════════════════════════
const ComplianceBadge = ({ score }: { score: number | null }) => {
  if (score === null) {
    return (
      <div className="ssense-badge">
        <div className="ssense-badge-dot" style={{ background: 'var(--ssense-accent-amber)', animation: 'ssense-pulse 1.5s infinite' }} />
        <span style={{ fontSize: '11px', fontWeight: 500, color: 'var(--ssense-text-secondary)' }}>Scanning</span>
      </div>
    );
  }
  const color = score >= 80 ? 'var(--ssense-accent-emerald)' : score >= 50 ? 'var(--ssense-accent-amber)' : 'var(--ssense-accent-rose)';
  const label = score >= 80 ? 'Compliant' : score >= 50 ? 'Caution' : 'Violations';
  return (
    <div className="ssense-badge">
      <div className="ssense-badge-dot" style={{ background: color }} />
      <span style={{ fontSize: '11px', fontWeight: 600, color }}>{score}</span>
      <span style={{ fontSize: '11px', fontWeight: 500, color: 'var(--ssense-text-muted)' }}>{label}</span>
    </div>
  );
};

const MessageBubble = React.memo(({ msg }: { msg: { role: 'user' | 'ai'; text: string } }) => (
  <div className={`ssense-msg ssense-animate-in ${msg.role === 'user' ? 'ssense-msg-user' : 'ssense-msg-ai'}`}>
    <div className={`ssense-msg-bubble ${msg.role}`}>
      {msg.role === 'ai' && (
        <div className="ssense-msg-header">
          <div className="ssense-msg-header-dot" />
          <span className="ssense-gradient-text">Ssense AI</span>
        </div>
      )}
      {msg.role === 'ai' ? parseMarkdownNodes(msg.text) : msg.text}
    </div>
  </div>
));

// ═══════════════════════════════════════════════════════════════
// 4. MAIN CO-PILOT INTERFACE
// ═══════════════════════════════════════════════════════════════
export const ChatInterface: React.FC<{ onOpenHistory?: () => void; onOpenPrivacy?: () => void }> = ({ onOpenHistory, onOpenPrivacy }) => {
  const [domain, setDomain] = useState<string | null>(null);
  const [isSystemPage, setIsSystemPage] = useState(false);
  const [trustScore, setTrustScore] = useState<number | null>(null);
  const [auditReport, setAuditReport] = useState<DpdpAuditReport | null>(null);
  const [auditError, setAuditError] = useState('');
  const [showAuditDetails, setShowAuditDetails] = useState(false);
  const [messages, setMessages] = useState<{ role: 'user' | 'ai'; text: string }[]>([]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [loadingText, setLoadingText] = useState('Waking Edge AI...');
  const [serviceAvailable, setServiceAvailable] = useState(true);
  const [serviceError, setServiceError] = useState('');
  const [showExplainability, setShowExplainability] = useState(false);
  const [siteHistory, setSiteHistory] = useState<any | null>(null);
  const [tps, setTps] = useState<number>(120);
  const [showShieldSettings, setShowShieldSettings] = useState(false);
  const [shieldSettings, setShieldSettings] = useState({
    blockTrackers: true,
    spoofHardware: true,
    injectGPC: true,
  });

  const exportAuditReport = () => {
    if (!auditReport || !domain) return;
    const md = [
      `# Ssense DPDP Compliance Forensic Audit Report`,
      `**Target Domain:** \`${domain}\``,
      `**DPDP Trust Score:** \`${auditReport.dpdp_trust_score}/100\``,
      `**Subtlety & Obfuscation Score:** \`${auditReport.subtlety_score}/100\``,
      `**Audit Date:** \`${new Date().toISOString()}\`\n`,
      auditReport.explainability ? `**Explainability:** \`${auditReport.explainability.method}\`` : '',
      `## Global Legal Reasoning`,
      `${auditReport.global_legal_reasoning}\n`,
      `## Detected Violations (${auditReport.violations.length})`,
      ...auditReport.violations.map((v, i) => [
        `### ${i + 1}. ${v.violation_type.replace(/_/g, ' ')}`,
        `- **Statute Reference:** ${v.statute_reference || 'DPDP Act Section 8'}`,
        `- **Enforcement Action:** \`${v.network_action}\``,
        `- **Evidence Quote:** "${v.evidence_quote}"`,
        `- **Semantic Justification:** ${v.step_3_semantic_justification}`,
        v.offending_entities && v.offending_entities.length > 0 ? `- **Offending Entities:** ${v.offending_entities.join(', ')}` : ''
      ].filter(Boolean).join('\n'))
    ].join('\n\n');

    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ssense_audit_${domain.replace(/[^a-zA-Z0-9]/g, '_')}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  const scrollRef = useRef<HTMLDivElement>(null);
  const currentDomainRef = useRef<string | null>(null);

  useEffect(() => {
    // NOTE: CSS injection lives in App.tsx now (mounted once, never torn
    // down) so switching between ChatInterface and HistoryView doesn't
    // rip the stylesheet out from under whichever view is showing.
  }, []);

  useEffect(() => {
    // NOTE: GET_ENGINE_CONFIG no longer has a `mode` field (dual-mode was
    // removed — SLM server is the only backend). This check now just
    // confirms the ping below actually finds a reachable server config.

    const pingService = () => {
      chrome.runtime.sendMessage({ type: 'HEALTH_CHECK', requestId: crypto.randomUUID() })
        .then((res) => {
          const ok = Boolean(res?.success) && res?.modelLoaded !== false;
          setServiceAvailable(ok);
          if (!ok) setServiceError(res?.error || (res?.modelLoaded === false ? 'The AI model is not loaded. Open Settings or enable Offline Mode after the model download completes.' : 'The AI service is unavailable.'));
          else setServiceError('');
          if (res?.avgTokensPerSecond) setTps(res.avgTokensPerSecond);
        })
        .catch((err) => {
          setServiceAvailable(false);
          setServiceError(err?.message || 'Could not contact the Ssense service.');
        });
    };

    pingService();
    const interval = setInterval(pingService, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const handleNewTab = async (urlStr: string | undefined) => {
      if (!urlStr || !urlStr.startsWith('http')) {
        setIsSystemPage(true);
        setDomain(null);
        currentDomainRef.current = null;
        setAuditReport(null);
        setAuditError('');
        return;
      }
      setIsSystemPage(false);
      const newDomain = new URL(urlStr).hostname;
      if (newDomain !== currentDomainRef.current) {
        setDomain(newDomain);
        currentDomainRef.current = newDomain;
        setTrustScore(null);
        setAuditReport(null);
        setAuditError('');
        setShowAuditDetails(false);
        setMessages([]);
        setSiteHistory(null);
        try {
          const [scoreRes, historyRes, chatRes] = await Promise.all([
            chrome.runtime.sendMessage({ type: 'GET_TRUST_SCORE', domain: newDomain }),
            chrome.runtime.sendMessage({ type: 'GET_SITE_HISTORY', domain: newDomain }),
            chrome.runtime.sendMessage({ type: 'GET_CHAT_HISTORY', domain: newDomain }),
          ]);
          if (scoreRes?.success) setTrustScore(scoreRes.score);
          if (historyRes?.success) setSiteHistory(historyRes.entry || null);
          if (chatRes?.success && Array.isArray(chatRes.messages)) {
            // Only role/text matter here — chat-store also carries id/domain/timestamp,
            // which this view doesn't render.
            setMessages(chatRes.messages.map((m: any) => ({ role: m.role, text: m.text })));
          }
        } catch (e) {
          setServiceError('Could not retrieve this site\'s local history.');
        }
      }
    };

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => handleNewTab(tabs[0]?.url));
    const handleTabUpdate = (_tabId: number, changeInfo: chrome.tabs.TabChangeInfo, tab: chrome.tabs.Tab) => {
      if (tab.active && (changeInfo.status === 'complete' || changeInfo.url)) handleNewTab(tab.url);
    };
    const handleTabActivated = async (activeInfo: chrome.tabs.TabActiveInfo) => {
      const tab = await chrome.tabs.get(activeInfo.tabId);
      handleNewTab(tab.url);
    };

    const messageListener = (msg: any) => {
      if (msg.type === 'AUDIT_COMPLETE' && msg.domain === currentDomainRef.current) {
        setTrustScore(msg.score);
        setAuditError('');
        if (msg.report) {
          setAuditReport(msg.report);
          setShowAuditDetails(msg.report.violations.length > 0);
          chrome.runtime.sendMessage({ type: 'GET_SITE_HISTORY', domain: msg.domain })
            .then((historyRes) => { if (historyRes?.success) setSiteHistory(historyRes.entry || null); })
            .catch(() => {});
        }
        setServiceAvailable(true);
        setServiceError('');
      }
      if (msg.type === 'AUDIT_ERROR' && msg.domain === currentDomainRef.current) {
        setAuditError(msg.error || 'The privacy policy audit could not be completed.');
        setServiceError(msg.error || 'The AI service could not complete the audit.');
        setServiceAvailable(false);
      }
    };

    chrome.tabs.onUpdated.addListener(handleTabUpdate);
    chrome.tabs.onActivated.addListener(handleTabActivated);
    chrome.runtime.onMessage.addListener(messageListener);

    return () => { 
      chrome.tabs.onUpdated.removeListener(handleTabUpdate);
      chrome.tabs.onActivated.removeListener(handleTabActivated);
      chrome.runtime.onMessage.removeListener(messageListener);
    };
  }, []);

  useLayoutEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, isThinking]);

  useEffect(() => {
    if (!isThinking) return;
    const stages = ['Connecting to Ssense AI...', 'Scanning local context...', 'Reasoning over DPDP Act...', 'Formatting legal response...'];
    let step = 0;
    setLoadingText(stages[0]);
    const interval = setInterval(() => {
      step = (step + 1) % stages.length;
      setLoadingText(stages[step]);
    }, 3500);
    return () => clearInterval(interval);
  }, [isThinking]);

  const handleSend = useCallback(async (text?: string) => {
    const prompt = text || input;
    if (!prompt.trim() || isThinking || !domain || isSystemPage || !serviceAvailable) return;

    setMessages(prev => [...prev, { role: 'user', text: prompt }]);
    setInput('');
    setIsThinking(true);

    try {
      const response = await chrome.runtime.sendMessage({
        type: 'CHAT', domain, userPrompt: prompt
      }) as ChatResponse;

      setMessages(prev => [...prev, { 
        role: 'ai', 
        text: response.success ? response.message : `⚠️ ${response.error || 'Ssense AI could not complete this request.'}` 
      }]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Could not reach the Ssense AI service.';
      setMessages(prev => [...prev, { role: 'ai', text: `⚠️ ${message}` }]);
      setServiceAvailable(false);
      setServiceError(message);
    } finally {
      setIsThinking(false);
    }
  }, [input, isThinking, domain, isSystemPage, serviceAvailable]);

  const handleClearChat = useCallback(async () => {
    if (!domain) return;
    setMessages([]);
    try {
      await chrome.runtime.sendMessage({ type: 'CLEAR_CHAT_HISTORY', domain });
    } catch {
      // Best-effort — the visible chat is already cleared either way.
    }
  }, [domain]);

  const quickPrompts = domain ? [
    `Is ${domain} selling my data?`,
    `Where is my data stored?`,
    `Explain their retention policy.`
  ] : [];

  return (
    <div className="ssense-root">
      <div style={{ position: 'absolute', top: '-30%', left: '50%', transform: 'translateX(-50%)', width: '120%', height: '60%', background: `radial-gradient(circle, ${trustScore !== null && trustScore < 50 ? 'rgba(244, 63, 94, 0.06)' : 'rgba(6, 182, 212, 0.04)'} 0%, transparent 70%)`, pointerEvents: 'none', zIndex: 0, filter: 'blur(40px)' }} />

      {!serviceAvailable && (
        <div className="ssense-offline-banner">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>{serviceError || 'AI service unavailable. Check Settings or Offline Mode.'}</span>
        </div>
      )}

      <header className="ssense-header">
        <div className="ssense-header-top">
          <div className="ssense-header-left">
            <div className="ssense-header-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            </div>
            <div className="ssense-header-info">
              <div className="ssense-domain">{isSystemPage ? 'System Page' : (domain || 'Detecting...')}</div>
              {!isSystemPage && <ComplianceBadge score={trustScore} />}
            </div>
          </div>
          {!isSystemPage && (
            <div className="ssense-tps-pill" title="Tokens per second">
              <span>{tps}</span><span style={{ opacity: 0.7 }}>TPS</span>
            </div>
          )}
        </div>

        <nav className="ssense-toolbar">
          <button
            className="ssense-toolbar-btn"
            onClick={() => onOpenHistory?.()}
            title="View browsing history and past audits"
          >
            <span aria-hidden="true">🕘</span><span>History</span>
          </button>
          <button
            className="ssense-toolbar-btn"
            onClick={() => onOpenPrivacy?.()}
            title="View the privacy policy Ssense actually retrieved"
          >
            <span aria-hidden="true">🔎</span><span>Privacy</span>
          </button>
          <button
            className="ssense-toolbar-btn"
            onClick={() => chrome.runtime.openOptionsPage()}
            title="Configure server URL, API key, and HMAC secret"
          >
            <span aria-hidden="true">⚙️</span><span>Settings</span>
          </button>
          <span className="ssense-toolbar-spacer" />
          {messages.length > 0 && (
            <button
              className="ssense-toolbar-btn"
              onClick={handleClearChat}
              title="Clear this site's saved chat history"
            >
              <span aria-hidden="true">🗑️</span><span>Clear chat</span>
            </button>
          )}
          <button
            className={`ssense-toolbar-btn${showShieldSettings ? ' ssense-toolbar-btn--active' : ''}`}
            onClick={() => setShowShieldSettings(!showShieldSettings)}
            title="Configure Active DOM & Network Shield"
          >
            <span aria-hidden="true">🛡️</span><span>Shield</span>
          </button>
        </nav>
      </header>

      {showShieldSettings && (
        <div style={{ background: 'var(--ssense-bg-card)', borderBottom: '1px solid var(--ssense-border)', padding: '12px 16px', fontSize: '11px', display: 'flex', flexDirection: 'column', gap: '8px', zIndex: 10 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontWeight: 600, color: '#fff' }}>
            <span>Granular Shield Controls</span>
            <span style={{ fontSize: '10px', color: 'var(--ssense-accent-cyan)', cursor: 'pointer' }} onClick={() => setShowShieldSettings(false)}>✕ Close</span>
          </div>
          <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', color: 'var(--ssense-text-main)' }}>
            <span>Block Third-Party Trackers &amp; Iframes</span>
            <input type="checkbox" checked={shieldSettings.blockTrackers} onChange={e => setShieldSettings({ ...shieldSettings, blockTrackers: e.target.checked })} style={{ cursor: 'pointer' }} />
          </label>
          <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', color: 'var(--ssense-text-main)' }}>
            <span>Spoof Hardware APIs (Canvas/Audio/Battery)</span>
            <input type="checkbox" checked={shieldSettings.spoofHardware} onChange={e => setShieldSettings({ ...shieldSettings, spoofHardware: e.target.checked })} style={{ cursor: 'pointer' }} />
          </label>
          <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', color: 'var(--ssense-text-main)' }}>
            <span>Inject Global Privacy Control (GPC) Signal</span>
            <input type="checkbox" checked={shieldSettings.injectGPC} onChange={e => setShieldSettings({ ...shieldSettings, injectGPC: e.target.checked })} style={{ cursor: 'pointer' }} />
          </label>
        </div>
      )}

      {auditError && !isSystemPage && (
        <div style={{ margin: '10px 20px 0', padding: '10px 12px', borderRadius: 9, border: '1px solid rgba(245,158,11,.25)', background: 'rgba(245,158,11,.07)', color: 'var(--ssense-accent-amber)', fontSize: 10.5, lineHeight: 1.45 }}>
          <strong>Audit unavailable.</strong> {auditError}
          <div style={{ marginTop: 5, color: 'var(--ssense-text-muted)' }}>No compliance score is shown until a complete, validated report is available.</div>
          <button
            onClick={async () => {
              if (!domain || isThinking) return;
              setIsThinking(true);
              setAuditError('');
              try {
                const snap = await chrome.runtime.sendMessage({ type: 'GET_PRIVACY_SNAPSHOT', domain });
                if (!snap?.success) throw new Error(snap?.error || 'No retrieved privacy policy is available to retry.');
                const res = await chrome.runtime.sendMessage({ type: 'AUDIT_POLICY', domain, policyText: snap.snapshot.policyText });
                if (res?.type === 'ERROR') throw new Error(res.error || 'Retry failed.');
              } catch (err) {
                setAuditError(err instanceof Error ? err.message : 'Retry failed.');
                setServiceAvailable(false);
              } finally {
                setIsThinking(false);
              }
            }}
            disabled={isThinking}
            style={{ marginTop: 8, border: '1px solid rgba(245,158,11,.3)', background: 'transparent', color: 'var(--ssense-accent-amber)', borderRadius: 6, padding: '5px 8px', fontSize: 9.5, cursor: 'pointer' }}
          >
            Retry audit from retrieved policy
          </button>
        </div>
      )}

      {auditReport && !isSystemPage && (
        <div className="ssense-audit-card">
          <div className="ssense-audit-header" onClick={() => setShowAuditDetails(!showAuditDetails)}>
            <span className="ssense-gradient-text" style={{ fontWeight: 600, fontSize: '12px' }}>
              {auditReport.violations.length === 0 ? '✅ Policy Compliant' : `⚠️ ${auditReport.violations.length} Violations Found`}
            </span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ transform: showAuditDetails ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </div>
          {showAuditDetails && (
            <div className="ssense-audit-body">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255, 255, 255, 0.03)', padding: '8px', borderRadius: '6px', marginBottom: '8px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                <div>
                  <div style={{ fontSize: '10px', color: 'var(--ssense-text-muted)' }}>DPDP Trust Score</div>
                  <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--ssense-accent-cyan)' }}>{auditReport.dpdp_trust_score} / 100</div>
                </div>
                <div>
                  <div style={{ fontSize: '10px', color: 'var(--ssense-text-muted)' }}>Obfuscation Subtlety</div>
                  <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--ssense-accent-blue)' }} title="Higher score indicates complex legal phrasing designed to obscure violations">{auditReport.subtlety_score} / 100</div>
                </div>
                <button 
                  onClick={(e) => { e.stopPropagation(); exportAuditReport(); }}
                  style={{ background: 'var(--ssense-accent-gradient)', border: 'none', color: '#000', fontWeight: 600, fontSize: '10px', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer' }}
                >
                  📥 Export Report
                </button>
              </div>
              <p className="ssense-audit-reasoning">{auditReport.global_legal_reasoning}</p>

              {(auditReport.explainability || siteHistory) && (
                <div style={{ marginBottom: 10, border: '1px solid var(--ssense-border)', borderRadius: 8, padding: 9, background: 'rgba(6,182,212,.025)' }}>
                  <button onClick={(e) => { e.stopPropagation(); setShowExplainability(v => !v); }} style={{ width: '100%', background: 'transparent', border: 0, color: 'var(--ssense-text-primary)', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', fontSize: 10.5, fontWeight: 700, padding: 0 }}>
                    <span>{auditReport.explainability?.method === 'SHAP' ? 'Why this score? · Model attribution' : 'Why this score? · Site history'}</span>
                    <span>{showExplainability ? '−' : '+'}</span>
                  </button>
                  {showExplainability && (
                    <div style={{ marginTop: 9 }}>
                      {auditReport.explainability?.method === 'SHAP' && auditReport.explainability.features?.some((f) => Number.isFinite(Number(f.shap_value))) ? (
                        <>
                          <div style={{ fontSize: 9.5, color: 'var(--ssense-text-muted)', marginBottom: 7 }}>Model-supplied SHAP attribution. Values below come from the audit model, not from browser history.</div>
                          {auditReport.explainability.features.slice(0, 8).map((feature, idx) => {
                            const value = Number(feature.shap_value);
                            const width = Math.min(100, Math.abs(value) * 100);
                            return <div key={`${feature.feature}-${idx}`} style={{ marginBottom: 7 }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 9.5 }}>
                                <span style={{ color: 'var(--ssense-text-secondary)' }}>{feature.feature}</span>
                                <strong style={{ color: value >= 0 ? 'var(--ssense-accent-rose)' : 'var(--ssense-accent-emerald)' }}>{value >= 0 ? '+' : ''}{value.toFixed(3)}</strong>
                              </div>
                              <div style={{ height: 4, marginTop: 3, background: 'rgba(255,255,255,.06)', borderRadius: 4 }}>
                                <div style={{ width: `${Math.max(3, width)}%`, height: '100%', borderRadius: 4, background: value >= 0 ? 'var(--ssense-accent-rose)' : 'var(--ssense-accent-emerald)' }} />
                              </div>
                              {feature.evidence && <div style={{ fontSize: 8.5, color: 'var(--ssense-text-muted)', marginTop: 2 }}>{feature.evidence}</div>}
                            </div>;
                          })}
                        </>
                      ) : (
                        <>
                          <div style={{ fontSize: 9.5, color: 'var(--ssense-text-muted)', marginBottom: 8 }}>
                            This build does not receive genuine mathematical SHAP values from the backend yet. Instead of presenting extension-collected fields as SHAP, Ssense shows the site's local audit history.
                          </div>
                          {siteHistory ? (
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 7 }}>
                              {[
                                ['Current score', siteHistory.lastScore !== null ? `${siteHistory.lastScore}/100` : '—'],
                                ['Visits', String(siteHistory.visitCount)],
                                ['Time on site', formatDuration(siteHistory.totalTimeMs)],
                                ['Last audit', siteHistory.lastAuditAt ? new Date(siteHistory.lastAuditAt).toLocaleString() : 'Not audited'],
                                ['Violations', String(siteHistory.lastReport?.violations?.length || 0)],
                                ['First seen', new Date(siteHistory.firstVisit).toLocaleDateString()],
                              ].map(([label, value]) => (
                                <div key={label} style={{ padding: 7, borderRadius: 6, background: 'rgba(255,255,255,.035)', border: '1px solid rgba(255,255,255,.05)' }}>
                                  <div style={{ fontSize: 8.5, color: 'var(--ssense-text-muted)' }}>{label}</div>
                                  <div style={{ fontSize: 10, fontWeight: 650, marginTop: 2, color: 'var(--ssense-text-primary)' }}>{value}</div>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div style={{ fontSize: 9.5, color: 'var(--ssense-text-muted)' }}>No local history has been recorded for this site yet.</div>
                          )}
                          {auditReport.violations.length > 0 && <div style={{ marginTop: 8, fontSize: 9.5, color: 'var(--ssense-text-secondary)' }}>Score context: {auditReport.violations.length} detected violation{auditReport.violations.length === 1 ? '' : 's'} in the latest validated audit.</div>}
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}
              
              {auditReport.violations.map((v, i) => (
                <div key={i} className="ssense-violation-card">
                  <div className="ssense-violation-top">
                    <span className="ssense-violation-type">{v.violation_type.replace(/_/g, ' ')}</span>
                    <span className="ssense-violation-action">{v.network_action.replace(/_/g, ' ')}</span>
                  </div>
                  
                  {v.evidence_quote && (
                    <blockquote 
                      className="ssense-evidence" 
                      style={{ cursor: 'pointer', transition: 'border-color 0.2s' }}
                      title="Click to locate and highlight this exact text in the active tab"
                      onClick={() => {
                        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                          if (tabs[0]?.id) {
                            chrome.tabs.sendMessage(tabs[0].id, { type: 'HIGHLIGHT_IN_DOM', quote: v.evidence_quote }).catch(() => {});
                          }
                        });
                      }}
                    >
                      "{v.evidence_quote}"
                      <div style={{ fontSize: '9.5px', color: 'var(--ssense-accent-cyan)', marginTop: '4px', fontStyle: 'normal', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span>🔍 Click to highlight in page</span>
                      </div>
                    </blockquote>
                  )}
                  
                  {v.offending_entities && v.offending_entities.length > 0 && (
                    <div className="ssense-entities-list">
                      {v.offending_entities.map((e, idx) => (
                        <span key={`${i}-${idx}`} className="ssense-entity-tag">{e}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div ref={scrollRef} className="ssense-stream">
        {isSystemPage ? (
           <div className="ssense-empty-state" style={{ marginTop: '30%', color: 'var(--ssense-text-muted)' }}>Ssense AI is disabled on internal browser pages.</div>
        ) : (
          <>
            {messages.length === 0 && !isThinking && domain && !auditReport && (
              <div className="ssense-empty-state">
                <h2 className="ssense-gradient-text" style={{ fontSize: '22px', fontWeight: 700, margin: 0, letterSpacing: '-0.02em' }}>Ssense Co-Pilot</h2>
                <p style={{ color: 'var(--ssense-text-secondary)', fontSize: '13px', marginTop: '8px', lineHeight: 1.5 }}>Your local AI legal auditor.<br/>Ask anything about this site's data practices.</p>
                <div className="ssense-quick-prompts">
                  {quickPrompts.map((p, i) => <button key={i} className="ssense-quick-prompt" onClick={() => handleSend(p)}>{p}</button>)}
                </div>
              </div>
            )}

            {messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}

            {isThinking && (
              <div className="ssense-animate-in" style={{ display: 'flex', alignItems: 'center', gap: '10px', paddingLeft: '4px' }}>
                <div className="ssense-thinking-dot" />
                <div className="ssense-thinking-dot" style={{ animationDelay: '0.2s' }} />
                <div className="ssense-thinking-dot" style={{ animationDelay: '0.4s' }} />
                <span style={{ fontSize: '12px', color: 'var(--ssense-text-muted)', marginLeft: '4px', fontWeight: 500 }}>{loadingText}</span>
              </div>
            )}
          </>
        )}
      </div>

      <div className="ssense-input-dock">
        <div className="ssense-input-container">
          <input type="text" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSend()} placeholder={isSystemPage ? "Disabled on system pages" : "Interrogate this privacy policy..."} className="ssense-input-field" disabled={isThinking || !domain || isSystemPage || !serviceAvailable} />
          <button onClick={() => handleSend()} disabled={!input.trim() || isThinking || !domain || isSystemPage || !serviceAvailable} className={`ssense-send-btn ${input.trim() && !isThinking ? 'active' : ''}`}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
          </button>
        </div>
        <div style={{ textAlign: 'center', marginTop: '12px', fontSize: '10px', color: 'var(--ssense-text-muted)', letterSpacing: '0.02em' }}>Powered by Local Edge AI • Zero Data Leaves Your Machine</div>
      </div>
    </div>
  );
};