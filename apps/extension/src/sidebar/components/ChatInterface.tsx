// apps/extension/src/sidebar/components/ChatInterface.tsx

import React, { useState, useEffect, useLayoutEffect, useRef, useCallback } from 'react';
import type { ChatResponse, DpdpAuditReport } from '../../types/native-protocol';

// ═══════════════════════════════════════════════════════════════
// 1. DESIGN SYSTEM
// ═══════════════════════════════════════════════════════════════
const DESIGN_SYSTEM_CSS = `
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

  .ssense-header { padding: 16px 20px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--ssense-border); z-index: 10; position: relative; background: rgba(9, 9, 11, 0.8); backdrop-filter: blur(12px); }
  .ssense-header-left { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
  .ssense-header-icon { width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0; background: var(--ssense-gradient-ai); display: flex; align-items: center; justify-content: center; }
  .ssense-header-info { flex: 1; min-width: 0; }
  .ssense-domain { font-size: 13px; font-weight: 600; color: var(--ssense-text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ssense-badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 12px; background: rgba(255,255,255,0.05); margin-top: 4px; }
  .ssense-badge-dot { width: 6px; height: 6px; border-radius: 50%; }

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
export const ChatInterface: React.FC = () => {
  const [domain, setDomain] = useState<string | null>(null);
  const [isSystemPage, setIsSystemPage] = useState(false);
  const [trustScore, setTrustScore] = useState<number | null>(null);
  const [auditReport, setAuditReport] = useState<DpdpAuditReport | null>(null);
  const [showAuditDetails, setShowAuditDetails] = useState(false);
  const [messages, setMessages] = useState<{ role: 'user' | 'ai'; text: string }[]>([]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [loadingText, setLoadingText] = useState('Waking Edge AI...');
  const [isDaemonOnline, setIsDaemonOnline] = useState(true);
  
  const scrollRef = useRef<HTMLDivElement>(null);
  const currentDomainRef = useRef<string | null>(null);

  useEffect(() => {
    const styleTag = document.createElement('style');
    styleTag.innerHTML = DESIGN_SYSTEM_CSS;
    document.head.appendChild(styleTag);
    return () => { document.head.removeChild(styleTag); };
  }, []);

  // 🚀 SOTA FIX 3: Continuous Heartbeat Poll
  useEffect(() => {
    const pingDaemon = () => {
      chrome.runtime.sendMessage({ type: 'HEALTH_CHECK', requestId: 'ping' })
        .then(() => setIsDaemonOnline(true))
        .catch(() => setIsDaemonOnline(false));
    };
    
    pingDaemon(); // Initial check
    const interval = setInterval(pingDaemon, 10000); // Check every 10 seconds
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const handleNewTab = async (urlStr: string | undefined) => {
      if (!urlStr || !urlStr.startsWith('http')) {
        setIsSystemPage(true);
        setDomain(null);
        currentDomainRef.current = null;
        setAuditReport(null);
        return;
      }
      setIsSystemPage(false);
      const newDomain = new URL(urlStr).hostname;
      if (newDomain !== currentDomainRef.current) {
        setDomain(newDomain);
        currentDomainRef.current = newDomain;
        setTrustScore(null);
        setAuditReport(null);
        setShowAuditDetails(false);
        setMessages([]);
        try {
          const res = await chrome.runtime.sendMessage({ type: 'GET_TRUST_SCORE', domain: newDomain });
          if (res?.success) setTrustScore(res.score);
        } catch (e) { setIsDaemonOnline(false); }
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
        if (msg.report) {
          setAuditReport(msg.report);
          setShowAuditDetails(msg.report.violations.length > 0);
        }
        setIsDaemonOnline(true);
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
    const stages = ['Loading Q4_K_M weights...', 'Scanning local context...', 'Reasoning over DPDP Act...', 'Formatting legal response...'];
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
    if (!prompt.trim() || isThinking || !domain || isSystemPage || !isDaemonOnline) return;

    setMessages(prev => [...prev, { role: 'user', text: prompt }]);
    setInput('');
    setIsThinking(true);

    try {
      const response = await chrome.runtime.sendMessage({
        type: 'CHAT', domain, userPrompt: prompt
      }) as ChatResponse;

      setMessages(prev => [...prev, { 
        role: 'ai', 
        text: response.success ? response.message : '⚠️ Connection to Ssense Edge AI lost.' 
      }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'ai', text: '⚠️ Fatal Error: Could not reach the local Rust daemon.' }]);
      setIsDaemonOnline(false);
    } finally {
      setIsThinking(false);
    }
  }, [input, isThinking, domain, isSystemPage, isDaemonOnline]);

  const quickPrompts = domain ? [
    `Is ${domain} selling my data?`,
    `Where is my data stored?`,
    `Explain their retention policy.`
  ] : [];

  return (
    <div className="ssense-root">
      <div style={{ position: 'absolute', top: '-30%', left: '50%', transform: 'translateX(-50%)', width: '120%', height: '60%', background: `radial-gradient(circle, ${trustScore !== null && trustScore < 50 ? 'rgba(244, 63, 94, 0.06)' : 'rgba(6, 182, 212, 0.04)'} 0%, transparent 70%)`, pointerEvents: 'none', zIndex: 0, filter: 'blur(40px)' }} />

      {!isDaemonOnline && (
        <div className="ssense-offline-banner">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>Edge AI Daemon Offline. Please run the Rust backend.</span>
        </div>
      )}

      <header className="ssense-header">
        <div className="ssense-header-left">
          <div className="ssense-header-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
          </div>
          <div className="ssense-header-info">
            <div className="ssense-domain">{isSystemPage ? 'System Page' : (domain || 'Detecting...')}</div>
            {!isSystemPage && <ComplianceBadge score={trustScore} />}
          </div>
        </div>
        <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ssense-text-muted)', flexShrink: 0 }}>Co-Pilot</div>
      </header>

      {/* 🚀 SOTA FIX 2: Single, Sleek Accordion. No double-rendering. */}
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
              <p className="ssense-audit-reasoning">{auditReport.global_legal_reasoning}</p>
              
              {auditReport.violations.map((v, i) => (
                <div key={i} className="ssense-violation-card">
                  <div className="ssense-violation-top">
                    <span className="ssense-violation-type">{v.violation_type.replace(/_/g, ' ')}</span>
                    <span className="ssense-violation-action">{v.network_action.replace(/_/g, ' ')}</span>
                  </div>
                  
                  {v.evidence_quote && (
                    <blockquote className="ssense-evidence">"{v.evidence_quote}"</blockquote>
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
          <input type="text" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSend()} placeholder={isSystemPage ? "Disabled on system pages" : "Interrogate this privacy policy..."} className="ssense-input-field" disabled={isThinking || !domain || isSystemPage || !isDaemonOnline} />
          <button onClick={() => handleSend()} disabled={!input.trim() || isThinking || !domain || isSystemPage || !isDaemonOnline} className={`ssense-send-btn ${input.trim() && !isThinking ? 'active' : ''}`}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
          </button>
        </div>
        <div style={{ textAlign: 'center', marginTop: '12px', fontSize: '10px', color: 'var(--ssense-text-muted)', letterSpacing: '0.02em' }}>Powered by Local Edge AI • Zero Data Leaves Your Machine</div>
      </div>
    </div>
  );
};