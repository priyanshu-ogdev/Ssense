import React, { useEffect, useState } from 'react';

interface PrivacySnapshot {
  domain: string;
  policyUrl: string;
  pageUrl: string;
  extractedAt: number;
  textLength: number;
  textSha256: string;
  policyText: string;
  transport: 'cloud' | 'offline' | 'unknown';
  lastAuditAt: number | null;
}

export const PrivacyView: React.FC<{ domain: string | null; onBack: () => void }> = ({ domain, onBack }) => {
  const [snapshot, setSnapshot] = useState<PrivacySnapshot | null>(null);
  const [error, setError] = useState('');
  const [showText, setShowText] = useState(false);

  useEffect(() => {
    setSnapshot(null);
    setError('');
    if (!domain) return;
    chrome.runtime.sendMessage({ type: 'GET_PRIVACY_SNAPSHOT', domain })
      .then((res) => {
        if (res?.success) setSnapshot(res.snapshot);
        else setError(res?.error || 'No privacy snapshot is available yet.');
      })
      .catch((err) => setError(err?.message || 'Could not retrieve the local privacy snapshot.'));
  }, [domain]);

  return (
    <div className="ssense-root" style={{ padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 18 }}>
        <button onClick={onBack} style={buttonStyle}>← Back</button>
        <div>
          <div style={{ fontSize: 16, fontWeight: 750 }}>Privacy Retrieval</div>
          <div style={{ fontSize: 11, color: 'var(--ssense-text-muted)', marginTop: 3 }}>{domain || 'No site selected'}</div>
        </div>
      </div>

      {error && <div style={errorStyle}>{error}</div>}
      {!error && !snapshot && <div style={mutedStyle}>Retrieving the locally stored policy snapshot…</div>}

      {snapshot && (
        <>
          <div style={cardStyle}>
            <div style={sectionTitle}>Source</div>
            <a href={snapshot.policyUrl} target="_blank" rel="noreferrer" style={linkStyle}>{snapshot.policyUrl}</a>
            <div style={metaGrid}>
              <span>Extracted</span><strong>{new Date(snapshot.extractedAt).toLocaleString()}</strong>
              <span>Characters</span><strong>{snapshot.textLength.toLocaleString()}</strong>
              <span>Audit transport</span><strong>{snapshot.transport === 'offline' ? 'Local / Offline' : snapshot.transport === 'cloud' ? 'Cloud' : 'Unknown'}</strong>
              <span>Last audit</span><strong>{snapshot.lastAuditAt ? new Date(snapshot.lastAuditAt).toLocaleString() : 'Not audited'}</strong>
            </div>
            <div style={{ marginTop: 10, fontSize: 9.5, color: 'var(--ssense-text-muted)', wordBreak: 'break-all' }}>
              SHA-256: {snapshot.textSha256}
            </div>
          </div>

          <div style={cardStyle}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={sectionTitle}>What Ssense retrieved</div>
                <div style={mutedStyle}>Only the extracted privacy-policy text is retained here for audit transparency.</div>
              </div>
              <button onClick={() => setShowText(v => !v)} style={buttonStyle}>{showText ? 'Hide' : 'View policy'}</button>
            </div>
            {showText && <pre style={textStyle}>{snapshot.policyText}</pre>}
          </div>

          <div style={infoStyle}>
            <strong>Privacy boundary</strong>
            <div style={{ marginTop: 4 }}>This snapshot is stored in the extension's local IndexedDB. In Cloud Mode, the extracted policy text is sent to your configured Ssense server for auditing. In Offline Mode, inference stays on this device. The transport field above records which path completed the last audit.</div>
          </div>
        </>
      )}
    </div>
  );
};

const buttonStyle: React.CSSProperties = {
  background: 'rgba(255,255,255,.05)', border: '1px solid var(--ssense-border)', color: 'var(--ssense-text-primary)',
  borderRadius: 7, padding: '6px 9px', fontSize: 10.5, cursor: 'pointer'
};
const cardStyle: React.CSSProperties = { background: 'var(--ssense-bg-surface)', border: '1px solid var(--ssense-border)', borderRadius: 11, padding: 14, marginBottom: 10 };
const sectionTitle: React.CSSProperties = { fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', color: 'var(--ssense-text-secondary)', marginBottom: 7 };
const mutedStyle: React.CSSProperties = { color: 'var(--ssense-text-muted)', fontSize: 11, lineHeight: 1.5 };
const errorStyle: React.CSSProperties = { background: 'rgba(244,63,94,.08)', border: '1px solid rgba(244,63,94,.2)', color: 'var(--ssense-accent-rose)', padding: 12, borderRadius: 9, fontSize: 11, lineHeight: 1.5 };
const linkStyle: React.CSSProperties = { color: 'var(--ssense-accent-cyan)', fontSize: 10.5, wordBreak: 'break-all' };
const metaGrid: React.CSSProperties = { display: 'grid', gridTemplateColumns: '1fr 1.4fr', gap: '7px 10px', marginTop: 12, fontSize: 10.5, color: 'var(--ssense-text-muted)' };
const textStyle: React.CSSProperties = { marginTop: 12, maxHeight: 420, overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 10.5, lineHeight: 1.55, color: 'var(--ssense-text-secondary)', background: 'rgba(0,0,0,.18)', padding: 10, borderRadius: 8 };
const infoStyle: React.CSSProperties = { fontSize: 10.5, lineHeight: 1.5, color: 'var(--ssense-text-muted)', padding: 12, border: '1px solid var(--ssense-border)', borderRadius: 9 };
