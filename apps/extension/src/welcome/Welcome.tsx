// First-run onboarding: sign in with Google or continue as guest → done.
import React, { useState } from 'react';
import { BrandMark, Icon } from '../ui/components';
import { GoogleButton, SignInError, SignInPromise, useGoogleSignIn } from '../ui/SignIn';
import { useAuth, usePrefs, useTheme } from '../ui/hooks';

export const Welcome: React.FC = () => {
  const { auth, reload } = useAuth();
  const { prefs } = usePrefs();
  useTheme(prefs?.theme);
  const [step, setStep] = useState<0 | 1>(0);
  const { busy, error, signIn } = useGoogleSignIn(async () => { await reload(); setStep(1); });
  const current = auth?.signedIn && step === 0 ? 1 : step;

  return (
    <div className="wl">
      <aside className="wl-art">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <BrandMark size={40} /><div className="sx-display" style={{ fontSize: 22 }}>Ssense</div>
        </div>
        <div style={{ position: 'relative', zIndex: 1 }}>
          <div className="sx-eyebrow" style={{ marginBottom: 14 }}>DPDP Act 2023 · privacy shield</div>
          <h1 className="sx-display wl-h1">Every site’s privacy policy, <em>read for you.</em></h1>
          <p style={{ fontSize: 16, lineHeight: 1.6, color: 'var(--ssense-text-secondary)', maxWidth: 460, marginTop: 20 }}>
            Ssense audits sites as you open them, flags the clauses that fall short of Indian law, and keeps the results with you on every device.
          </p>
        </div>
        <div className="sx-muted" style={{ fontSize: 12, position: 'relative', zIndex: 1 }}>Your browsing never leaves your device.</div>
      </aside>

      <main className="wl-main">
        <div className="wl-card">
          <div className="wl-steps" aria-label={`Step ${current + 1} of 2`}>{[0, 1].map((i) => <i key={i} data-on={i <= current} />)}</div>

          {current === 0 && (
            <>
              <div><div className="sx-eyebrow">Step 1 of 2</div><h2 className="sx-display" style={{ fontSize: 28, margin: '6px 0 0' }}>Connect your Google account</h2></div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <GoogleButton busy={busy} onClick={signIn} />
                <button
                  type="button"
                  className="sx-btn sx-btn--ghost sx-btn--block"
                  onClick={() => {
                    chrome.storage.local.set({ ssense_onboarded: true }).catch(() => {});
                    setStep(1);
                  }}
                  style={{ justifyContent: 'center' }}
                >
                  Skip for now (Continue as Guest)
                </button>
              </div>
              <SignInError message={error ? `${error} (You can Skip for now to use local protection)` : ''} />
              <SignInPromise />
            </>
          )}

          {current === 1 && (
            <>
              <div><div className="sx-eyebrow">Step 2 of 2</div><h2 className="sx-display" style={{ fontSize: 28, margin: '6px 0 0' }}>You’re protected.</h2></div>
              <ol style={{ margin: 0, paddingLeft: 20, display: 'grid', gap: 10, fontSize: 14, lineHeight: 1.5, color: 'var(--ssense-text-secondary)' }}>
                <li><b style={{ color: 'var(--ssense-text-primary)' }}>Pin Ssense</b> — click the puzzle icon in the toolbar, then the pin beside Ssense.</li>
                <li><b style={{ color: 'var(--ssense-text-primary)' }}>Open any website.</b> The toolbar badge shows its trust score once the scan finishes.</li>
                <li><b style={{ color: 'var(--ssense-text-primary)' }}>Another browser or computer?</b> Install Ssense there and sign in with the same Google account.</li>
              </ol>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="sx-btn sx-btn--primary" style={{ flex: 1 }} onClick={() => {
                  chrome.storage.local.set({ ssense_onboarded: true }).catch(() => {});
                  window.close();
                }}>
                  <Icon name="check" size={15} /> Start browsing
                </button>
                <button className="sx-btn" onClick={() => chrome.runtime.openOptionsPage()}><Icon name="settings" size={15} /> Settings</button>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
};
