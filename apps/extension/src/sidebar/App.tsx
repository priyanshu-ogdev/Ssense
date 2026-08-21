import { useEffect, useState } from 'react';
import { ChatInterface, DESIGN_SYSTEM_CSS } from './components/ChatInterface';
import { HistoryView } from './components/HistoryView';
import { PrivacyView } from './components/PrivacyView';

type View = 'audit' | 'history' | 'privacy';

function App() {
  const [view, setView] = useState<View>('audit');
  const [domain, setDomain] = useState<string | null>(null);

  useEffect(() => {
    const styleTag = document.createElement('style');
    styleTag.innerHTML = DESIGN_SYSTEM_CSS;
    document.head.appendChild(styleTag);
    return () => { document.head.removeChild(styleTag); };
  }, []);

  useEffect(() => {
    const readTab = () => chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
      try { setDomain(tab?.url?.startsWith('http') ? new URL(tab.url).hostname : null); } catch { setDomain(null); }
    }).catch(() => setDomain(null));
    readTab();
    chrome.storage.local.get('ssense_sidepanel_view').then((data) => {
      const requested = data.ssense_sidepanel_view;
      if (requested === 'history' || requested === 'privacy' || requested === 'audit') setView(requested);
      if (requested) chrome.storage.local.remove('ssense_sidepanel_view');
    }).catch(() => {});
    chrome.tabs.onActivated.addListener(readTab);
    return () => chrome.tabs.onActivated.removeListener(readTab);
  }, []);

  if (view === 'history') return <HistoryView onBack={() => setView('audit')} />;
  if (view === 'privacy') return <PrivacyView domain={domain} onBack={() => setView('audit')} />;
  return <ChatInterface onOpenHistory={() => setView('history')} onOpenPrivacy={() => setView('privacy')} />;
}

export default App;
