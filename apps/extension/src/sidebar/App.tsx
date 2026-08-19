// apps/extension/src/sidebar/App.tsx
import { useEffect, useState } from 'react';
import { ChatInterface, DESIGN_SYSTEM_CSS } from './components/ChatInterface';
import { HistoryView } from './components/HistoryView';

type View = 'audit' | 'history';

function App() {
  const [view, setView] = useState<View>('audit');

  // Injected once at the App level (not inside ChatInterface) so it
  // survives switching between Audit and History views — neither view
  // owns the stylesheet's lifecycle.
  useEffect(() => {
    const styleTag = document.createElement('style');
    styleTag.innerHTML = DESIGN_SYSTEM_CSS;
    document.head.appendChild(styleTag);
    return () => { document.head.removeChild(styleTag); };
  }, []);

  if (view === 'history') {
    return <HistoryView onBack={() => setView('audit')} />;
  }
  return <ChatInterface onOpenHistory={() => setView('history')} />;
}

export default App;