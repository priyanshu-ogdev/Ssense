// apps/extension/src/sidebar/App.tsx
import React from 'react';
import { ChatInterface } from './components/ChatInterface';

// 🚀 SOTA FIX: Strip away the placeholder. 
// The sole purpose of App.tsx is to mount the Co-Pilot engine.
function App() {
  return <ChatInterface />;
}

export default App;