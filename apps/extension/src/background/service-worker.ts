// apps/extension/src/background/service-worker.ts
import { sendToNativeDaemon } from './native-messaging';
import type { DaemonResponse } from '../types/native-protocol';

console.log('[Ssense] Service Worker initialized');

const activeAudits = new Map<string, Promise<DaemonResponse>>();

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender)
    .then(response => {
      try {
        sendResponse(response);
      } catch (e) { /* UI was closed, safe to ignore */ }
    })
    .catch(err => {
      console.error('[Ssense SW Error]:', err);
      // 🚀 SOTA FIX 3: Swallow "port closed" errors to prevent SW crashes
      try {
        sendResponse({ success: false, error: err.message });
      } catch (e) { /* UI was closed, safe to ignore */ }
    });
  
  return true; // Keep channel open for async response
});

async function handleMessage(message: any, sender: chrome.runtime.MessageSender): Promise<any> {
  const requestId = crypto.randomUUID();
  const tabId = sender.tab?.id;
  
  switch (message.type) {
    case 'PROXY_FETCH': {
      try {
        // Mode 'cors' omitted. Extension <all_urls> permissions bypass origin restrictions natively.
        const response = await fetch(message.url, { credentials: 'omit' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const html = await response.text();
        return { success: true, html };
      } catch (err: any) {
        return { success: false, error: err.message };
      }
    }

    case 'HEALTH_CHECK': {
      return await sendToNativeDaemon({
        type: 'HEALTH_CHECK',
        requestId,
      });
    }

    case 'AUDIT_POLICY': {
      const cacheKey = message.domain;
      
      // 1. Deduplication Check
      if (activeAudits.has(cacheKey)) {
        console.log(`[Ssense] Deduplicating audit request for ${message.domain}`);
        const response = await activeAudits.get(cacheKey)!;
        
        // 🚀 SOTA FIX 1: Ensure parallel tabs receive the DOM enforcement payload!
        if (response.type === 'AUDIT_POLICY_RESULT' && response.success && tabId) {
            chrome.tabs.sendMessage(tabId, {
              type: 'ENFORCE_DPDP_RULES',
              report: response.report
            }).catch(() => {}); // Safe ignore if tab closed
        }
        return response;
      }

      // 2. Execution & State Wrapper
      const auditExecution = (async () => {
        try {
          const response = await sendToNativeDaemon({ 
            type: 'AUDIT_POLICY', 
            requestId, 
            domain: message.domain, 
            policyText: message.policyText 
          });

          if (response.type === 'AUDIT_POLICY_RESULT' && response.success) {
            chrome.runtime.sendMessage({
              type: 'AUDIT_COMPLETE',
              domain: message.domain,
              score: response.report.dpdp_trust_score,
              report: response.report
            }).catch(() => {});

            if (tabId) {
              chrome.tabs.sendMessage(tabId, {
                type: 'ENFORCE_DPDP_RULES',
                report: response.report
              }).catch(() => {});
            }
          } else if (response.type === 'ERROR') {
            chrome.runtime.sendMessage({
              type: 'AUDIT_ERROR',
              domain: message.domain,
              error: response.error
            }).catch(() => {});
          }
          
          return response;
        } finally {
          activeAudits.delete(cacheKey);
        }
      })();

      activeAudits.set(cacheKey, auditExecution);
      return await auditExecution;
    }

    case 'CHAT':
      return await sendToNativeDaemon({ 
        type: 'CHAT', 
        requestId, 
        domain: message.domain, 
        userPrompt: message.userPrompt 
      });

    case 'GET_TRUST_SCORE':
      return await sendToNativeDaemon({ 
        type: 'GET_TRUST_SCORE', 
        requestId, 
        domain: message.domain 
      });

    default:
      throw new Error(`Unknown message type: ${message.type}`);
  }
}

chrome.action.onClicked.addListener(async (tab) => {
  if (tab.id) {
    await chrome.sidePanel.open({ tabId: tab.id });
  }
});