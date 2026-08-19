// apps/extension/src/background/service-worker.ts
import { getServerConfig, serverHealthCheck, serverAuditPolicy, serverChat, serverTrustScore } from './api-client';
import type { DaemonResponse } from '../types/native-protocol';
import * as historyStore from './history-store';
import * as chatStore from './chat-store';

console.log('[Ssense] Service Worker initialized — SLM Server mode only.');

// ═══════════════════════════════════════════════════════════════
// TIME-ON-SITE TRACKING
// ═══════════════════════════════════════════════════════════════
// Tracks time spent on the CURRENTLY ACTIVE tab of the CURRENTLY FOCUSED
// window only — switching tabs, switching windows, or the browser losing
// focus (e.g. alt-tabbing to another app) all correctly stop the clock for
// the previous domain and start it for the new one. Flushed to IndexedDB
// via history-store.addTime() in small increments rather than one big
// write at the end, so a crashed/killed service worker doesn't lose more
// than a few seconds of accumulated time.
let _trackedDomain: string | null = null;
let _trackedSince: number | null = null;
let _windowFocused = true;

function hostnameFromUrl(url: string | undefined): string | null {
  if (!url) return null;
  try {
    const h = new URL(url).hostname;
    return h || null;
  } catch {
    return null;
  }
}

async function flushTrackedTime(): Promise<void> {
  if (_trackedDomain && _trackedSince) {
    const delta = Date.now() - _trackedSince;
    if (delta > 0) await historyStore.addTime(_trackedDomain, delta);
  }
  _trackedSince = null;
}

async function startTracking(domain: string | null): Promise<void> {
  await flushTrackedTime();
  _trackedDomain = domain;
  _trackedSince = domain && _windowFocused ? Date.now() : null;
}

async function refreshActiveTabTracking(): Promise<void> {
  try {
    const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    const domain = hostnameFromUrl(tab?.url);
    if (domain !== _trackedDomain) {
      if (domain) await historyStore.recordVisit(domain);
      await startTracking(domain);
    }
  } catch {
    /* no active tab (e.g. all windows closed) — nothing to track */
  }
}

chrome.tabs.onActivated.addListener(() => { refreshActiveTabTracking(); });
chrome.tabs.onUpdated.addListener((_id, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.active) refreshActiveTabTracking();
});
chrome.windows.onFocusChanged.addListener(async (windowId) => {
  _windowFocused = windowId !== chrome.windows.WINDOW_ID_NONE;
  if (_windowFocused) {
    await refreshActiveTabTracking();
  } else {
    // Browser lost OS focus entirely — stop the clock but keep the domain
    // "selected" so we resume timing the same tab on refocus.
    await flushTrackedTime();
  }
});
// Periodic safety flush every 20s, in case a service worker teardown
// happens before a natural start/stop event fires.
setInterval(() => { flushTrackedTime().then(() => { _trackedSince = _trackedDomain && _windowFocused ? Date.now() : null; }); }, 20000);
refreshActiveTabTracking();

const activeAudits = new Map<string, Promise<DaemonResponse>>();
const completedAuditsCache = new Map<string, { timestamp: number; response: DaemonResponse }>();
const CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes TTL

async function computePolicyHash(domain: string, text: string): Promise<string> {
  const enc = new TextEncoder();
  const data = enc.encode(`${domain}:${text}`);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender)
    .then(response => {
      try {
        sendResponse(response);
      } catch (e) { /* UI was closed, safe to ignore */ }
    })
    .catch(err => {
      console.error('[Ssense SW Error]:', err);
      try {
        sendResponse({ success: false, error: err.message });
      } catch (e) { /* UI was closed, safe to ignore */ }
    });
  
  return true;
});

// Single-mode routing: every request goes to the SLM server. No local
// daemon, no AUTO failover — one code path, one thing to keep secure and
// correct. If the server is unreachable, callers get a clear ERROR
// response (see api-client.ts) instead of silently switching backends.
async function routeRequest(
  type: 'HEALTH_CHECK' | 'AUDIT_POLICY' | 'CHAT' | 'GET_TRUST_SCORE',
  payload: any,
  requestId: string
): Promise<DaemonResponse> {
  switch (type) {
    case 'HEALTH_CHECK': return await serverHealthCheck(requestId);
    case 'AUDIT_POLICY': return await serverAuditPolicy(payload.domain, payload.policyText, requestId);
    case 'CHAT': return await serverChat(payload.domain, payload.userPrompt, requestId);
    case 'GET_TRUST_SCORE': return await serverTrustScore(payload.domain, requestId);
  }
}

async function handleMessage(message: any, sender: chrome.runtime.MessageSender): Promise<any> {
  const requestId = crypto.randomUUID();
  const tabId = sender.tab?.id;
  
  switch (message.type) {
    case 'PROXY_FETCH': {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000); // 15s cap
        try {
          const response = await fetch(message.url, { credentials: 'omit', signal: controller.signal });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const html = await response.text();
          return { success: true, html };
        } finally {
          clearTimeout(timeoutId);
        }
      } catch (err: any) {
        const isTimeout = err.name === 'AbortError';
        return { success: false, error: isTimeout ? 'Policy fetch timed out after 15s' : err.message };
      }
    }

    case 'GET_ENGINE_CONFIG': {
      return await getServerConfig();
    }

    case 'CLEAR_INFERENCE_CACHE': {
      completedAuditsCache.clear();
      return { success: true, message: 'Inference cache cleared.' };
    }

    case 'GET_HISTORY': {
      const entries = await historyStore.getAllEntries();
      entries.sort((a, b) => b.lastVisit - a.lastVisit);
      return { success: true, entries };
    }

    case 'CLEAR_HISTORY': {
      await historyStore.clearAllEntries();
      return { success: true };
    }

    case 'HEALTH_CHECK': {
      return await routeRequest('HEALTH_CHECK', {}, requestId);
    }

    case 'AUDIT_POLICY': {
      const cacheKey = await computePolicyHash(message.domain, message.policyText || '');
      
      // Check LRU completed cache first (<5ms hit)
      const cachedEntry = completedAuditsCache.get(cacheKey);
      if (cachedEntry && Date.now() - cachedEntry.timestamp < CACHE_TTL_MS) {
        console.log(`[Ssense] Cache hit for ${message.domain} (<5ms instant response)`);
        const response = cachedEntry.response;
        if (response.type === 'AUDIT_POLICY_RESULT' && response.success && tabId) {
          chrome.tabs.sendMessage(tabId, {
            type: 'ENFORCE_DPDP_RULES',
            report: response.report
          }).catch(() => {});
        }
        return response;
      }

      if (activeAudits.has(cacheKey)) {
        console.log(`[Ssense] Deduplicating audit request for ${message.domain}`);
        const response = await activeAudits.get(cacheKey)!;
        
        if (response.type === 'AUDIT_POLICY_RESULT' && response.success && tabId) {
          chrome.tabs.sendMessage(tabId, {
            type: 'ENFORCE_DPDP_RULES',
            report: response.report
          }).catch(() => {});
        }
        return response;
      }

      const auditExecution = (async () => {
        try {
          const response = await routeRequest('AUDIT_POLICY', { domain: message.domain, policyText: message.policyText }, requestId);

          if (response.type === 'AUDIT_POLICY_RESULT' && response.success) {
            completedAuditsCache.set(cacheKey, { timestamp: Date.now(), response });
            await historyStore.recordAudit(message.domain, response.report);
            
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
      {
        const response = await routeRequest('CHAT', { domain: message.domain, userPrompt: message.userPrompt }, requestId);
        if (response.type === 'CHAT_RESULT' && response.success) {
          // Persist both sides of the exchange, extension-origin (see chat-store.ts).
          await chatStore.addMessage(message.domain, 'user', message.userPrompt);
          await chatStore.addMessage(message.domain, 'ai', response.message);
        }
        return response;
      }

    case 'GET_CHAT_HISTORY': {
      const msgs = await chatStore.getMessagesForDomain(message.domain);
      return { success: true, messages: msgs };
    }

    case 'CLEAR_CHAT_HISTORY': {
      await chatStore.clearMessagesForDomain(message.domain);
      return { success: true };
    }

    case 'OPEN_OPTIONS_PAGE': {
      chrome.runtime.openOptionsPage();
      return { success: true };
    }

    case 'GET_TRUST_SCORE':
      return await routeRequest('GET_TRUST_SCORE', { domain: message.domain }, requestId);

    default:
      throw new Error(`Unknown message type: ${message.type}`);
  }
}

// NOTE: chrome.action.onClicked never fires once manifest.json sets
// "default_popup" (popup.html) — the popup opens instead, and its
// "View Full Report" button opens the side panel from there.