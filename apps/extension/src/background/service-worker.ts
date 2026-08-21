// apps/extension/src/background/service-worker.ts
import {
  getRouterConfig,
  executeHealthCheck,
  executeAuditPolicy,
  executeChat,
  executeDownloadModels,
} from './api-client';
import { subscribeNativeEvents } from './native-messaging';
import type { DaemonResponse } from '../types/native-protocol';
import * as historyStore from './history-store';
import * as chatStore from './chat-store';
import * as privacyStore from './privacy-store';

console.log('[Ssense] Service Worker initialized — Cloud-first / opt-in Offline mode.');

// Native download events are broadcast to the popup/options UI. The popup may
// close during a download; progress is also persisted so it can recover.
subscribeNativeEvents(async (event) => {
  if (event.type === 'DOWNLOAD_PROGRESS') {
    await chrome.storage.local.set({
      ssense_download_progress: {
        file: event.file,
        pct: event.pct,
        mbPerSec: event.mbPerSec,
        requestId: event.requestId,
        updatedAt: Date.now(),
      }
    });
    chrome.runtime.sendMessage({ ...event, type: 'OFFLINE_DOWNLOAD_PROGRESS' }).catch(() => {});
  } else if (event.type === 'STATUS' && event.status === 'success') {
    await chrome.storage.local.set({
      ssense_offline_mode: true,
      ssense_download_progress: { file: '', pct: 100, mbPerSec: 0, updatedAt: Date.now() },
      ssense_offline_error: '',
    });
    chrome.runtime.sendMessage({ type: 'OFFLINE_READY', message: event.message }).catch(() => {});
  } else if (event.type === 'STATUS' && event.status !== 'success') {
    await chrome.storage.local.set({
      ssense_offline_mode: false,
      ssense_offline_error: event.message || 'Offline model preparation failed.',
      ssense_download_progress: { file: '', pct: 0, mbPerSec: 0, updatedAt: Date.now() },
    });
    chrome.runtime.sendMessage({ type: 'OFFLINE_DOWNLOAD_ERROR', error: event.message || 'Offline model preparation failed.' }).catch(() => {});
  }
});

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

// Cloud-first routing with explicit Offline/Privacy Mode. The user controls
// whether inference is sent to the cloud or handled by the local daemon;
// there is no silent backend failover that could violate the user's privacy choice.
async function routeRequest(
  type: 'HEALTH_CHECK' | 'AUDIT_POLICY' | 'CHAT',
  payload: any,
  requestId: string
): Promise<DaemonResponse> {
  switch (type) {
    case 'HEALTH_CHECK': return await executeHealthCheck(requestId);
    case 'AUDIT_POLICY': return await executeAuditPolicy(payload.domain, payload.policyText, requestId);
    case 'CHAT': return await executeChat(payload.domain, payload.userPrompt, requestId);
  }
}
// NOTE: GET_TRUST_SCORE is intentionally NOT routed here. Trust score is derived
// locally from the last completed audit stored in IndexedDB (see the
// GET_TRUST_SCORE case in handleMessage below) — there is no cloud
// "/v1/trust-score" endpoint on the backend, so routing it through here would
// always 404 in Cloud Mode.

async function fetchPublicPolicyDocument(initialUrl: string, signal: AbortSignal): Promise<Response> {
  let current = new URL(initialUrl);
  for (let hop = 0; hop < 4; hop++) {
    const host = current.hostname.toLowerCase();
    if (!['http:', 'https:'].includes(current.protocol)) throw new Error('Only HTTP/HTTPS privacy-policy URLs are allowed.');
    if (/^(localhost|127\.|0\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.)/.test(host) || host === '::1' || host.startsWith('fe80:') || host.startsWith('fc')) {
      throw new Error('The privacy-policy URL points to a private or local network address.');
    }

    const response = await fetch(current.href, { credentials: 'omit', redirect: 'manual', signal });
    if (response.status >= 300 && response.status < 400) {
      const location = response.headers.get('location');
      if (!location) throw new Error('The privacy-policy server returned an invalid redirect.');
      current = new URL(location, current.href);
      continue;
    }
    return response;
  }
  throw new Error('The privacy-policy URL redirected too many times.');
}

async function handleMessage(message: any, sender: chrome.runtime.MessageSender): Promise<any> {
  const requestId = crypto.randomUUID();
  const tabId = sender.tab?.id;
  
  switch (message.type) {
    case 'PROXY_FETCH': {
      try {
        const target = new URL(String(message.url || ''));
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000); // 15s cap
        try {
          const response = await fetchPublicPolicyDocument(target.href, controller.signal);
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const contentType = response.headers.get('content-type') || '';
          if (contentType && !/(text\/html|application\/xhtml\+xml|application\/pdf)/i.test(contentType)) {
            throw new Error('The privacy-policy URL did not return an HTML or PDF document.');
          }
          const declaredLength = Number(response.headers.get('content-length') || 0);
          if (declaredLength > 8 * 1024 * 1024) throw new Error('Privacy-policy document is larger than the 8 MB safety limit.');
          const html = await response.text();
          if (html.length > 8 * 1024 * 1024) throw new Error('Privacy-policy document exceeded the 8 MB safety limit.');
          return { success: true, html };
        } finally {
          clearTimeout(timeoutId);
        }
      } catch (err: any) {
        const isTimeout = err.name === 'AbortError';
        return { success: false, error: isTimeout ? 'Policy fetch timed out after 15s' : err.message };
      }
    }

    case 'PRIVACY_SNAPSHOT': {
      if (!message.domain || !message.policyUrl || !message.policyText) {
        return { success: false, error: 'Privacy snapshot is incomplete.' };
      }
      const snapshot = await privacyStore.saveSnapshot({
        domain: message.domain,
        pageUrl: message.pageUrl || '',
        policyUrl: message.policyUrl,
        policyText: message.policyText,
      });
      return { success: true, snapshot: { ...snapshot, policyText: undefined } };
    }

    case 'GET_PRIVACY_SNAPSHOT': {
      const snapshot = await privacyStore.getSnapshot(String(message.domain || ''));
      if (!snapshot) return { success: false, error: 'No privacy-policy snapshot is available for this site yet.' };
      return { success: true, snapshot };
    }

    case 'CLEAR_PRIVACY_SNAPSHOTS': {
      await privacyStore.clearSnapshots();
      return { success: true };
    }

    case 'GET_ENGINE_CONFIG': {
      return await getRouterConfig();
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

    case 'GET_SITE_HISTORY': {
      const entry = await historyStore.getEntry(String(message.domain || ''));
      return { success: true, entry: entry || null };
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
            const router = await getRouterConfig();
            await privacyStore.markAudited(message.domain, router.offlineMode ? 'offline' : 'cloud');
            
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
              error: response.error,
              errorKind: response.type === 'ERROR' ? response.errorKind : undefined,
              retryable: response.type === 'ERROR' ? response.retryable : undefined
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

    case 'DOWNLOAD_OFFLINE': {
      if (!message.requestId) message.requestId = requestId;
      if (!navigator.onLine) {
        return { type: 'ERROR', requestId, success: false, error: 'Internet connection is required for the first offline model download.' };
      }
      const response = await executeDownloadModels(message.requestId);
      if (response.type === 'STATUS' && response.status === 'success') {
        await chrome.storage.local.set({ ssense_offline_mode: true });
      }
      return response;
    }

    case 'SET_OFFLINE_MODE': {
      const enabled = Boolean(message.enabled);
      if (!enabled) {
        await chrome.storage.local.set({ ssense_offline_mode: false });
        return { success: true, offlineMode: false };
      }
      // Enabling always goes through the explicit download flow.
      if (!message.startDownload) {
        return { success: true, offlineMode: false, needsDownload: true };
      }
      if (!navigator.onLine) {
        return { success: false, error: 'Connect to the internet to download Offline Mode models.' };
      }
      const response = await executeDownloadModels(requestId);
      if (response.type === 'STATUS' && response.status === 'success') {
        await chrome.storage.local.set({ ssense_offline_mode: true });
        return { success: true, offlineMode: true };
      }
      return response;
    }

    case 'OPEN_OPTIONS_PAGE': {
      chrome.runtime.openOptionsPage();
      return { success: true };
    }

    case 'GET_TRUST_SCORE': {
      // Trust is a property of the last completed local audit. Reading it
      // from IndexedDB avoids a second network dependency and keeps the
      // header useful even when the cloud service is temporarily down.
      const entry = (await historyStore.getAllEntries()).find(e => e.domain === message.domain);
      if (entry?.lastScore !== null && entry?.lastScore !== undefined) {
        return { type: 'TRUST_SCORE_RESULT', requestId, success: true, score: entry.lastScore };
      }
      return { type: 'TRUST_SCORE_RESULT', requestId, success: true, score: null };
    }

    default:
      throw new Error(`Unknown message type: ${message.type}`);
  }
}

// NOTE: chrome.action.onClicked never fires once manifest.json sets
// "default_popup" (popup.html) — the popup opens instead, and its
// "View Full Report" button opens the side panel from there.