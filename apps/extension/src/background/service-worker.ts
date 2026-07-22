// apps/extension/src/background/service-worker.ts
import { sendToNativeDaemon } from './native-messaging';
import { getServerConfig, serverHealthCheck, serverAuditPolicy, serverChat, serverTrustScore } from './api-client';
import type { DaemonResponse } from '../types/native-protocol';

console.log('[Ssense] Service Worker initialized with Dual-Mode AI Routing');

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

async function routeRequest(
  type: 'HEALTH_CHECK' | 'AUDIT_POLICY' | 'CHAT' | 'GET_TRUST_SCORE',
  payload: any,
  requestId: string
): Promise<DaemonResponse> {
  const config = await getServerConfig();

  if (config.mode === 'CLOUD_SERVER') {
    switch (type) {
      case 'HEALTH_CHECK': return await serverHealthCheck(requestId);
      case 'AUDIT_POLICY': return await serverAuditPolicy(payload.domain, payload.policyText, requestId);
      case 'CHAT': return await serverChat(payload.domain, payload.userPrompt, requestId);
      case 'GET_TRUST_SCORE': return await serverTrustScore(payload.domain, requestId);
    }
  }

  if (config.mode === 'LOCAL_DAEMON') {
    return await sendToNativeDaemon({ type, requestId, ...payload });
  }

  // AUTO Mode: Try local native daemon first, failover to cloud server if offline or error
  try {
    const res = await sendToNativeDaemon({ type, requestId, ...payload });
    if (res.type === 'ERROR' && (res.error.includes('Disconnected') || res.error.includes('Offline') || res.error.includes('could not connect'))) {
      console.warn('[Ssense SW] Local daemon offline. Failing over to Cloud Virtual Server...');
      return await routeRequestToCloud(type, payload, requestId);
    }
    return res;
  } catch (err: any) {
    console.warn('[Ssense SW] Native messaging exception. Failing over to Cloud Virtual Server...', err);
    return await routeRequestToCloud(type, payload, requestId);
  }
}

async function routeRequestToCloud(
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
        const response = await fetch(message.url, { credentials: 'omit' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const html = await response.text();
        return { success: true, html };
      } catch (err: any) {
        return { success: false, error: err.message };
      }
    }

    case 'GET_ENGINE_CONFIG': {
      return await getServerConfig();
    }

    case 'SET_ENGINE_MODE': {
      await chrome.storage.local.set({ ssense_engine_mode: message.mode });
      return { success: true, mode: message.mode };
    }

    case 'CLEAR_INFERENCE_CACHE': {
      completedAuditsCache.clear();
      return { success: true, message: 'Inference cache cleared.' };
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
      return await routeRequest('CHAT', { domain: message.domain, userPrompt: message.userPrompt }, requestId);

    case 'GET_TRUST_SCORE':
      return await routeRequest('GET_TRUST_SCORE', { domain: message.domain }, requestId);

    default:
      throw new Error(`Unknown message type: ${message.type}`);
  }
}

chrome.action.onClicked.addListener(async (tab) => {
  if (tab.id) {
    await chrome.sidePanel.open({ tabId: tab.id });
  }
});