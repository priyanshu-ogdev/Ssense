// apps/extension/src/background/api-client.ts

import { nativeBridge } from './native-messaging';
import type { DaemonRequest, DaemonResponse } from '../types/native-protocol';

const DEFAULT_SERVER_URL = 'http://localhost:8080';

export interface RouterConfig {
  url: string;
  apiKey: string;
  hmacSecret: string;
  offlineMode: boolean;
  configured: boolean;
}

export async function getRouterConfig(): Promise<RouterConfig> {
  const data = await chrome.storage.local.get([
    'ssense_server_url',
    'ssense_api_key',
    'ssense_hmac_secret',
    'ssense_offline_mode'
  ]);

  return {
    url: data.ssense_server_url || DEFAULT_SERVER_URL,
    apiKey: data.ssense_api_key || '',
    hmacSecret: data.ssense_hmac_secret || '',
    offlineMode: Boolean(data.ssense_offline_mode),
    configured: Boolean(data.ssense_api_key && data.ssense_hmac_secret),
  };
}

async function computeHmacSignature(
  secret: string,
  method: string,
  endpoint: string,
  timestamp: string,
  nonce: string
): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    'raw',
    enc.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
  const payload = `${method.toUpperCase()}:${endpoint}:${timestamp}:${nonce}`;
  const sigBuffer = await crypto.subtle.sign('HMAC', key, enc.encode(payload));
  return Array.from(new Uint8Array(sigBuffer))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

// ═══════════════════════════════════════════════════════════════
// CLOUD TRANSPORT: Retries, Network Checks & SSE Streaming
// ═══════════════════════════════════════════════════════════════

async function fetchCloudStream(
  endpoint: string,
  body: any,
  config: RouterConfig,
  retries = 2,
  onChunk?: (token: string, isFinal: boolean) => void
): Promise<{ text: string; error?: string }> {

  if (!navigator.onLine) {
    throw new Error('No internet connection. Please connect to Wi-Fi or ensure Ssense Offline Mode is fully downloaded.');
  }

  const url = `${config.url.replace(/\/$/, '')}${endpoint}`;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const timestamp = Date.now().toString();
    const nonce = crypto.randomUUID();
    const signature = await computeHmacSignature(config.hmacSecret, 'POST', endpoint, timestamp, nonce);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Ssense-API-Key': config.apiKey,
          'X-Ssense-Signature': signature,
          'X-Ssense-Timestamp': timestamp,
          'X-Ssense-Nonce': nonce,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
        credentials: 'omit',
      });

      if (!response.ok || !response.body) {
        const errText = await response.text().catch(() => 'Unknown Cloud Server Error');
        if (response.status >= 500 && attempt < retries) {
          clearTimeout(timeoutId);
          await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 500));
          continue; // Trigger retry
        }
        throw new Error(`HTTP ${response.status}: ${errText}`);
      }

      // If connection succeeds, we process the stream (No retries for mid-stream failures)
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let accumulated = '';
      let streamError: string | undefined;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const rawEvent of events) {
          const line = rawEvent.trim();
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();

          if (payload === '[DONE]') {
            if (onChunk) onChunk('', true);
            continue;
          }

          let parsed: any;
          try { parsed = JSON.parse(payload); } catch { continue; }

          if (parsed.status === 'error') {
            streamError = parsed.message || 'Cloud reported a streaming error.';
            continue;
          }

          const delta = parsed?.choices?.[0]?.delta?.content;
          if (typeof delta === 'string') {
            accumulated += delta;
            if (onChunk) onChunk(delta, false);
          }
        }
      }

      clearTimeout(timeoutId);
      if (onChunk && !streamError) onChunk('', true);
      return { text: accumulated, error: streamError };

    } catch (err: any) {
      clearTimeout(timeoutId);
      if (attempt < retries && (err.name === 'AbortError' || err.message?.includes('Failed to fetch'))) {
        await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 500));
        continue; // Trigger exponential backoff
      }
      throw err;
    }
  }
  throw new Error('Cloud server request failed after maximum retries.');
}

async function fetchCloudJson<T>(endpoint: string, method: 'GET' | 'POST', body: any, config: RouterConfig, retries = 2): Promise<T> {
  if (!navigator.onLine) {
    throw new Error('No internet connection. Please connect to Wi-Fi.');
  }

  const url = `${config.url.replace(/\/$/, '')}${endpoint}`;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const timestamp = Date.now().toString();
    const nonce = crypto.randomUUID();
    const signature = await computeHmacSignature(config.hmacSecret, method, endpoint, timestamp, nonce);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
          'X-Ssense-API-Key': config.apiKey,
          'X-Ssense-Signature': signature,
          'X-Ssense-Timestamp': timestamp,
          'X-Ssense-Nonce': nonce,
        },
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errText = await response.text().catch(() => 'Unknown Cloud Error');
        if (response.status >= 500 && attempt < retries) {
          await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 500));
          continue;
        }
        throw new Error(`HTTP ${response.status}: ${errText}`);
      }
      return (await response.json()) as T;
    } catch (err: any) {
      clearTimeout(timeoutId);
      if (attempt < retries && (err.name === 'AbortError' || err.message?.includes('Failed to fetch'))) {
        await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 500));
        continue;
      }
      throw err;
    }
  }
  throw new Error('Cloud JSON request failed after retries.');
}

// ═══════════════════════════════════════════════════════════════
// THE MASTER ROUTER
// ═══════════════════════════════════════════════════════════════

export async function executeHealthCheck(requestId: string): Promise<DaemonResponse> {
  const config = await getRouterConfig();

  if (config.offlineMode) {
    return nativeBridge.sendRequest<DaemonResponse>({ type: 'HEALTH_CHECK', requestId });
  }

  if (!config.configured) {
    return { type: 'ERROR', requestId, success: false, error: 'Cloud credentials missing. Check Options.' };
  }

  try {
    const data = await fetchCloudJson<any>('/health', 'GET', null, config);
    return {
      type: 'HEALTH_CHECK_RESULT',
      requestId,
      success: data.status === 'online',
      modelLoaded: data.modelLoaded ?? true,
      cacheSize: 0,
      totalInferences: data.totalInferences ?? 0,
      avgTokensPerSecond: data.avgTokensPerSecond ?? 120,
    };
  } catch (err: any) {
    return { type: 'ERROR', requestId, success: false, error: `Cloud Offline: ${err.message}` };
  }
}

export async function executeAuditPolicy(domain: string, policyText: string, requestId: string): Promise<DaemonResponse> {
  const config = await getRouterConfig();

  if (config.offlineMode) {
    return nativeBridge.sendRequest<DaemonResponse>({ type: 'AUDIT_POLICY', requestId, domain, policyText });
  }

  try {
    const { text, error } = await fetchCloudStream('/v1/audit', { requestId, domain, policyText }, config);
    if (error) return { type: 'ERROR', requestId, success: false, error: `Cloud Audit Error: ${error}` };

    const report = JSON.parse(text);
    return { type: 'AUDIT_POLICY_RESULT', requestId, success: true, report, cached: false };
  } catch (err: any) {
    return { type: 'ERROR', requestId, success: false, error: `Cloud Audit Error: ${err.message}` };
  }
}

export async function executeChat(
  domain: string,
  userPrompt: string,
  requestId: string,
  onChunk: (token: string, isFinal: boolean) => void
): Promise<void> {
  const config = await getRouterConfig();

  if (config.offlineMode) {
    await nativeBridge.sendChatStream({ type: 'CHAT', requestId, domain, userPrompt }, onChunk);
    return;
  }

  try {
    const { error } = await fetchCloudStream('/v1/chat', { requestId, domain, userPrompt }, config, 2, onChunk);
    if (error) throw new Error(error);
  } catch (err: any) {
    onChunk(`\n\n[System Error: ${err.message}]`, true);
  }
}

export async function executeTrustScore(domain: string, requestId: string): Promise<DaemonResponse> {
  const config = await getRouterConfig();

  if (config.offlineMode) {
    return nativeBridge.sendRequest<DaemonResponse>({ type: 'GET_TRUST_SCORE', requestId, domain });
  }

  try {
    const data = await fetchCloudJson<any>('/v1/trust-score', 'POST', { requestId, domain }, config);
    return data as DaemonResponse;
  } catch (err: any) {
    return { type: 'ERROR', requestId, success: false, error: `Cloud Trust Score Error: ${err.message}` };
  }
}

export async function executeDownloadModels(requestId: string): Promise<DaemonResponse> {
  if (!navigator.onLine) {
    return {
      type: 'ERROR',
      requestId,
      success: false,
      error: 'Cannot download Offline Models. You have no internet connection.'
    };
  }
  return nativeBridge.sendRequest<DaemonResponse>({ type: 'DOWNLOAD_MODELS', requestId });
}