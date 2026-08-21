// apps/extension/src/background/api-client.ts

import { nativeBridge } from './native-messaging';
import type { DaemonResponse } from '../types/native-protocol';
import { LOCAL_SERVER_URL, ONLINE_SERVER_URL, DEFAULT_SERVER_MODE, type ServerMode } from '../config';

export interface RouterConfig {
  url: string;
  resolvedMode: 'local' | 'online';
  apiKey: string;
  hmacSecret: string;
  offlineMode: boolean;
  configured: boolean;
}

// ═══════════════════════════════════════════════════════════════
// ENDPOINT RESOLUTION
// ═══════════════════════════════════════════════════════════════
// There is no editable "server URL" field anymore. The user only picks
// a mode — Auto / Local / Online — and we resolve that to one of the
// two fixed constants in config.ts. In "auto" mode we probe the local
// server first (short timeout) and fall back to the hosted one; the
// result is cached briefly so every single request doesn't re-probe.

const LOCAL_PROBE_TIMEOUT_MS = 600;
const AUTO_RESOLUTION_TTL_MS = 15000;

let cachedResolution: { mode: 'local' | 'online'; expiresAt: number } | null = null;

async function isLocalServerReachable(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), LOCAL_PROBE_TIMEOUT_MS);
    const res = await fetch(`${LOCAL_SERVER_URL}/health`, { method: 'GET', signal: controller.signal });
    clearTimeout(timeoutId);
    return res.ok;
  } catch {
    return false;
  }
}

async function resolveMode(mode: ServerMode): Promise<'local' | 'online'> {
  if (mode === 'local') return 'local';
  if (mode === 'online') return 'online';

  // mode === 'auto'
  const now = Date.now();
  if (cachedResolution && cachedResolution.expiresAt > now) {
    return cachedResolution.mode;
  }
  const reachable = await isLocalServerReachable();
  const resolved: 'local' | 'online' = reachable ? 'local' : 'online';
  cachedResolution = { mode: resolved, expiresAt: now + AUTO_RESOLUTION_TTL_MS };
  return resolved;
}

export async function getRouterConfig(): Promise<RouterConfig> {
  const data = await chrome.storage.local.get([
    'ssense_server_mode',
    'ssense_api_key',
    'ssense_hmac_secret',
    'ssense_offline_mode'
  ]);

  const mode: ServerMode = data.ssense_server_mode || DEFAULT_SERVER_MODE;
  const resolvedMode = await resolveMode(mode);
  const url = resolvedMode === 'local' ? LOCAL_SERVER_URL : ONLINE_SERVER_URL;

  return {
    url,
    resolvedMode,
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
// USER-SAFE ERROR NORMALIZATION
// ═══════════════════════════════════════════════════════════════
export type ServiceErrorKind = 'network' | 'server' | 'auth' | 'timeout' | 'model' | 'native' | 'parse' | 'unknown';

export class SsenseServiceError extends Error {
  readonly kind: ServiceErrorKind;
  readonly retryable: boolean;
  readonly status?: number;

  constructor(message: string, kind: ServiceErrorKind = 'unknown', retryable = false, status?: number) {
    super(message);
    this.name = 'SsenseServiceError';
    this.kind = kind;
    this.retryable = retryable;
    this.status = status;
  }
}

function classifyHttp(status: number, body: string): SsenseServiceError {
  if (status === 401 || status === 403) return new SsenseServiceError('Cloud authentication failed. Check your API key and HMAC secret in Settings.', 'auth', false, status);
  if (status === 429) return new SsenseServiceError('Cloud server is rate-limiting requests. Please wait a moment and try again.', 'server', true, status);
  if (status >= 500) return new SsenseServiceError('Cloud AI service is temporarily unavailable. Your page data was not retried indefinitely.', 'server', true, status);
  return new SsenseServiceError(body || `Cloud request failed (HTTP ${status}).`, 'unknown', false, status);
}

function normalizeError(err: unknown, fallback: string): SsenseServiceError {
  if (err instanceof SsenseServiceError) return err;
  const e = err as any;
  if (e?.name === 'AbortError') return new SsenseServiceError('The AI service timed out. Please try again.', 'timeout', true);
  if (String(e?.message || '').includes('Failed to fetch')) return new SsenseServiceError('The AI server cannot be reached. Check your connection or Server URL.', 'network', true);
  if (String(e?.message || '').toLowerCase().includes('native')) return new SsenseServiceError(e.message, 'native', true);
  return new SsenseServiceError(e?.message || fallback, 'unknown', false);
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
        if ((response.status >= 500 || response.status === 429) && attempt < retries) {
          clearTimeout(timeoutId);
          await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 500));
          continue; // Trigger retry
        }
        throw classifyHttp(response.status, errText);
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
          const dataLine = rawEvent.split('\n').map(line => line.trim()).find(line => line.startsWith('data:'));
          if (!dataLine) continue;
          const payload = dataLine.slice(5).trim();

          if (payload === '[DONE]') {
            if (onChunk) onChunk('', true);
            continue;
          }

          let parsed: any;
          try { parsed = JSON.parse(payload); } catch { continue; }

          if (parsed.status === 'error' || parsed.event === 'error') {
            streamError = parsed.message || parsed.data || 'Cloud reported a streaming error.';
            continue;
          }

          if (parsed.event === 'done') {
            onChunk?.('', true);
            continue;
          }

          const delta = typeof parsed.data === 'string'
            ? parsed.data
            : parsed?.choices?.[0]?.delta?.content;
          if (typeof delta === 'string') {
            accumulated += delta;
            onChunk?.(delta, false);
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
      throw normalizeError(err, 'Cloud server request failed.');
    }
  }
  throw new SsenseServiceError('Cloud server request failed after maximum retries.', 'server', true);
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
        if ((response.status >= 500 || response.status === 429) && attempt < retries) {
          await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 500));
          continue;
        }
        throw classifyHttp(response.status, errText);
      }
      return (await response.json()) as T;
    } catch (err: any) {
      clearTimeout(timeoutId);
      if (attempt < retries && (err.name === 'AbortError' || err.message?.includes('Failed to fetch'))) {
        await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 500));
        continue;
      }
      throw normalizeError(err, 'Cloud server request failed.');
    }
  }
  throw new SsenseServiceError('Cloud JSON request failed after retries.', 'server', true);
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
      success: data.status === 'online' && data.rag_ready !== false,
      modelLoaded: data.modelLoaded ?? (data.rag_ready !== false),
      cacheSize: 0,
      totalInferences: data.totalInferences ?? 0,
      avgTokensPerSecond: data.avgTokensPerSecond ?? 120,
    };
  } catch (err: any) {
    const e = normalizeError(err, 'Cloud health check failed.');
    return { type: 'ERROR', requestId, success: false, error: e.message, errorKind: e.kind, retryable: e.retryable };
  }
}

export async function executeAuditPolicy(domain: string, policyText: string, requestId: string): Promise<DaemonResponse> {
  const config = await getRouterConfig();

  if (config.offlineMode) {
    try {
      return await nativeBridge.sendRequest<DaemonResponse>({ type: 'AUDIT_POLICY', requestId, domain, policyText });
    } catch (err) {
      const e = normalizeError(err, 'Local AI model is unavailable.');
      return { type: 'ERROR', requestId, success: false, error: e.message, errorKind: e.kind, retryable: e.retryable };
    }
  }

  if (!config.configured) {
    return { type: 'ERROR', requestId, success: false, error: 'Cloud Server is not configured. Open Settings and add the API credentials.', errorKind: 'auth', retryable: false };
  }

  try {
    // The production server returns a normal JSON audit response, not SSE.
    const data = await fetchCloudJson<any>('/v1/audit', 'POST', { requestId, domain, policyText }, config);
    const report = data?.data ?? data?.report ?? data;
    if (!report || !Array.isArray(report.violations)) throw new SsenseServiceError('Cloud returned an invalid audit report. No decision was applied.', 'parse', false);
    return { type: 'AUDIT_POLICY_RESULT', requestId, success: true, report, cached: data?.source === 'memory_cache' };
  } catch (err) {
    const e = normalizeError(err, 'Cloud audit failed.');
    return { type: 'ERROR', requestId, success: false, error: e.message, errorKind: e.kind, retryable: e.retryable };
  }
}

export async function executeChat(
  domain: string,
  userPrompt: string,
  requestId: string,
  onChunk?: (token: string, isFinal: boolean) => void
): Promise<DaemonResponse> {
  const config = await getRouterConfig();

  if (config.offlineMode) {
    try {
      let text = '';
      await nativeBridge.sendChatStream({ type: 'CHAT', requestId, domain, userPrompt }, (token, isFinal) => {
        text += token;
        onChunk?.(token, isFinal);
      });
      if (!text) return { type: 'ERROR', requestId, success: false, error: 'Local AI returned no response. The model may be unavailable or malfunctioning.', errorKind: 'model', retryable: true };
      return { type: 'CHAT_RESULT', requestId, success: true, message: text };
    } catch (err) {
      const e = normalizeError(err, 'Local AI service failed.');
      onChunk?.('', true);
      return { type: 'ERROR', requestId, success: false, error: e.message, errorKind: e.kind, retryable: e.retryable };
    }
  }

  if (!config.configured) {
    const error = 'Cloud Server is not configured. Open Settings and add the API credentials.';
    onChunk?.('', true);
    return { type: 'ERROR', requestId, success: false, error, errorKind: 'auth', retryable: false };
  }

  try {
    const result = await fetchCloudStream('/v1/chat/stream', { requestId, domain, userPrompt }, config, 2, onChunk);
    if (result.error) throw new SsenseServiceError(result.error, 'server', true);
    if (!result.text.trim()) throw new SsenseServiceError('Cloud AI returned an empty response. The model may be unhealthy.', 'model', true);
    return { type: 'CHAT_RESULT', requestId, success: true, message: result.text };
  } catch (err) {
    const e = normalizeError(err, 'Chat request failed.');
    onChunk?.('', true);
    return { type: 'ERROR', requestId, success: false, error: e.message, errorKind: e.kind, retryable: e.retryable };
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