// apps/extension/src/background/api-client.ts
import type { DaemonResponse } from '../types/native-protocol';

const DEFAULT_SERVER_URL = 'http://localhost:8080';
const DEFAULT_API_KEY = 'ssense_dev_key_2026';
const DEFAULT_HMAC_SECRET = 'ssense_secret_key_2026_prod';

export async function getServerConfig(): Promise<{ url: string; apiKey: string; hmacSecret: string; mode: 'LOCAL_DAEMON' | 'CLOUD_SERVER' | 'AUTO' }> {
  const data = await chrome.storage.local.get(['ssense_server_url', 'ssense_api_key', 'ssense_hmac_secret', 'ssense_engine_mode']);
  return {
    url: data.ssense_server_url || DEFAULT_SERVER_URL,
    apiKey: data.ssense_api_key || DEFAULT_API_KEY,
    hmacSecret: data.ssense_hmac_secret || DEFAULT_HMAC_SECRET,
    mode: data.ssense_engine_mode || 'AUTO',
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

async function fetchServer<T>(endpoint: string, method: 'GET' | 'POST' = 'GET', body?: any, retries = 2): Promise<T> {
  const config = await getServerConfig();
  const url = `${config.url.replace(/\/$/, '')}${endpoint}`;
  
  const timestamp = Date.now().toString();
  const nonce = crypto.randomUUID();
  const signature = await computeHmacSignature(
    config.hmacSecret || DEFAULT_HMAC_SECRET,
    method,
    endpoint,
    timestamp,
    nonce
  );

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Ssense-API-Key': config.apiKey,
    'X-Ssense-Signature': signature,
    'X-Ssense-Timestamp': timestamp,
    'X-Ssense-Nonce': nonce,
  };

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
        credentials: 'omit',
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errText = await response.text().catch(() => 'Unknown Server Error');
        if (response.status >= 500 && attempt < retries) {
          await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 500));
          continue;
        }
        throw new Error(`HTTP ${response.status}: ${errText}`);
      }

      return (await response.json()) as T;
    } catch (err: any) {
      clearTimeout(timeoutId);
      if (
        attempt < retries &&
        (err.name === 'AbortError' ||
          err.message?.includes('Failed to fetch') ||
          err.message?.includes('NetworkError'))
      ) {
        await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 500));
        continue;
      }
      if (err.name === 'AbortError') {
        throw new Error('Server request timed out after 60 seconds.');
      }
      throw err;
    }
  }
  throw new Error('Server request failed after retries.');
}

export async function serverHealthCheck(requestId: string): Promise<DaemonResponse> {
  try {
    const data = await fetchServer<any>('/health');
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
    return {
      type: 'ERROR',
      requestId,
      success: false,
      error: `Cloud Server Offline: ${err.message}`,
    };
  }
}

export async function serverAuditPolicy(domain: string, policyText: string, requestId: string): Promise<DaemonResponse> {
  try {
    const data = await fetchServer<any>('/v1/audit', 'POST', {
      requestId,
      domain,
      policyText,
    });
    return data as DaemonResponse;
  } catch (err: any) {
    return {
      type: 'ERROR',
      requestId,
      success: false,
      error: `Cloud Audit Error: ${err.message}`,
    };
  }
}

export async function serverChat(domain: string, userPrompt: string, requestId: string): Promise<DaemonResponse> {
  try {
    const data = await fetchServer<any>('/v1/chat', 'POST', {
      requestId,
      domain,
      userPrompt,
    });
    return data as DaemonResponse;
  } catch (err: any) {
    return {
      type: 'ERROR',
      requestId,
      success: false,
      error: `Cloud Chat Error: ${err.message}`,
    };
  }
}

export async function serverTrustScore(domain: string, requestId: string): Promise<DaemonResponse> {
  try {
    const data = await fetchServer<any>('/v1/trust-score', 'POST', {
      requestId,
      domain,
    });
    return data as DaemonResponse;
  } catch (err: any) {
    return {
      type: 'ERROR',
      requestId,
      success: false,
      error: `Cloud Trust Score Error: ${err.message}`,
    };
  }
}
