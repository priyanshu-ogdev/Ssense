// apps/extension/src/background/native-messaging.ts
import type { DaemonRequest, DaemonResponse } from '../types/native-protocol';

const NATIVE_HOST_NAME = 'com.ssense.daemon';
let port: chrome.runtime.Port | null = null;

const pendingRequests = new Map<string, { 
  resolve: (value: DaemonResponse) => void; 
  reject: (reason: Error) => void;
  timeout: ReturnType<typeof setTimeout>;
}>();

function getPort(): chrome.runtime.Port {
  if (!port) {
    port = chrome.runtime.connectNative(NATIVE_HOST_NAME);
    
    port.onMessage.addListener((message: DaemonResponse) => {
      const pending = pendingRequests.get(message.requestId);
      if (pending) {
        clearTimeout(pending.timeout);
        pendingRequests.delete(message.requestId);
        pending.resolve(message);
      }
    });
    
    port.onDisconnect.addListener(() => {
      const error = chrome.runtime.lastError;
      console.error('[Ssense] Native host disconnected:', error?.message || 'Unknown error');
      port = null;
      
      // Flush all hanging promises instantly
      for (const [, pending] of pendingRequests) {
        clearTimeout(pending.timeout);
        pending.reject(new Error(error?.message || 'Native host disconnected'));
      }
      pendingRequests.clear();
    });
  }
  return port;
}

export async function sendToNativeDaemon(request: DaemonRequest): Promise<DaemonResponse> {
  return new Promise((resolve, reject) => {
    // Timeout defined BEFORE try block to guarantee memory leak prevention
    const timeoutId = setTimeout(() => {
      if (pendingRequests.has(request.requestId)) {
        pendingRequests.delete(request.requestId);
        reject(new Error('Native daemon timeout (45s). The AI is likely still processing.'));
      }
    }, 45000);

    try {
      const p = getPort();
      pendingRequests.set(request.requestId, { resolve, reject, timeout: timeoutId });
      p.postMessage(request);
    } catch (err: any) {
      clearTimeout(timeoutId);
      pendingRequests.delete(request.requestId);
      
      // 🚀 SOTA FIX 2: Aggressively murder Zombie Ports.
      // If postMessage throws, the port is dead. Force a clean state.
      if (port) {
        try { port.disconnect(); } catch (e) { /* ignore */ }
        port = null;
      }
      
      reject(new Error(`Native IPC failed: ${err.message}`));
    }
  });
}