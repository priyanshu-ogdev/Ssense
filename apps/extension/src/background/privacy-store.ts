const DB_NAME = 'ssense_privacy';
const DB_VERSION = 1;
const STORE = 'snapshots';

export interface PrivacySnapshot {
  domain: string;
  policyUrl: string;
  pageUrl: string;
  extractedAt: number;
  textLength: number;
  textSha256: string;
  policyText: string;
  transport: 'cloud' | 'offline' | 'unknown';
  lastAuditAt: number | null;
}

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: 'domain' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

async function sha256(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, '0')).join('');
}

export async function saveSnapshot(input: {
  domain: string;
  policyUrl: string;
  pageUrl: string;
  policyText: string;
  transport?: 'cloud' | 'offline' | 'unknown';
}): Promise<PrivacySnapshot> {
  const db = await openDb();
  const existing = await getSnapshot(input.domain);
  const snapshot: PrivacySnapshot = {
    domain: input.domain,
    policyUrl: input.policyUrl,
    pageUrl: input.pageUrl,
    extractedAt: Date.now(),
    textLength: input.policyText.length,
    textSha256: await sha256(input.policyText),
    policyText: input.policyText,
    transport: input.transport ?? existing?.transport ?? 'unknown',
    lastAuditAt: existing?.lastAuditAt ?? null,
  };
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(snapshot);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  return snapshot;
}

export async function getSnapshot(domain: string): Promise<PrivacySnapshot | null> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(domain);
    req.onsuccess = () => resolve(req.result ?? null);
    req.onerror = () => reject(req.error);
  });
}

export async function markAudited(domain: string, transport: 'cloud' | 'offline'): Promise<void> {
  const snapshot = await getSnapshot(domain);
  if (!snapshot) return;
  snapshot.transport = transport;
  snapshot.lastAuditAt = Date.now();
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(snapshot);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function clearSnapshots(): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
