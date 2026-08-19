// apps/extension/src/background/chat-store.ts
//
// Persists chatbot conversation history, per domain. Lives in the SERVICE
// WORKER (extension origin), not in the content script that renders the
// widget. This matters: a content script's IndexedDB is scoped to the
// HOST PAGE's origin, not the extension's — writing chat history there
// would silently fragment it across every site's own storage and leak
// extension data into site storage. Routing every write through a
// background message keeps it in one place, under the extension's own
// origin, matching how history-store.ts already handles audit data.

const DB_NAME = 'ssense_chat_history';
const DB_VERSION = 1;
const STORE = 'messages';

export interface ChatMessage {
  id?: number;       // autoIncrement primary key
  domain: string;
  role: 'user' | 'ai';
  text: string;
  timestamp: number;
}

let _dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (_dbPromise) return _dbPromise;
  _dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
        store.createIndex('domain', 'domain', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return _dbPromise;
}

export async function addMessage(domain: string, role: 'user' | 'ai', text: string): Promise<void> {
  if (!domain || !text) return;
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).add({ domain, role, text, timestamp: Date.now() });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function getMessagesForDomain(domain: string): Promise<ChatMessage[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const index = tx.objectStore(STORE).index('domain');
    const req = index.getAll(IDBKeyRange.only(domain));
    req.onsuccess = () => {
      const results = (req.result || []) as ChatMessage[];
      results.sort((a, b) => a.timestamp - b.timestamp);
      resolve(results);
    };
    req.onerror = () => reject(req.error);
  });
}

export async function clearMessagesForDomain(domain: string): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const index = tx.objectStore(STORE).index('domain');
    const req = index.openCursor(IDBKeyRange.only(domain));
    req.onsuccess = () => {
      const cursor = req.result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
