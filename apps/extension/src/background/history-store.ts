// apps/extension/src/background/history-store.ts
//
// Durable history for every site visited: audit results, violation counts,
// and time spent. Uses IndexedDB (available inside MV3 service workers)
// rather than chrome.storage.local, since this data grows unbounded over
// normal use and chrome.storage.local has a much smaller practical quota.

import type { DpdpAuditReport } from '../types/native-protocol';

const DB_NAME = 'ssense_history';
const DB_VERSION = 1;
const STORE = 'site_visits';

export interface SiteHistoryEntry {
  domain: string;
  firstVisit: number;       // epoch ms
  lastVisit: number;        // epoch ms
  visitCount: number;
  totalTimeMs: number;      // accumulated active+focused time on this domain
  lastScore: number | null;
  lastReport: DpdpAuditReport | null;
  lastAuditAt: number | null;
}

let _dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (_dbPromise) return _dbPromise;
  _dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'domain' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return _dbPromise;
}

async function getEntry(domain: string): Promise<SiteHistoryEntry | null> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(domain);
    req.onsuccess = () => resolve(req.result ?? null);
    req.onerror = () => reject(req.error);
  });
}

async function putEntry(entry: SiteHistoryEntry): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(entry);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/** Call once per navigation to a domain (page load / tab activation). */
export async function recordVisit(domain: string): Promise<void> {
  if (!domain) return;
  const now = Date.now();
  const existing = await getEntry(domain);
  if (existing) {
    existing.visitCount += 1;
    existing.lastVisit = now;
    await putEntry(existing);
  } else {
    await putEntry({
      domain,
      firstVisit: now,
      lastVisit: now,
      visitCount: 1,
      totalTimeMs: 0,
      lastScore: null,
      lastReport: null,
      lastAuditAt: null,
    });
  }
}

/** Accumulate active-tab time for a domain. Called periodically, small deltas. */
export async function addTime(domain: string, deltaMs: number): Promise<void> {
  if (!domain || deltaMs <= 0) return;
  const existing = await getEntry(domain);
  if (existing) {
    existing.totalTimeMs += deltaMs;
    await putEntry(existing);
  } else {
    // Time arrived before any recorded visit (edge case) — create a minimal entry.
    await putEntry({
      domain,
      firstVisit: Date.now(),
      lastVisit: Date.now(),
      visitCount: 1,
      totalTimeMs: deltaMs,
      lastScore: null,
      lastReport: null,
      lastAuditAt: null,
    });
  }
}

/** Record the outcome of a completed audit for a domain. */
export async function recordAudit(domain: string, report: DpdpAuditReport): Promise<void> {
  if (!domain) return;
  const now = Date.now();
  const existing = await getEntry(domain);
  const base: SiteHistoryEntry = existing ?? {
    domain,
    firstVisit: now,
    lastVisit: now,
    visitCount: 1,
    totalTimeMs: 0,
    lastScore: null,
    lastReport: null,
    lastAuditAt: null,
  };
  base.lastScore = report.dpdp_trust_score;
  base.lastReport = report;
  base.lastAuditAt = now;
  await putEntry(base);
}

export async function getAllEntries(): Promise<SiteHistoryEntry[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}

export async function clearAllEntries(): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
