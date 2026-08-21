// apps/extension/src/sidebar/components/HistoryView.tsx

import React, { useEffect, useMemo, useState } from 'react';
import type { SiteHistoryEntry } from '../../background/history-store';

type SortKey = 'recent' | 'time' | 'violations';

function formatDuration(ms: number): string {
  const totalSec = Math.round(ms / 1000);
  if (totalSec < 60) return `${totalSec}s`;
  const min = Math.floor(totalSec / 60);
  if (min < 60) return `${min}m`;
  const hr = Math.floor(min / 60);
  return `${hr}h ${min % 60}m`;
}

function formatRelativeTime(ms: number): string {
  const diff = Date.now() - ms;
  const sec = Math.floor(diff / 1000);
  if (sec < 60) return 'just now';
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  return `${day}d ago`;
}

function scoreColor(score: number | null): string {
  if (score === null) return 'var(--ssense-text-muted)';
  if (score >= 80) return 'var(--ssense-accent-emerald)';
  if (score >= 50) return 'var(--ssense-accent-amber)';
  return 'var(--ssense-accent-rose)';
}

export const HistoryView: React.FC<{ onBack: () => void }> = ({ onBack }) => {
  const [entries, setEntries] = useState<SiteHistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<SortKey>('recent');
  const [filter, setFilter] = useState('');

  const loadHistory = async () => {
    setLoading(true);
    try {
      const res = await chrome.runtime.sendMessage({ type: 'GET_HISTORY' });
      setEntries(res?.entries || []);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadHistory(); }, []);

  const handleClear = async () => {
    if (!confirm('Clear all browsing and audit history? This cannot be undone.')) return;
    await chrome.runtime.sendMessage({ type: 'CLEAR_HISTORY' });
    await loadHistory();
  };

  const visible = useMemo(() => {
    let list = entries;
    if (filter.trim()) {
      const q = filter.toLowerCase();
      list = list.filter((e) => e.domain.toLowerCase().includes(q));
    }
    const sorted = [...list];
    if (sortKey === 'recent') sorted.sort((a, b) => b.lastVisit - a.lastVisit);
    if (sortKey === 'time') sorted.sort((a, b) => b.totalTimeMs - a.totalTimeMs);
    if (sortKey === 'violations') {
      sorted.sort((a, b) => (b.lastReport?.violations.length || 0) - (a.lastReport?.violations.length || 0));
    }
    return sorted;
  }, [entries, sortKey, filter]);

  return (
    <div className="ssense-root">
      <header className="ssense-header">
        <div className="ssense-header-top">
          <div className="ssense-header-left">
            <button
              onClick={onBack}
              style={{ background: 'none', border: 'none', color: 'var(--ssense-text-secondary)', cursor: 'pointer', fontSize: 16, padding: '2px 4px' }}
              title="Back"
            >
              ←
            </button>
            <div className="ssense-header-info">
              <div className="ssense-domain">History &amp; Audit Log</div>
              <div style={{ fontSize: 11, color: 'var(--ssense-text-muted)' }}>{entries.length} sites tracked</div>
            </div>
          </div>
          <button
            onClick={handleClear}
            style={{ background: 'rgba(244,63,94,0.1)', border: '1px solid rgba(244,63,94,0.2)', color: 'var(--ssense-accent-rose)', fontSize: 10.5, fontWeight: 600, padding: '4px 8px', borderRadius: 6, cursor: 'pointer', flexShrink: 0 }}
          >
            Clear All
          </button>
        </div>
      </header>

      <div style={{ padding: '10px 16px', display: 'flex', gap: 8, borderBottom: '1px solid var(--ssense-border)' }}>
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter by domain…"
          style={{ flex: 1, background: 'rgba(255,255,255,0.05)', border: '1px solid var(--ssense-border)', borderRadius: 7, color: 'var(--ssense-text-primary)', fontSize: 12, padding: '6px 10px', outline: 'none' }}
        />
        <select
          value={sortKey}
          onChange={(e) => setSortKey(e.target.value as SortKey)}
          style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--ssense-border)', borderRadius: 7, color: 'var(--ssense-text-primary)', fontSize: 11.5, padding: '6px 8px', outline: 'none', cursor: 'pointer' }}
        >
          <option value="recent">Most recent</option>
          <option value="time">Most time spent</option>
          <option value="violations">Most violations</option>
        </select>
      </div>

      <div className="ssense-scroll" style={{ flex: 1, overflowY: 'auto', padding: '8px 12px' }}>
        {loading && (
          <div style={{ textAlign: 'center', padding: 40, color: 'var(--ssense-text-muted)', fontSize: 12 }}>
            Loading history…
          </div>
        )}

        {!loading && visible.length === 0 && (
          <div style={{ textAlign: 'center', padding: 40, color: 'var(--ssense-text-muted)', fontSize: 12 }}>
            {entries.length === 0
              ? "No sites tracked yet. Browse normally — Ssense records visits and audits as they happen."
              : 'No sites match this filter.'}
          </div>
        )}

        {visible.map((entry) => {
          const violationCount = entry.lastReport?.violations.length ?? 0;
          return (
            <div
              key={entry.domain}
              className="ssense-animate-in"
              style={{
                background: 'var(--ssense-bg-surface)',
                border: '1px solid var(--ssense-border)',
                borderRadius: 10,
                padding: '12px 14px',
                marginBottom: 8,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {entry.domain}
                  </div>
                  <div style={{ fontSize: 10.5, color: 'var(--ssense-text-muted)', marginTop: 2 }}>
                    {formatRelativeTime(entry.lastVisit)} · {entry.visitCount} visit{entry.visitCount === 1 ? '' : 's'} · {formatDuration(entry.totalTimeMs)}
                  </div>
                </div>
                <div
                  style={{
                    fontSize: 13, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
                    color: scoreColor(entry.lastScore), flexShrink: 0,
                  }}
                >
                  {entry.lastScore !== null ? entry.lastScore : '—'}
                </div>
              </div>

              {violationCount > 0 && (
                <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                  {entry.lastReport!.violations.slice(0, 3).map((v, i) => (
                    <span
                      key={i}
                      title={v.statute_reference}
                      style={{
                        fontSize: 9.5, fontWeight: 600, color: 'var(--ssense-accent-rose)',
                        background: 'rgba(244,63,94,0.1)', border: '1px solid rgba(244,63,94,0.2)',
                        borderRadius: 4, padding: '2px 6px',
                      }}
                    >
                      {v.violation_type.replace(/_/g, ' ')}
                    </span>
                  ))}
                  {violationCount > 3 && (
                    <span style={{ fontSize: 9.5, color: 'var(--ssense-text-muted)', padding: '2px 4px' }}>
                      +{violationCount - 3} more
                    </span>
                  )}
                </div>
              )}

              {entry.lastAuditAt === null && (
                <div style={{ marginTop: 8, fontSize: 10.5, color: 'var(--ssense-text-muted)', fontStyle: 'italic' }}>
                  No policy audit recorded for this site yet.
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
