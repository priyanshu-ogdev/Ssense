// apps/native-daemon/src/cache/sqlite_store.rs

use anyhow::{Context, Result};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info};

use crate::messaging::protocol::DpdpAuditReport;

const TTL_SECONDS: i64 = 86400; // 24 Hours

pub struct SqliteStore {
    pool: Pool<SqliteConnectionManager>,
}

impl SqliteStore {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let db_path = data_dir.join("ssense_cache.db");
        
        // 🚀 SOTA FIX 3: Thread-local PRAGMAs only. 
        // Prevents database-level lock contention when r2d2 spawns new threads.
        let manager = SqliteConnectionManager::file(&db_path)
            .with_init(|conn| {
                conn.execute_batch(
                    "PRAGMA busy_timeout=5000;        -- Wait 5s for locks
                     PRAGMA cache_size=-20000;        -- Allocate 20MB of RAM for page cache
                     PRAGMA temp_store=MEMORY;        -- Force temp tables/sorts into RAM"
                )
            });
            
        let pool = Pool::builder()
            .max_size(5)
            .build(manager)
            .context("Failed to create SQLite connection pool")?;

        let conn = pool.get().context("Failed to get initial DB connection")?;
        
        // 🚀 SOTA FIX 3: Global PRAGMAs and Schema execution.
        // These alter the physical .db file and only need to be run once on startup.
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS audits (
                domain_hash TEXT PRIMARY KEY,
                domain_name TEXT NOT NULL,
                report_json TEXT NOT NULL,
                trust_score INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );"
        ).context("Failed to initialize SQLite schema")?;

        info!("✅ SQLite Connection Pool initialized at {:?}", db_path);

        // 🚀 SOTA FIX 1: Active Pruning on Boot.
        // Prevents the "Lazy Purge" disk leak by wiping all expired audits instantly.
        let now = Self::now_unix();
        let expiration_threshold = now - TTL_SECONDS;
        match conn.execute("DELETE FROM audits WHERE created_at <= ?1", params![expiration_threshold]) {
            Ok(purged) if purged > 0 => info!("🧹 Active Pruning: Purged {} expired audits from disk", purged),
            Ok(_) => debug!("🧹 Active Pruning: Cache is clean."),
            Err(e) => error!("❌ Failed to purge expired audits: {}", e),
        }

        Ok(Self { pool })
    }

    /// 🚀 SOTA FIX 2: Domain Normalization
    /// Strips "www." to ensure 'www.amazon.com' and 'amazon.com' share the same cache hit.
    fn normalize_domain(domain: &str) -> String {
        domain.to_lowercase().trim_start_matches("www.").to_string()
    }

    fn hash_domain(domain: &str) -> String {
        let normalized = Self::normalize_domain(domain);
        let mut hasher = Sha256::new();
        hasher.update(normalized.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn now_unix() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(std::time::Duration::ZERO)
            .as_secs() as i64
    }

    pub fn get_trust_score(&self, domain: &str) -> Option<i32> {
        let conn = self.pool.get().ok()?;
        let hash = Self::hash_domain(domain);

        let result = conn.query_row(
            "SELECT trust_score, created_at FROM audits WHERE domain_hash = ?1",
            params![hash],
            |row| Ok((row.get::<_, i32>(0)?, row.get::<_, i64>(1)?)),
        );

        match result {
            Ok((score, created_at)) => {
                if Self::now_unix() - created_at > TTL_SECONDS {
                    let _ = conn.execute("DELETE FROM audits WHERE domain_hash = ?1", params![hash]);
                    return None;
                }
                Some(score)
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(e) => {
                error!("❌ SQLite read error (trust_score): {}", e);
                None
            }
        }
    }

    pub fn get_full_report(&self, domain: &str) -> Option<DpdpAuditReport> {
        let conn = self.pool.get().ok()?;
        let hash = Self::hash_domain(domain);

        let result = conn.query_row(
            "SELECT report_json, created_at FROM audits WHERE domain_hash = ?1",
            params![hash],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        );

        match result {
            Ok((json_str, created_at)) => {
                if Self::now_unix() - created_at > TTL_SECONDS {
                    let _ = conn.execute("DELETE FROM audits WHERE domain_hash = ?1", params![hash]);
                    return None;
                }

                match serde_json::from_str(&json_str) {
                    Ok(report) => Some(report),
                    Err(e) => {
                        error!("❌ Failed to parse cached JSON for {}: {}", domain, e);
                        let _ = conn.execute("DELETE FROM audits WHERE domain_hash = ?1", params![hash]);
                        None
                    }
                }
            }
            _ => None,
        }
    }

    pub fn save_audit(&self, domain: &str, report: &DpdpAuditReport) -> Result<()> {
        let conn = self.pool.get().context("Failed to get DB connection for save")?;
        let hash = Self::hash_domain(domain);
        // We still save the normalized domain name for logging/debugging context
        let normalized_domain = Self::normalize_domain(domain); 
        let json_str = serde_json::to_string(report).context("Failed to serialize report to JSON")?;
        let now = Self::now_unix();

        conn.execute(
            "INSERT OR REPLACE INTO audits (domain_hash, domain_name, report_json, trust_score, created_at) 
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![hash, normalized_domain, json_str, report.dpdp_trust_score, now],
        ).context("Failed to execute INSERT OR REPLACE")?;

        info!("💾 Cached audit for {} (Score: {})", normalized_domain, report.dpdp_trust_score);
        Ok(())
    }

    pub fn get_cache_size(&self) -> usize {
        if let Ok(conn) = self.pool.get() {
            conn.query_row("SELECT COUNT(*) FROM audits", [], |row| row.get::<_, usize>(0)).unwrap_or(0)
        } else {
            0
        }
    }
}