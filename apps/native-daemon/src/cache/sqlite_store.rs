use anyhow::{Context, Result};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info};

use crate::messaging::protocol::DpdpAuditReport;

const TTL_SECONDS: i64 = 86400; // 24 Hours Cache Expiration

pub struct SqliteStore {
    pool: Pool<SqliteConnectionManager>,
}

impl SqliteStore {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let db_path = data_dir.join("ssense_cache.db");
        
        // SOTA FIX 1: Thread-local PRAGMAs for high-speed r2d2 pooling
        let manager = SqliteConnectionManager::file(&db_path)
            .with_init(|conn| {
                conn.execute_batch(
                    "PRAGMA busy_timeout=5000;
                     PRAGMA cache_size=-20000;
                     PRAGMA temp_store=MEMORY;"
                )
            });
            
        let pool = Pool::builder()
            .max_size(5)
            .build(manager)
            .context("Failed to create SQLite connection pool")?;

        let conn = pool.get().context("Failed to get initial DB connection")?;
        
        // SOTA FIX 2: Global PRAGMAs (WAL mode for zero-blocking concurrent reads/writes)
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

        info!("✅ [SqliteStore] Connected to Local Cache at {:?}", db_path);

        // Run initial prune
        Self::prune_expired(&conn);

        Ok(Self { pool })
    }

    /// SOTA FIX 3: Domain Normalization
    /// Strips "www." and "en." prefixes so "www.amazon.in" and "amazon.in" hit the same cache.
    fn normalize_domain(domain: &str) -> String {
        let lower = domain.to_lowercase();
        let stripped = lower.trim_start_matches("www.").trim_start_matches("en.");
        stripped.to_string()
    }

    /// SOTA FIX 4: Replaced heavy external `sha2` crate with Rust's native deterministic hasher
    /// for smaller binary size and faster local execution.
    fn hash_domain(domain: &str) -> String {
        let normalized = Self::normalize_domain(domain);
        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn now_unix() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(std::time::Duration::ZERO)
            .as_secs() as i64
    }

    /// Wipes expired records. Can be called safely from a background thread.
    pub fn prune_expired(conn: &rusqlite::Connection) {
        let expiration_threshold = Self::now_unix() - TTL_SECONDS;
        match conn.execute("DELETE FROM audits WHERE created_at <= ?1", params![expiration_threshold]) {
            Ok(purged) if purged > 0 => info!("🧹 [SqliteStore] Purged {} expired audits from disk", purged),
            Ok(_) => debug!("🧹 [SqliteStore] Cache is clean."),
            Err(e) => error!("❌ [SqliteStore] Failed to purge expired audits: {}", e),
        }
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
            _ => None,
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

                match serde_json::from_str::<DpdpAuditReport>(&json_str) {
                    Ok(report) => {
                        info!("🎯 [SqliteStore] CACHE HIT: {} (Score: {})", domain, report.dpdp_trust_score);
                        Some(report)
                    },
                    Err(e) => {
                        error!("❌ [SqliteStore] Failed to parse cached JSON for {}: {}", domain, e);
                        let _ = conn.execute("DELETE FROM audits WHERE domain_hash = ?1", params![hash]);
                        None
                    }
                }
            }
            _ => None,
        }
    }

    /// Can be called by the Chrome Extension directly when it receives a report from the Cloud Server
    /// OR by the LocalEngine when running in Offline Mode.
    pub fn save_audit(&self, domain: &str, report: &DpdpAuditReport) -> Result<()> {
        let conn = self.pool.get().context("Failed to get DB connection for save")?;
        let hash = Self::hash_domain(domain);
        let normalized_domain = Self::normalize_domain(domain); 
        
        let json_str = serde_json::to_string(report).context("Failed to serialize report to JSON")?;
        let now = Self::now_unix();

        conn.execute(
            "INSERT OR REPLACE INTO audits (domain_hash, domain_name, report_json, trust_score, created_at) 
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![hash, normalized_domain, json_str, report.dpdp_trust_score, now],
        ).context("Failed to execute INSERT OR REPLACE")?;

        info!("💾 [SqliteStore] Saved audit for {} (Score: {}) to local disk.", normalized_domain, report.dpdp_trust_score);
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