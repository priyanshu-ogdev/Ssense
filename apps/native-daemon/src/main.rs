mod messaging;
mod cache;

use anyhow::Result;
use directories::ProjectDirs;
use messaging::framing::{read_message, write_message};
use messaging::protocol::{DaemonRequest, DaemonResponse, DpdpAuditReport};
use std::sync::Arc;
use tokio::io::{stdin, stdout};
use tokio::sync::mpsc;
use tracing::{error, info};

// Global application state shared across async and blocking threads
pub struct AppState {
    pub cache: cache::sqlite_store::SqliteStore,
    // pub inference: inference::HybridRouter, // Phase 3
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Resolve OS-specific data directory for logging
    let proj_dirs = ProjectDirs::from("com", "Ssense", "SsenseDaemon")
        .expect("Failed to resolve OS project directories");
    let data_dir = proj_dirs.data_local_dir();
    std::fs::create_dir_all(data_dir)?;

    // 2. Initialize Non-Blocking File Appender
    let file_appender = tracing_appender::rolling::never(data_dir, "daemon.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter("info")
        .init();

    info!("🚀 Ssense Native Daemon booting on {:?}", data_dir);

    // 3. Initialize State (SQLite Cache Pool)
    let cache = cache::sqlite_store::SqliteStore::new(data_dir)?;
    let state = Arc::new(AppState { cache });

    // 4. MPSC Channel for non-blocking stdout multiplexing
    let (tx, mut rx) = mpsc::channel::<DaemonResponse>(100);

    // Dedicated Writer Task: Ensures stdout is written sequentially
    let writer_task = tokio::spawn(async move {
        let mut writer = stdout();
        while let Some(response) = rx.recv().await {
            match serde_json::to_vec(&response) {
                Ok(bytes) => {
                    if let Err(e) = write_message(&mut writer, &bytes).await {
                        error!("❌ Fatal stdout pipe error: {}. Exiting.", e);
                        std::process::exit(1); // Prevent Zombie Process
                    }
                }
                Err(e) => error!("❌ Failed to serialize response: {}", e),
            }
        }
    });

    let mut reader = stdin();
    info!("✅ Listening on stdin for Chrome extension...");

    // 5. Main Reader Loop
    loop {
        let request_bytes = match read_message(&mut reader).await {
            Ok(bytes) => bytes,
            Err(_) => {
                info!("👋 Chrome disconnected (EOF). Shutting down gracefully.");
                break;
            }
        };

        let request: DaemonRequest = match serde_json::from_slice(&request_bytes) {
            Ok(req) => req,
            Err(e) => {
                error!("❌ JSON parse error: {}", e);
                continue;
            }
        };

        let req_id = request.request_id().to_string();
        let req_type = match &request {
            DaemonRequest::AuditPolicy(_) => "AUDIT_POLICY",
            DaemonRequest::Chat(_) => "CHAT",
            DaemonRequest::GetTrustScore(_) => "GET_TRUST_SCORE",
            DaemonRequest::HealthCheck(_) => "HEALTH_CHECK",
        };
        
        info!("📨 Processing: {} (ID: {})", req_type, req_id);

        let state_clone = Arc::clone(&state);
        let tx_clone = tx.clone();
        let req_id_clone = req_id.clone();

        // 6. Fire and forget to Thread Pool
        tokio::spawn(async move {
            let response = tokio::task::spawn_blocking(move || {
                route_request(request, state_clone)
            })
            .await
            .unwrap_or_else(|e| {
                error!("Task panic: {}", e);
                DaemonResponse::Error {
                    request_id: req_id_clone.clone(),
                    success: false,
                    error: "Internal Daemon Panic".to_string(),
                }
            });

            // If the channel is dead, the writer task died. Zombie prevention.
            if tx_clone.send(response).await.is_err() {
                error!("❌ Failed to send to writer task. Zombie prevention triggered.");
                std::process::exit(1);
            }
        });
    }

    // 7. Graceful Shutdown Sequence
    drop(tx); // Drop the sender so the receiver loop in writer_task terminates
    let _ = writer_task.await; // Wait for any pending messages to finish writing to Chrome
    drop(_guard); // Explicitly flush the background logging buffer to SSD

    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// THE BLOCKING ROUTER (Runs on OS Thread Pool)
// ═══════════════════════════════════════════════════════════════

fn route_request(request: DaemonRequest, state: Arc<AppState>) -> DaemonResponse {
    let req_id = request.request_id().to_string();

    match request {
        DaemonRequest::HealthCheck(_) => DaemonResponse::HealthCheckResult {
            request_id: req_id,
            success: true,
            model_loaded: false, // Will become dynamic in Phase 3
            cache_size: state.cache.get_cache_size(),
        },
        DaemonRequest::GetTrustScore(req) => DaemonResponse::TrustScoreResult {
            request_id: req_id,
            success: true,
            score: state.cache.get_trust_score(&req.domain),
        },
        DaemonRequest::AuditPolicy(req) => DaemonResponse::AuditPolicyResult {
            request_id: req_id,
            success: true,
            // CRITICAL FIX: Explicitly return a strict schema object, eliminating the null trap.
            report: DpdpAuditReport {
                global_legal_reasoning: "Phase 3 Placeholder: LLM Engine not yet initialized.".to_string(),
                violations: vec![],
                dpdp_trust_score: 100,
            },
            cached: false,
        },
        DaemonRequest::Chat(req) => DaemonResponse::ChatResult {
            request_id: req_id,
            success: true,
            message: format!("Echo: {}", req.user_prompt), 
        },
    }
}