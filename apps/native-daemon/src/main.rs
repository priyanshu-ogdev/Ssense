// apps/native-daemon/src/main.rs

mod messaging;
mod cache;
mod inference;
mod model_manager;

use anyhow::Result;
use directories::ProjectDirs;
use inference::LocalEngine;
use inference::hardware_profiler::HardwareProfiler;
use model_manager::ModelManager;
use messaging::framing::{read_message, write_message};
use messaging::protocol::{DaemonRequest, DaemonResponse};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{stdin, stdout};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{error, info};

pub struct AppState {
    pub cache: cache::sqlite_store::SqliteStore,
    pub inference_engine: LocalEngine,
    pub model_manager: ModelManager,
    pub inference_lock: Mutex<()>,
}

const INFERENCE_TIMEOUT_SECS: u64 = 60;

#[tokio::main]
async fn main() -> Result<()> {
    let proj_dirs = ProjectDirs::from("com", "Ssense", "SsenseDaemon")
        .expect("Failed to resolve OS project directories");
    let data_dir = proj_dirs.data_local_dir();
    std::fs::create_dir_all(data_dir)?;

    let file_appender = tracing_appender::rolling::never(data_dir, "daemon.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter("info")
        .init();

    info!("🚀 Ssense Native Daemon booting on {:?}", data_dir);

    // 1. Initialize SQLite Cache
    let cache = cache::sqlite_store::SqliteStore::new(data_dir)?;

    // 2. Active Hardware Handshake on Boot
    let hardware_profile = match HardwareProfiler::verify_system_capabilities() {
        Ok(profile) => profile,
        Err(e) => {
            error!("❌ Hardware capability check failed: {}", e);
            // Safe fallback: 1 thread ensures the daemon can still boot and report errors to the UI
            inference::hardware_profiler::HardwareProfile { optimal_threads: 1 } 
        }
    };

    // 3. Initialize Model Manager
    let model_manager = ModelManager::new(data_dir);
    let model_path = model_manager.get_model_path();

    // 4. Initialize Local Engine (Lazy load)
    let inference_engine = LocalEngine::new(&model_path, hardware_profile.optimal_threads)?;

    let state = Arc::new(AppState {
        cache,
        inference_engine,
        model_manager,
        inference_lock: Mutex::new(()),
    });

    let (tx, mut rx) = mpsc::channel::<DaemonResponse>(100);

    // Dedicated Writer Task: Ensures stdout is written sequentially
    let writer_task = tokio::spawn(async move {
        let mut writer = stdout();
        while let Some(response) = rx.recv().await {
            match serde_json::to_vec(&response) {
                Ok(bytes) => {
                    if let Err(e) = write_message(&mut writer, &bytes).await {
                        error!("❌ Fatal stdout pipe error: {}. Exiting.", e);
                        std::process::exit(1); // Zombie prevention
                    }
                }
                Err(e) => error!("❌ Failed to serialize response: {}", e),
            }
        }
    });

    let mut reader = stdin();
    info!("✅ Listening on stdin for Chrome extension...");

    // Main Reader Loop
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
                // 🚀 SOTA: The RawEnvelope rescue mechanism
                if let Ok(raw) = serde_json::from_slice::<messaging::protocol::RawEnvelope>(&request_bytes) {
                    if let Some(req_id) = raw.request_id {
                        let _ = tx.send(DaemonResponse::Error {
                            request_id: req_id,
                            success: false,
                            error: format!("Invalid request payload: {}", e),
                        }).await;
                    }
                }
                continue;
            }
        };

        let req_id = request.request_id().to_string();
        let state_clone = Arc::clone(&state);
        let tx_clone = tx.clone();

        // Async Pre-Flight & Network Downloads
        tokio::spawn(async move {
            
            // ASYNC PHASE: Network I/O (Downloading the Model)
            match &request {
                DaemonRequest::AuditPolicy(_) | DaemonRequest::Chat(_) => {
                    if !state_clone.inference_engine.is_loaded() {
                        if let Err(e) = HardwareProfiler::verify_system_capabilities() {
                            let _ = tx_clone.send(DaemonResponse::Error {
                                request_id: req_id.clone(),
                                success: false,
                                error: format!("Hardware limits exceeded: {}", e),
                            }).await;
                            return; 
                        }

                        if let Err(e) = state_clone.model_manager.ensure_model_available().await {
                            let _ = tx_clone.send(DaemonResponse::Error {
                                request_id: req_id.clone(),
                                success: false,
                                error: format!("Model download failed: {}", e),
                            }).await;
                            return; 
                        }
                    }
                },
                _ => {} 
            }

            // SYNC PHASE: CPU-Bound Math (SQLite & C++)
            let req_id_clone = req_id.clone();
            let response = timeout(
                Duration::from_secs(INFERENCE_TIMEOUT_SECS + 10),
                tokio::task::spawn_blocking(move || {
                    route_request(request, state_clone)
                })
            )
            .await
            .unwrap_or_else(|_| {
                error!("⏱️ Task timed out after {} seconds", INFERENCE_TIMEOUT_SECS + 10);
                DaemonResponse::Error {
                    request_id: req_id_clone.clone(),
                    success: false,
                    error: format!("Operation timed out after {} seconds", INFERENCE_TIMEOUT_SECS + 10),
                }
            })
            .unwrap_or_else(|e| {
                error!("Task panic: {}", e);
                DaemonResponse::Error {
                    request_id: req_id_clone,
                    success: false,
                    error: "Internal Daemon Panic".to_string(),
                }
            });

            if tx_clone.send(response).await.is_err() {
                error!("❌ Failed to send to writer task. Zombie prevention triggered.");
                std::process::exit(1);
            }
        });
    }

    // Graceful Shutdown Sequence
    drop(tx); 
    let _ = writer_task.await; 
    drop(_guard); 

    info!("🛑 Ssense Native Daemon shut down gracefully.");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// THE BLOCKING ROUTER (Runs on OS Thread Pool)
// ═══════════════════════════════════════════════════════════════

fn route_request(request: DaemonRequest, state: Arc<AppState>) -> DaemonResponse {
    let req_id = request.request_id().to_string();

    match request {
        DaemonRequest::HealthCheck(_) => {
            let model_loaded = state.inference_engine.is_loaded();
            let metrics = state.inference_engine.get_metrics();
            
            // 🚀 SOTA POLISH: Sanitize f64 to prevent u32::MAX (Infinity) in the UI
            let safe_tps = if metrics.avg_tokens_per_second.is_finite() {
                metrics.avg_tokens_per_second as u32
            } else {
                0
            };

            DaemonResponse::HealthCheckResult {
                request_id: req_id,
                success: true,
                model_loaded,
                cache_size: state.cache.get_cache_size(),
                total_inferences: metrics.total_inferences,
                avg_tokens_per_second: safe_tps,
            }
        }

        DaemonRequest::GetTrustScore(req) => {
            DaemonResponse::TrustScoreResult {
                request_id: req_id,
                success: true,
                score: state.cache.get_trust_score(&req.domain),
            }
        }

        DaemonRequest::AuditPolicy(req) => {
            let domain = req.domain.clone();

            // Fast Path: Lock-Free Cache Check
            if let Some(cached_report) = state.cache.get_full_report(&domain) {
                info!("⚡ Cache HIT for {}", domain);
                return DaemonResponse::AuditPolicyResult {
                    request_id: req_id, success: true, report: cached_report, cached: true,
                };
            }

            // Slow Path: Global Double-Checked Lock
            let _guard = state.inference_lock.lock().unwrap();

            if let Some(cached_report) = state.cache.get_full_report(&domain) {
                info!("⚡ Cache HIT for {} (Resolved Thundering Herd)", domain);
                return DaemonResponse::AuditPolicyResult {
                    request_id: req_id, success: true, report: cached_report, cached: true,
                };
            }

            info!("🧠 Cache MISS for {}. Running Edge AI...", domain);
            match state.inference_engine.audit_policy(&domain, &req.policy_text) {
                Ok(report) => {
                    if let Err(e) = state.cache.save_audit(&domain, &report) {
                        error!("❌ Failed to cache audit for {}: {}", domain, e);
                    }
                    info!("✅ Audit complete for {} (Score: {})", domain, report.dpdp_trust_score);
                    DaemonResponse::AuditPolicyResult {
                        request_id: req_id, success: true, report, cached: false,
                    }
                }
                Err(e) => DaemonResponse::Error {
                    request_id: req_id, success: false, error: format!("Inference failed: {}", e),
                }
            }
        }

        DaemonRequest::Chat(req) => {
            let _guard = state.inference_lock.lock().unwrap();

            match state.cache.get_full_report(&req.domain) {
                Some(context) => {
                    match state.inference_engine.chat_with_context(&req.domain, &req.user_prompt, &context) {
                        Ok(message) => DaemonResponse::ChatResult { request_id: req_id, success: true, message },
                        Err(e) => DaemonResponse::Error { request_id: req_id, success: false, error: format!("Chat failed: {}", e) }
                    }
                }
                None => DaemonResponse::ChatResult {
                    request_id: req_id, success: true, 
                    message: "I need to analyze this site's privacy policy before I can answer questions. Please visit the site first.".to_string(),
                }
            }
        }
    }
}