mod messaging;
mod cache;
mod inference;
mod model_manager;
mod rag_engine;

use anyhow::Result;
use directories::ProjectDirs;
use inference::LocalEngine;
use inference::hardware_profiler::HardwareProfiler;
use model_manager::ModelManager;
use messaging::framing::{read_message, write_message};
use messaging::protocol::{DaemonRequest, DaemonResponse};
use rag_engine::RagEngine;

use std::sync::Arc;
use std::time::Duration;
use tokio::io::{stdin, stdout};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::time::timeout;
use tracing::{error, info, warn};

pub struct AppState {
    pub cache: Arc<cache::sqlite_store::SqliteStore>,
    pub inference_engine: Arc<LocalEngine>,
    pub model_manager: Arc<ModelManager>,
    pub rag_engine: Arc<RwLock<Option<RagEngine>>>,
    // Strict mutex preventing concurrent GPU requests from melting VRAM
    pub inference_lock: Mutex<()>,
    // Detected once at boot (verify_system_capabilities); surfaced in HealthCheck responses
    // so the popup/side panel can show whether inference is running on GPU or CPU-only.
    pub has_gpu_acceleration: bool,
}

const INFERENCE_TIMEOUT_SECS: u64 = 180; // Allow 3 mins for deep reasoning on slow laptops

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Resolve Safe OS Directories
    let proj_dirs = ProjectDirs::from("com", "Ssense", "ssense-native-daemon")
        .expect("Failed to resolve OS project directories");
    let data_dir = proj_dirs.data_dir();
    std::fs::create_dir_all(data_dir)?;

    // 2. Telemetry & Logging
    let file_appender = tracing_appender::rolling::never(data_dir, "daemon.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        // Configurable via the RUST_LOG env var (e.g. RUST_LOG=debug or
        // RUST_LOG=ssense_native_daemon::inference=trace for per-token generation
        // detail) — falls back to "info" if unset, same as before.
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .init();

    info!("🚀 [Main] Ssense Native Daemon V3.0 booting on {:?}", data_dir);

    // 3. Initialize OS Subsystems
    let cache = Arc::new(cache::sqlite_store::SqliteStore::new(data_dir)?);
    let model_manager = Arc::new(ModelManager::new()?);

    let hardware_profile = match HardwareProfiler::verify_system_capabilities() {
        Ok(profile) => profile,
        Err(e) => {
            error!("❌ [Main] Hardware capability check failed: {}", e);
            std::process::exit(1);
        }
    };

    let audit_model_path = model_manager.get_artifact_path("audit-model.Q4_K_M.gguf");
    let chatbot_model_path = model_manager.get_artifact_path("chatbot-model.Q4_K_M.gguf");

    // Boot Engine (Will only map models to memory when first requested)
    let inference_engine = Arc::new(LocalEngine::new(
        &audit_model_path, 
        &chatbot_model_path, 
        hardware_profile.optimal_threads
    )?);

    // Boot RAG if Safetensors exist locally
    let json_path = model_manager.get_artifact_path("dpdp_index.json");
    let safe_path = model_manager.get_artifact_path("dpdp_embeddings.safetensors");
    let rag_engine = if json_path.exists() && safe_path.exists() {
        match RagEngine::new(&json_path, &safe_path) {
            Ok(rag) => Some(rag),
            Err(e) => {
                error!("❌ [Main] Failed to initialize Local RAG Engine: {}", e);
                None
            }
        }
    } else {
        warn!("⚠️ [Main] RAG Safetensors missing. Chatbot will run without retrieval until models are downloaded.");
        None
    };

    let state = Arc::new(AppState {
        cache,
        inference_engine,
        model_manager,
        rag_engine: Arc::new(RwLock::new(rag_engine)),
        inference_lock: Mutex::new(()),
        has_gpu_acceleration: hardware_profile.has_gpu_acceleration,
    });

    // 4. Native Messaging IPC Pipes
    let (tx, mut rx) = mpsc::channel::<DaemonResponse>(100);

    // ── BACKGROUND WRITER TASK ──
    // Writes responses to Chrome asynchronously to prevent blocking the engine
    let writer_task = tokio::spawn(async move {
        let mut writer = stdout();
        while let Some(response) = rx.recv().await {
            match serde_json::to_vec(&response) {
                Ok(bytes) => {
                    if let Err(e) = write_message(&mut writer, &bytes).await {
                        error!("❌ [Main] Fatal stdout pipe error: {}. Chrome likely killed extension. Exiting.", e);
                        std::process::exit(1);
                    }
                }
                Err(e) => error!("❌ [Main] Failed to serialize IPC response: {}", e),
            }
        }
    });

    let mut reader = stdin();
    info!("✅ [Main] Listening on IPC stdin for Chrome Native Messages...");

    // ── MAIN EVENT LOOP ──
    loop {
        let request_bytes = match read_message(&mut reader).await {
            Ok(bytes) => bytes,
            Err(_) => {
                info!("👋 [Main] Chrome disconnected (EOF). Shutting down daemon cleanly.");
                break;
            }
        };

        let request: DaemonRequest = match serde_json::from_slice(&request_bytes) {
            Ok(req) => req,
            Err(e) => {
                error!("❌ [Main] IPC JSON parse error: {}", e);
                if let Ok(raw) = serde_json::from_slice::<messaging::protocol::RawEnvelope>(&request_bytes) {
                    if let Some(req_id) = raw.request_id {
                        let _ = tx.send(DaemonResponse::Error {
                            request_id: req_id,
                            success: false,
                            error: format!("Invalid Native Message JSON: {}", e),
                        }).await;
                    }
                }
                continue;
            }
        };

        let req_id = request.request_id().to_string();
        let state_clone = Arc::clone(&state);
        let tx_clone = tx.clone();
        
        let json_path_clone = json_path.clone();
        let safe_path_clone = safe_path.clone();

        // ── REQUEST ROUTER ──
        tokio::spawn(async move {
            match request {
                
                // 1. UI Toggles Offline Download
                DaemonRequest::DownloadModels(_) => {
                    info!("📥 [Main] User requested Offline Models Sync...");
                    let result = state_clone.model_manager.ensure_all_available(Some(tx_clone.clone()), Some(req_id.clone())).await;
                    
                    if let Err(e) = result {
                        let _ = tx_clone.send(DaemonResponse::Error {
                            request_id: req_id.clone(),
                            success: false,
                            error: format!("Network download failed: {}", e),
                        }).await;
                        return;
                    }

                    // Dynamically boot RAG once downloaded
                    let needs_rag = state_clone.rag_engine.read().await.is_none();
                    if needs_rag {
                        if let Ok(rag) = RagEngine::new(&json_path_clone, &safe_path_clone) {
                            let mut guard = state_clone.rag_engine.write().await;
                            *guard = Some(rag);
                            info!("✅ [Main] Dynamically loaded Local RAG engine post-download.");
                        }
                    }
                }

                // 1b. UI requests the in-flight download stop early (partial file kept for resume)
                DaemonRequest::PauseDownload(_) => {
                    info!("⏸ [Main] User requested to pause the offline model download...");
                    state_clone.model_manager.request_pause();
                    // Ack THIS request immediately (its own request_id) so the popup's
                    // pause button doesn't hang waiting on a reply. The original
                    // DOWNLOAD_MODELS request (a *different* request_id) is resolved
                    // separately, on its own timeline, once ensure_all_available()
                    // actually finishes flushing and sends its terminal "paused" Status.
                    let _ = tx_clone.send(DaemonResponse::Status {
                        status: "pause_acknowledged".to_string(),
                        message: "Pause requested.".to_string(),
                        request_id: Some(req_id.clone()),
                    }).await;
                }

                // 2. Handle LLM Execution in a Blocking Thread
                _ => {
                    let state_for_blocking = state_clone.clone();
                    let tx_for_blocking = tx_clone.clone();

                    let response_future = tokio::task::spawn_blocking(move || {
                        // SOTA FIX: Acquire the global inference lock. 
                        // If Chrome sends an Audit and a Chat simultaneously, they queue here instead of OOM-crashing.
                        let lock_handle = state_for_blocking.clone();
                        let _guard = lock_handle.inference_lock.blocking_lock();
                        
                        let rt = tokio::runtime::Handle::current();
                        route_request(request, state_for_blocking, tx_for_blocking, rt)
                    });

                    let final_response = timeout(Duration::from_secs(INFERENCE_TIMEOUT_SECS), response_future).await
                        .unwrap_or_else(|_| {
                            error!("⏱️ [Main] LLM Engine timed out after {}s", INFERENCE_TIMEOUT_SECS);
                            Ok(DaemonResponse::Error {
                                request_id: req_id.clone(),
                                success: false,
                                error: format!("LLM inference timed out after {}s", INFERENCE_TIMEOUT_SECS),
                            })
                        })
                        .unwrap_or_else(|e| {
                            error!("❌ [Main] LLM Engine panic: {}", e);
                            DaemonResponse::Error {
                                request_id: req_id.clone(),
                                success: false,
                                error: "Internal Local Engine Panic".to_string(),
                            }
                        });

                    let _ = tx_clone.send(final_response).await;
                }
            }
        });
    }

    drop(tx); 
    let _ = writer_task.await; 

    info!("🛑 [Main] Ssense Native Daemon terminated.");
    Ok(())
}

fn route_request(
    request: DaemonRequest, 
    state: Arc<AppState>, 
    _tx: mpsc::Sender<DaemonResponse>, 
    rt: tokio::runtime::Handle
) -> DaemonResponse {
    
    let req_id = request.request_id().to_string();

    match request {
        DaemonRequest::HealthCheck(_) => {
            // is_loaded() only reflects whether a model has ALREADY been memory-mapped
            // by a prior audit/chat request (LocalEngine loads lazily on first use — see
            // the boot comment above). On a fresh daemon process that's legitimately
            // false even when the model files are fully downloaded and verified, which
            // wrongly reported "model not loaded" here while the popup (which checks
            // file presence via is_offline_ready) correctly said "ready." Report ready
            // if EITHER a model is already resident, OR the files are present and will
            // load on first use.
            let already_resident = state.inference_engine.is_loaded();
            let files_ready = rt.block_on(state.model_manager.is_offline_ready());
            let model_loaded = already_resident || files_ready;
            let metrics = state.inference_engine.get_metrics();
            let safe_tps = if metrics.avg_tokens_per_second.is_finite() { metrics.avg_tokens_per_second as u32 } else { 0 };

            DaemonResponse::HealthCheckResult {
                request_id: req_id,
                success: true,
                model_loaded,
                cache_size: state.cache.get_cache_size(),
                total_inferences: metrics.total_inferences,
                avg_tokens_per_second: safe_tps,
                has_gpu_acceleration: state.has_gpu_acceleration,
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
            if let Some(cached_report) = state.cache.get_full_report(&req.domain) {
                return DaemonResponse::AuditPolicyResult {
                    request_id: req_id, success: true, report: cached_report, cached: true,
                };
            }

            match state.inference_engine.audit_policy(&req.domain, &req.policy_text) {
                Ok(report) => {
                    let _ = state.cache.save_audit(&req.domain, &report);
                    DaemonResponse::AuditPolicyResult {
                        request_id: req_id, success: true, report, cached: false,
                    }
                }
                Err(e) => DaemonResponse::Error {
                    request_id: req_id, success: false, error: format!("Offline inference failed: {}", e),
                }
            }
        }

        DaemonRequest::Chat(req) => {
            let context = state.cache.get_full_report(&req.domain);
            let audit_ref = context.as_ref();

            if audit_ref.is_none() {
                return DaemonResponse::ChatStreamChunk {
                    request_id: req_id, 
                    token: "Please run an Audit on this site before asking questions.".to_string(),
                    is_final: true,
                };
            }

            let mut rag_guard = rt.block_on(async { state.rag_engine.write().await });

            // SOTA FIX: Simulate streaming by executing the monolithic inference 
            // and chunking it back over IPC. (True llama-cpp-2 streaming requires callbacks).
            match state.inference_engine.chat_with_context(&req.domain, &req.user_prompt, audit_ref.unwrap(), rag_guard.as_mut()) {
                Ok(message) => {
                    // Send message natively to UI
                    DaemonResponse::ChatStreamChunk {
                        request_id: req_id,
                        token: message,
                        is_final: true,
                    }
                }
                Err(e) => DaemonResponse::Error { 
                    request_id: req_id, success: false, error: format!("Offline chat failed: {}", e) 
                }
            }
        }
        
        DaemonRequest::DownloadModels(_) => {
            // Unreachable: Handled async upstream
            DaemonResponse::Error { request_id: req_id, success: false, error: "Routing error".to_string() }
        }

        DaemonRequest::PauseDownload(_) => {
            // Unreachable: Handled async upstream
            DaemonResponse::Error { request_id: req_id, success: false, error: "Routing error".to_string() }
        }
    }
}