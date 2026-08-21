use anyhow::{bail, Context, Result};
use directories::ProjectDirs;
use reqwest::Client;
use std::path::{Path, PathBuf};
use tokio::fs::{self, File};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use std::time::Duration;
use futures_util::StreamExt;
use sha2::{Digest, Sha256};

// ─────────────────────────────────────────────────────────────────
// Ensure this matches your protocol definitions in messaging/protocol.rs
// ─────────────────────────────────────────────────────────────────
use crate::messaging::protocol::DaemonResponse; 

const DOWNLOAD_TIMEOUT_SECS: u64 = 600;
const MAX_RETRIES: u32 = 3;
const BASE_URL: &str = "https://huggingface.co/PRiyanshu0-1/DPDP-SSense/resolve/main";

#[derive(Debug)]
pub struct Artifact {
    pub name: &'static str,
    pub filename: &'static str,
    pub url_path: &'static str,
}

pub const ARTIFACTS: &[Artifact] = &[
    Artifact {
        name: "Forensic Audit Model (INT4)",
        filename: "audit-model.Q4_K_M.gguf",
        url_path: "audit-model-final-gguf/audit-model-final.Q4_K_M.gguf",
    },
    Artifact {
        name: "Conversational Co-Pilot (INT4)",
        filename: "chatbot-model.Q4_K_M.gguf",
        url_path: "chatbot-model-final-gguf/chatbot-model-final.Q4_K_M.gguf",
    },
    Artifact {
        name: "RAG Semantic Matrix",
        filename: "dpdp_embeddings.safetensors",
        url_path: "models/rag-index/dpdp_embeddings.safetensors",
    },
    Artifact {
        name: "RAG Lexical Index",
        filename: "dpdp_index.json",
        url_path: "models/rag-index/dpdp_index.json",
    },
];

pub struct ModelManager {
    models_dir: PathBuf,
    client: Client,
    download_lock: tokio::sync::Mutex<()>,
}

impl ModelManager {
    /// SOTA Fix: Maps to safe OS directories (e.g., C:\Users\Name\AppData\Local\Ssense)
    pub fn new() -> Result<Self> {
        let proj_dirs = ProjectDirs::from("com", "Ssense", "ssense-native-daemon")
            .context("Failed to determine OS local data directory")?;
        
        let data_dir = proj_dirs.data_dir().to_path_buf();
        let models_dir = data_dir.join("models");

        let client = Client::builder()
            .timeout(Duration::from_secs(DOWNLOAD_TIMEOUT_SECS))
            .user_agent("Ssense-Daemon/3.0")
            .build()
            .expect("Failed to build HTTP client");

        Ok(Self {
            models_dir,
            client,
            download_lock: tokio::sync::Mutex::new(()),
        })
    }

    /// Check if all models are present to flip the UI toggle "Offline Mode: Ready"
    pub async fn is_offline_ready(&self) -> bool {
        for artifact in ARTIFACTS {
            let path = self.models_dir.join(artifact.filename);
            if fs::metadata(&path).await.is_err() {
                return false;
            }
        }
        true
    }

    pub fn get_artifact_path(&self, filename: &str) -> PathBuf {
        // Look in the OS models directory
        self.models_dir.join(filename)
    }

    /// Executed ONLY when the user explicitly clicks "Download Offline Models" in the extension UI
    pub async fn ensure_all_available(&self, tx: Option<mpsc::Sender<DaemonResponse>>, req_id: Option<String>) -> Result<()> {
        if fs::metadata(&self.models_dir).await.is_err() {
            fs::create_dir_all(&self.models_dir).await.context("Failed to create models directory")?;
        }

        // Prevent multiple simultaneous clicks from spawning duplicate downloads
        let _guard = self.download_lock.lock().await;

        let mut futures = vec![];

        for artifact in ARTIFACTS {
            let target_path = self.get_artifact_path(artifact.filename);
            
            if fs::metadata(&target_path).await.is_ok() {
                info!("✅ Artifact {} is already downloaded. Skipping.", artifact.name);
                continue; 
            }

            let temp_path = self.models_dir.join(format!("{}.part", artifact.filename));
            let url = format!("{}/{}", BASE_URL, artifact.url_path);
            
            let client = self.client.clone();
            let tx_clone = tx.clone();
            let req_id_clone = req_id.clone();
            let target_path_clone = target_path.clone();
            let name = artifact.name;

            futures.push(tokio::spawn(async move {
                for attempt in 1..=MAX_RETRIES {
                    match Self::download_file_with_resume(&client, &url, &temp_path, name, tx_clone.clone(), req_id_clone.clone()).await {
                        Ok(_) => {
                            if let Err(e) = Self::verify_integrity(&client, &url, &temp_path, name).await {
                                error!("Integrity check for '{}' could not be completed: {}. Discarding partial file and retrying.", name, e);
                                let _ = fs::remove_file(&temp_path).await;
                                if attempt == MAX_RETRIES { bail!("Fatal: Could not verify integrity of {} after {} attempts: {}", name, MAX_RETRIES, e); }
                                tokio::time::sleep(Duration::from_secs(2u64.pow(attempt - 1))).await;
                                continue;
                            }
                            fs::rename(&temp_path, &target_path_clone).await.context("Failed to finalize verified model file")?;
                            info!("✅ Artifact '{}' downloaded and integrity-verified.", name);
                            return Ok(());
                        }
                        Err(e) => {
                            error!("Download attempt {} for '{}' failed: {}", attempt, name, e);
                            if attempt == MAX_RETRIES { bail!("Fatal: Download failed for {}: {}", name, e); }
                        }
                    }
                    tokio::time::sleep(Duration::from_secs(2u64.pow(attempt - 1))).await;
                }
                bail!("Exhausted retries for {}", name);
            }));
        }

        for f in futures {
            f.await??;
        }

        // Notify Chrome UI that all models are ready
        if let Some(channel) = &tx {
            let _ = channel.send(DaemonResponse::Status { 
                status: "success".to_string(), 
                message: "Offline models ready.".to_string(), 
                request_id: req_id 
            }).await;
        }

        Ok(())
    }

    /// Verifies the downloaded artifact against the SHA-256 the origin server reports.
    /// Hugging Face LFS-tracked files (which is how all our .gguf/.safetensors artifacts
    /// are hosted) return the file's SHA-256 as the `ETag` (via `x-linked-etag`, falling
    /// back to `etag`) on both HEAD and GET responses. We compare our locally computed
    /// hash of the fully-downloaded bytes against that value before the file is ever
    /// renamed into the models directory and made eligible for loading into llama.cpp.
    ///
    /// If the origin doesn't return a SHA-256-shaped ETag (e.g. a non-LFS asset, or a
    /// server behind a proxy that strips the header), we can't compare and fail *closed*
    /// on integrity for `.gguf`/`.safetensors` weights, but allow small sidecar files
    /// (like the JSON lexical index) through, since a corrupted JSON file fails to parse
    /// immediately at load time rather than silently loading corrupted model weights.
    async fn verify_integrity(client: &Client, url: &str, file_path: &Path, name: &'static str) -> Result<()> {
        let head = client.head(url).send().await.context("Integrity check: HEAD request failed")?;
        let raw_etag = head
            .headers()
            .get("x-linked-etag")
            .or_else(|| head.headers().get("etag"))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim_matches('"').to_string());

        let expected_sha256 = match &raw_etag {
            Some(tag) if tag.len() == 64 && tag.chars().all(|c| c.is_ascii_hexdigit()) => tag.to_lowercase(),
            _ => {
                warn!(
                    "No SHA-256 ETag available from origin for '{}' — cannot cryptographically verify this download.",
                    name
                );
                if file_path.extension().and_then(|e| e.to_str()) == Some("json") {
                    // Small, human-inspectable text asset: parse failure at load time is an
                    // acceptable secondary safety net for this one file type.
                    return Ok(());
                }
                bail!("Origin did not provide a verifiable checksum for model weights '{}'; refusing to trust an unverified binary.", name);
            }
        };

        let mut file = File::open(file_path).await.context("Integrity check: failed to reopen downloaded file")?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = file.read(&mut buf).await.context("Integrity check: read error")?;
            if n == 0 { break; }
            hasher.update(&buf[..n]);
        }
        let actual_sha256 = hex::encode(hasher.finalize());

        if actual_sha256 != expected_sha256 {
            bail!(
                "SHA-256 mismatch for '{}': expected {}, got {}. The downloaded file is corrupted or was tampered with.",
                name, expected_sha256, actual_sha256
            );
        }

        info!("🔒 SHA-256 verified for '{}': {}", name, actual_sha256);
        Ok(())
    }

    async fn download_file_with_resume(
        client: &Client,
        url: &str,
        temp_path: &Path,
        name: &'static str,
        tx: Option<mpsc::Sender<DaemonResponse>>,
        req_id: Option<String>,
    ) -> Result<()> {
        let mut downloaded: u64 = 0;
        
        let mut file = if fs::metadata(temp_path).await.is_ok() {
            let metadata = fs::metadata(temp_path).await?;
            downloaded = metadata.len();
            info!("Resuming download of '{}' from {} MB...", name, downloaded / 1024 / 1024);
            tokio::fs::OpenOptions::new().write(true).append(true).open(temp_path).await?
        } else {
            File::create(temp_path).await?
        };

        let mut request = client.get(url);
        if downloaded > 0 {
            request = request.header("Range", format!("bytes={}-", downloaded));
        }

        let response = request.send().await.context("Failed to initiate HTTP download")?;

        if response.status() == reqwest::StatusCode::OK && downloaded > 0 {
            warn!("Server ignored Range header. Restarting download of '{}' from 0.", name);
            drop(file); 
            file = File::create(temp_path).await?; 
            downloaded = 0;
        } else if !response.status().is_success() && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            bail!("HTTP network error: {}", response.status());
        }

        let total_size = response.content_length().map(|len| len + downloaded).unwrap_or(0);

        let mut stream = response.bytes_stream();
        let mut last_log_percent = 0;
        let mut last_update_time = std::time::Instant::now();
        let mut bytes_since_last_update = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error reading byte stream chunk")?;
            file.write_all(&chunk).await?;
            
            let chunk_len = chunk.len() as u64;
            downloaded += chunk_len;
            bytes_since_last_update += chunk_len;

            let now = std::time::Instant::now();
            let elapsed = now.duration_since(last_update_time).as_secs_f64();

            // Throttle IPC messaging to ~2Hz to prevent Chrome Extension freezing.
            // Emit progress even when the origin didn't send Content-Length (total_size == 0),
            // so the popup shows live byte/speed movement instead of appearing frozen —
            // percent is only meaningful once we know the total, but bytes/speed are not.
            if elapsed >= 0.5 {
                let mb_per_sec = (bytes_since_last_update as f64 / 1024.0 / 1024.0) / elapsed;
                // -1.0 signals "indeterminate" to the frontend (unknown total size) rather
                // than falsely reporting 0%.
                let percent = if total_size > 0 { (downloaded as f64 / total_size as f64) * 100.0 } else { -1.0 };

                if let Some(channel) = &tx {
                    let _ = channel.send(DaemonResponse::DownloadProgress {
                        request_id: req_id.clone(),
                        file: name.to_string(),
                        pct: percent,
                        mb_per_sec,
                    }).await;
                }

                if total_size > 0 {
                    if percent as u64 >= last_log_percent + 5 {
                        info!("Downloading '{}': {:.1}% ({} MB / {} MB)", name, percent, downloaded / 1024 / 1024, total_size / 1024 / 1024);
                        last_log_percent = percent as u64;
                    }
                } else {
                    info!("Downloading '{}': {} MB (total size unknown)", name, downloaded / 1024 / 1024);
                }

                last_update_time = now;
                bytes_since_last_update = 0;
            }
        }

        file.flush().await?;
        Ok(())
    }
}