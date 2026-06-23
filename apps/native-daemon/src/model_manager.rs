use anyhow::{bail, Context, Result};
use reqwest::Client;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::fs::{self, File};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{error, info, warn};
use std::time::Duration;

const MODEL_FILENAME: &str = "qwen2.5-7b-instruct-q4_k_m.gguf";
const MODEL_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf";
const EXPECTED_SHA256: &str = "REPLACE_WITH_ACTUAL_HASH"; 
const DOWNLOAD_TIMEOUT_SECS: u64 = 3600;
const MAX_RETRIES: u32 = 3;

pub struct ModelManager {
    models_dir: PathBuf,
    client: Client,
    download_lock: tokio::sync::Mutex<()>, 
}

impl ModelManager {
    pub fn new(data_dir: &Path) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(DOWNLOAD_TIMEOUT_SECS))
            .user_agent("Ssense-Daemon/1.0")
            .build()
            .expect("Failed to build HTTP client");

        Self {
            models_dir: data_dir.join("models"),
            client,
            download_lock: tokio::sync::Mutex::new(()),
        }
    }

    pub async fn ensure_model_available(&self) -> Result<PathBuf> {
        // 🚀 SOTA FIX: Fully async directory check
        if fs::metadata(&self.models_dir).await.is_err() {
            fs::create_dir_all(&self.models_dir).await?;
        }

        let target_path = self.models_dir.join(MODEL_FILENAME);
        
        if self.is_model_available_at(&target_path).await {
            return Ok(target_path);
        }

        let _guard = self.download_lock.lock().await;
        
        if self.is_model_available_at(&target_path).await {
            return Ok(target_path);
        }

        let temp_path = self.models_dir.join(format!("{}.part", MODEL_FILENAME));
        
        for attempt in 1..=MAX_RETRIES {
            match self.download_model_with_resume(&temp_path).await {
                Ok(_) => {
                    match self.verify_checksum(&temp_path).await {
                        Ok(_) => {
                            fs::rename(&temp_path, &target_path).await.context("Failed to move verified model")?;
                            info!("✅ Model downloaded, verified, and installed successfully.");
                            return Ok(target_path);
                        }
                        Err(e) => {
                            error!("Checksum failed on attempt {}: {}", attempt, e);
                            let _ = fs::remove_file(&temp_path).await;
                            if attempt == MAX_RETRIES { bail!("Checksum verification failed after {} attempts", MAX_RETRIES); }
                        }
                    }
                }
                Err(e) => {
                    error!("Download attempt {} failed: {}", attempt, e);
                    if attempt == MAX_RETRIES { bail!("Download failed: {}", e); }
                }
            }
            tokio::time::sleep(Duration::from_secs(2u64.pow(attempt - 1))).await;
        }
        
        bail!("Model download failed after all retry attempts");
    }

    async fn download_model_with_resume(&self, temp_path: &Path) -> Result<()> {
        let mut downloaded: u64 = 0;
        
        // 🚀 SOTA FIX: Fully async file existence check
        let mut file = if fs::metadata(temp_path).await.is_ok() {
            let metadata = fs::metadata(temp_path).await?;
            downloaded = metadata.len();
            info!("Resuming download from {} MB", downloaded / 1024 / 1024);
            tokio::fs::OpenOptions::new().write(true).append(true).open(temp_path).await?
        } else {
            File::create(temp_path).await?
        };

        let mut request = self.client.get(MODEL_URL);
        if downloaded > 0 {
            request = request.header("Range", format!("bytes={}-", downloaded));
        }

        let response = request.send().await.context("Failed to initiate download")?;

        if response.status() == reqwest::StatusCode::OK && downloaded > 0 {
            warn!("Server ignored Range header. Restarting download from 0.");
            drop(file); 
            file = File::create(temp_path).await?; 
            downloaded = 0;
        } else if !response.status().is_success() && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            bail!("HTTP error: {}", response.status());
        }

        let total_size = response.content_length()
            .map(|len| len + downloaded)
            .unwrap_or(0); // Fallback if server doesn't send Content-Length

        let mut stream = response.bytes_stream();
        let mut last_log_percent = 0;

        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error reading download stream")?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            if total_size > 0 {
                let percent = (downloaded as f64 / total_size as f64 * 100.0) as u64;
                if percent >= last_log_percent + 10 {
                    info!("Download: {}% ({} MB / {} MB)", percent, downloaded / 1024 / 1024, total_size / 1024 / 1024);
                    last_log_percent = percent;
                }
            }
        }

        file.flush().await?;
        Ok(())
    }

    async fn verify_checksum(&self, file_path: &Path) -> Result<()> {
        info!("Verifying model checksum...");
        let mut file = File::open(file_path).await?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let n = file.read(&mut buffer).await?;
            if n == 0 { break; }
            hasher.update(&buffer[..n]);
        }

        let hash_hex = format!("{:x}", hasher.finalize());
        if hash_hex != EXPECTED_SHA256 {
            bail!("Checksum mismatch! Expected: {}, Actual: {}", EXPECTED_SHA256, hash_hex);
        }
        info!("Checksum verified.");
        Ok(())
    }

    pub fn get_model_path(&self) -> PathBuf {
        self.models_dir.join(MODEL_FILENAME)
    }

    // 🚀 SOTA FIX: Fully async existence check
    async fn is_model_available_at(&self, path: &Path) -> bool {
        fs::metadata(path).await.is_ok()
    }
}