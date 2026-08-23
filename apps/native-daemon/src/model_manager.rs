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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────
// Ensure this matches your protocol definitions in messaging/protocol.rs
// ─────────────────────────────────────────────────────────────────
use crate::messaging::protocol::DaemonResponse; 

// Max allowed gap between successful reads before a connection is considered dead.
// Resets after every chunk received — NOT a total download deadline (see client
// construction below for why that distinction matters for multi-GB downloads).
const STALL_TIMEOUT_SECS: u64 = 600;
const MAX_RETRIES: u32 = 3;
const BASE_URL: &str = "https://huggingface.co/PRiyanshu0-1/DPDP-SSense/resolve/main";
const HUB_REPO_ID: &str = "PRiyanshu0-1/DPDP-SSense";

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
        url_path: "models/audit-model-final-gguf_gguf/audit-model-final.Q4_K_M.gguf",
    },
    Artifact {
        name: "Conversational Co-Pilot (INT4)",
        filename: "chatbot-model.Q4_K_M.gguf",
        url_path: "models/chatbot-model-final-gguf_gguf/chatbot-model-final.Q4_K_M.gguf",
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
    // Cooperative pause flag checked inside the streaming loop of every in-flight
    // artifact download. Setting it doesn't kill the connection abruptly; each
    // download finishes writing its current chunk, flushes the `.part` file to
    // disk, and returns cleanly so the exact same Range-resume path used for a
    // crashed/closed browser also resumes a user-initiated pause.
    pause_requested: Arc<AtomicBool>,
}

/// What a single artifact's download attempt ended in. `Paused` is not an error —
/// it's a clean, user-requested stop with the partial file intentionally preserved.
enum DownloadOutcome {
    Completed,
    Paused,
}

impl ModelManager {
    /// SOTA Fix: Maps to safe OS directories (e.g., C:\Users\Name\AppData\Local\Ssense)
    pub fn new() -> Result<Self> {
        let proj_dirs = ProjectDirs::from("com", "Ssense", "ssense-native-daemon")
            .context("Failed to determine OS local data directory")?;
        
        let data_dir = proj_dirs.data_dir().to_path_buf();
        let models_dir = data_dir.join("models");

        let client = Client::builder()
            // IMPORTANT: do NOT use .timeout() here. reqwest's .timeout() is a total
            // request deadline covering connect + the *entire* response body — for a
            // multi-gigabyte model download, any connection slower than
            // (file_size / STALL_TIMEOUT_SECS) gets forcibly killed mid-transfer even
            // while it's actively receiving bytes. That's what caused downloads to appear
            // "stuck": the request silently died and retried from scratch every ~10 minutes.
            //
            // connect_timeout only bounds the TCP+TLS handshake (fails fast if the host is
            // unreachable). read_timeout bounds the gap between individual reads and RESETS
            // after every successful chunk — exactly what we want to detect a genuinely dead
            // connection without punishing a slow-but-alive one.
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(STALL_TIMEOUT_SECS))
            .user_agent("Ssense-Daemon/3.0")
            .build()
            .expect("Failed to build HTTP client");

        Ok(Self {
            models_dir,
            client,
            download_lock: tokio::sync::Mutex::new(()),
            pause_requested: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Called from the "Pause" button in the extension UI. Cooperative — the
    /// currently-streaming chunk still finishes, but no further bytes are requested
    /// after that, and the `.part` file is left on disk exactly as when Chrome is
    /// closed mid-download, so the existing Range-resume logic picks it back up.
    pub fn request_pause(&self) {
        self.pause_requested.store(true, Ordering::SeqCst);
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
    /// Fetches every file's true size directly from Hugging Face's Hub API
    /// (`/api/models/{repo}`), NOT from HTTP headers on the actual download URLs.
    ///
    /// This is deliberate: per-file `HEAD`/`GET` requests to `resolve/main/...` hit
    /// whatever CDN/proxy is actually serving the bytes, and for repos on Hugging
    /// Face's newer Xet storage backend that proxy can stream the response via
    /// chunked transfer-encoding with NO `Content-Length` header at all — for either
    /// HEAD or GET. When that happens there is nothing to "fill in later," and the
    /// download percentage is stuck on indeterminate for the whole run no matter how
    /// the retry/fallback logic downstream is written.
    ///
    /// The Hub API sidesteps all of that: it's Hugging Face's own metadata database,
    /// returned as one plain JSON document, and it lists every file's real byte size
    /// (`siblings[].lfs.size` for LFS/Xet-tracked files, `siblings[].size` otherwise)
    /// regardless of how the CDN happens to be streaming that file's actual content
    /// right now. One request gets sizes for the whole repo up front, reliably.
    async fn fetch_repo_file_sizes(client: &Client) -> Result<std::collections::HashMap<String, u64>> {
        let api_url = format!("https://huggingface.co/api/models/{}", HUB_REPO_ID);
        let resp = client.get(&api_url).send().await.context("Failed to reach Hugging Face Hub API for file sizes")?;
        if !resp.status().is_success() {
            bail!("Hugging Face Hub API returned {} for {}", resp.status(), api_url);
        }
        let json: serde_json::Value = resp.json().await.context("Hub API response was not valid JSON")?;
        let siblings = json.get("siblings").and_then(|s| s.as_array()).context("Hub API response had no 'siblings' array")?;

        let mut sizes = std::collections::HashMap::new();
        for sibling in siblings {
            let Some(filename) = sibling.get("rfilename").and_then(|v| v.as_str()) else { continue };
            // LFS/Xet-tracked files carry their real size under lfs.size (the `size`
            // field at the top level for these is the size of the LFS *pointer* text,
            // a few hundred bytes, not the actual model weights).
            let size = sibling
                .get("lfs")
                .and_then(|lfs| lfs.get("size"))
                .and_then(|v| v.as_u64())
                .or_else(|| sibling.get("size").and_then(|v| v.as_u64()));
            if let Some(size) = size {
                sizes.insert(filename.to_string(), size);
            }
        }
        Ok(sizes)
    }

    pub async fn ensure_all_available(&self, tx: Option<mpsc::Sender<DaemonResponse>>, req_id: Option<String>) -> Result<()> {
        if fs::metadata(&self.models_dir).await.is_err() {
            fs::create_dir_all(&self.models_dir).await.context("Failed to create models directory")?;
        }

        // Prevent multiple simultaneous clicks from spawning duplicate downloads
        let _guard = self.download_lock.lock().await;

        // A fresh call to ensure_all_available (initial click, or "Resume") always
        // starts from a clean pause state — otherwise a pause from a previous run
        // would immediately abort this one before it reads a single byte.
        self.pause_requested.store(false, Ordering::SeqCst);

        // ── Aggregate progress across ALL pending artifacts ──
        // Previously each artifact reported its OWN 0-100% independently, so the
        // popup would show the bar hit 100%, then reset to 0% and climb to 100%
        // again for the next file — looking like the download was looping forever
        // even though it was actually making forward progress through 4 files.
        // We instead sum the known sizes of every artifact that still needs
        // downloading up front, and report one running percent across that total.
        let mut pending: Vec<(&'static Artifact, PathBuf, PathBuf, String)> = vec![];
        for artifact in ARTIFACTS {
            let target_path = self.get_artifact_path(artifact.filename);
            if fs::metadata(&target_path).await.is_ok() {
                info!("✅ Artifact {} is already downloaded. Skipping.", artifact.name);
                continue;
            }
            let temp_path = self.models_dir.join(format!("{}.part", artifact.filename));
            let url = format!("{}/{}", BASE_URL, artifact.url_path);
            pending.push((artifact, target_path, temp_path, url));
        }

        let mut already_downloaded: u64 = 0;
        for (_, _, temp_path, _) in &pending {
            if let Ok(meta) = fs::metadata(temp_path).await {
                already_downloaded += meta.len();
            }
        }

        // Authoritative sizes from Hugging Face's Hub API — see fetch_repo_file_sizes'
        // doc comment for why this replaces per-file HEAD requests entirely. Any
        // artifact this call can't find a size for (API request failed outright, or
        // this particular file is missing from the response for some reason) falls
        // back to being filled in from its own GET response's Content-Length once its
        // download actually starts, same safety net as before.
        let mut known_total: u64 = 0;
        let mut unknown_size: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        match Self::fetch_repo_file_sizes(&self.client).await {
            Ok(sizes) => {
                for (artifact, _, _, _) in &pending {
                    match sizes.get(artifact.url_path) {
                        Some(size) => known_total += size,
                        None => {
                            warn!("Hub API had no size listed for '{}' ({}) — will fall back to its GET response.", artifact.name, artifact.url_path);
                            unknown_size.insert(artifact.name);
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Could not reach Hugging Face Hub API for file sizes ({}) — falling back to per-file GET responses for the progress total.", e);
                for (artifact, _, _, _) in &pending {
                    unknown_size.insert(artifact.name);
                }
            }
        }
        // batch_total is shared mutable state now (not a plain local), since it can grow
        // mid-batch as GET responses fill in sizes the Hub API lookup couldn't provide.
        let batch_total = Arc::new(AtomicU64::new(known_total));
        let shared_downloaded = Arc::new(AtomicU64::new(already_downloaded));

        // ── Sequential downloads, one artifact at a time ──
        // Previously all pending artifacts downloaded concurrently (each as its own
        // tokio::spawn task), which splits total available bandwidth across however
        // many files are in flight at once — a 20 Mbps connection downloading 4 files
        // at the same time gets ~5 Mbps each, not 20 Mbps for whichever one matters.
        // Going one file at a time gives each artifact the full connection, and keeps
        // the aggregate percent bar moving in a single, predictable direction instead
        // of jumping around as different concurrent tasks report at different rates.
        let mut failures: Vec<String> = vec![];
        let mut was_paused = false;

        'artifacts: for (artifact, target_path, temp_path, url) in pending {
            let name = artifact.name;
            // Only ever count this artifact's size into batch_total once, even across
            // retries — recomputing `size_already_known` from the set on every attempt
            // would double (or triple) count it if attempt 1 learned the size from GET
            // headers, then failed partway through the body and retried.
            let mut size_already_known = !unknown_size.contains(name);
            for attempt in 1..=MAX_RETRIES {
                let result = Self::download_file_with_resume(&self.client, &url, &temp_path, name, tx.clone(), req_id.clone(), shared_downloaded.clone(), batch_total.clone(), size_already_known, &self.pause_requested).await;
                // Whatever happened, any size this attempt could have learned from GET
                // response headers has already been folded into batch_total by now — a
                // retry must never re-add it, or the denominator inflates with each retry.
                size_already_known = true;
                match result {
                    Ok(DownloadOutcome::Paused) => {
                        info!("⏸ Download paused for '{}' — {} bytes retained on disk for resume.", name, fs::metadata(&temp_path).await.map(|m| m.len()).unwrap_or(0));
                        was_paused = true;
                        // Pausing stops the whole batch here — the remaining not-yet-started
                        // artifacts stay untouched on disk (nothing to resume for them since
                        // they never began), and Resume re-enters this same loop from the top,
                        // skipping whatever's already finished via the `pending` scan above.
                        break 'artifacts;
                    }
                    Ok(DownloadOutcome::Completed) => {
                        if let Err(e) = Self::verify_integrity(&self.client, &url, &temp_path, name, artifact.filename).await {
                            error!("Integrity check for '{}' could not be completed: {}. Discarding partial file and retrying.", name, e);
                            let _ = fs::remove_file(&temp_path).await;
                            if attempt == MAX_RETRIES {
                                failures.push(format!("Could not verify integrity of {} after {} attempts: {}", name, MAX_RETRIES, e));
                                continue 'artifacts;
                            }
                            tokio::time::sleep(Duration::from_secs(2u64.pow(attempt - 1))).await;
                            continue;
                        }
                        fs::rename(&temp_path, &target_path).await.context("Failed to finalize verified model file")?;
                        info!("✅ Artifact '{}' downloaded and integrity-verified.", name);
                        continue 'artifacts;
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        if let Some(rest) = msg.strip_prefix("NOT_FOUND:") {
                            // Non-retryable: the URL is wrong/missing. Don't burn retries on it.
                            error!("'{}' is unavailable at its source and will be skipped: {}", name, rest);
                            failures.push(format!("NOT_FOUND: {}", rest));
                            continue 'artifacts;
                        }
                        error!("Download attempt {} for '{}' failed: {}", attempt, name, e);
                        if attempt == MAX_RETRIES {
                            failures.push(format!("Download failed for {} after {} attempts: {}", name, MAX_RETRIES, e));
                            continue 'artifacts;
                        }
                        tokio::time::sleep(Duration::from_secs(2u64.pow(attempt - 1))).await;
                    }
                }
            }
        }

        if let Some(channel) = &tx {
            if was_paused && failures.is_empty() {
                let _ = channel.send(DaemonResponse::Status {
                    status: "paused".to_string(),
                    message: "Download paused. Your progress is saved — resume anytime.".to_string(),
                    request_id: req_id.clone(),
                }).await;
            } else if failures.is_empty() {
                let _ = channel.send(DaemonResponse::Status {
                    status: "success".to_string(),
                    message: "Offline models ready.".to_string(),
                    request_id: req_id.clone(),
                }).await;
            } else {
                warn!("Offline model download finished with {} failure(s): {:?}", failures.len(), failures);
                let _ = channel.send(DaemonResponse::Status {
                    status: "partial".to_string(),
                    message: format!(
                        "Some offline models are unavailable and were skipped ({} of {} failed). Cloud/Fast mode is unaffected. Details: {}",
                        failures.len(),
                        ARTIFACTS.len(),
                        failures.join("; ")
                    ),
                    request_id: req_id.clone(),
                }).await;
            }
        }

        if failures.is_empty() {
            Ok(())
        } else {
            // Return Ok rather than propagating a hard error: offline models are optional,
            // Cloud/Fast mode must keep working, and the UI already got a "partial" status
            // above with the details. A caller that wants to treat this as fatal can check
            // is_offline_ready() afterward.
            Ok(())
        }
    }

    /// Verifies the downloaded artifact against the true SHA-256 published in the file's Git
    /// LFS pointer (fetched in `fetch_authoritative_sha256` below), before the file is ever
    /// renamed into the models directory and made eligible for loading into llama.cpp.
    ///
    /// Fetches the authoritative SHA-256 for an LFS-tracked file straight from its Git LFS
    /// pointer (`{repo}/raw/{branch}/{path}`), which Hugging Face publishes in plain text as
    /// `oid sha256:<hex>` for every LFS-tracked path, regardless of which storage backend
    /// (classic LFS or the newer Xet) actually serves the bytes.
    ///
    /// We deliberately do NOT use the `ETag`/`x-linked-etag` HTTP headers for this: those
    /// only equal the file's SHA-256 for repos still on classic Git LFS. Repos migrated to
    /// Xet report a *different* content-addressing hash via those same headers — it's also
    /// 64 hex characters, so it looks exactly like a SHA-256, but it isn't one. Comparing our
    /// real SHA-256 against it fails every single time, deterministically, on an otherwise
    /// perfectly good download — which is exactly what was happening here.
    async fn fetch_authoritative_sha256(client: &Client, resolve_url: &str, name: &'static str) -> Result<String> {
        let raw_url = resolve_url.replacen("/resolve/", "/raw/", 1);
        let resp = client.get(&raw_url).send().await.context("Integrity check: failed to fetch LFS pointer")?;
        let text = resp.text().await.context("Integrity check: failed to read LFS pointer body")?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("oid sha256:") {
                let hash = rest.trim().to_lowercase();
                if hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Ok(hash);
                }
            }
        }
        bail!("'{}' has no LFS pointer with a sha256 oid at {} — not an LFS/Xet-tracked file, or the pointer format changed", name, raw_url);
    }

    async fn verify_integrity(client: &Client, url: &str, file_path: &Path, name: &'static str, final_filename: &str) -> Result<()> {
        let expected_sha256 = match Self::fetch_authoritative_sha256(client, url, name).await {
            Ok(hash) => hash,
            Err(e) => {
                warn!(
                    "No verifiable LFS SHA-256 available from origin for '{}' ({}) — cannot cryptographically verify this download.",
                    name, e
                );
                if final_filename.ends_with(".json") {
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
        shared_downloaded: Arc<AtomicU64>,
        batch_total: Arc<AtomicU64>,
        size_already_known: bool,
        pause_signal: &AtomicBool,
    ) -> Result<DownloadOutcome> {
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
            let status = response.status();
            if status.is_client_error() {
                // 404/403/etc mean the file isn't at this URL at all — retrying the same
                // URL 3 times will never succeed, so tag this as non-retryable.
                bail!("NOT_FOUND:{} for '{}': the model source URL returned {} — the file does not exist at the configured location.", status, name, status);
            }
            bail!("HTTP network error: {}", status);
        }

        // If HEAD couldn't tell us this file's size up front (common for Xet-backed
        // Hugging Face files, which don't always expose Content-Length on HEAD the way
        // they do on GET), fill it in now from this GET response instead of leaving the
        // batch total — and therefore the whole percentage — stuck as "unknown" for the
        // rest of the download. A 206 Partial Content response's Content-Length is only
        // the REMAINING bytes for a resumed download, so add back what we'd already
        // downloaded to get this artifact's true total size.
        if !size_already_known {
            if let Some(len) = response.content_length() {
                let full_size = if response.status() == reqwest::StatusCode::PARTIAL_CONTENT { downloaded + len } else { len };
                batch_total.fetch_add(full_size, Ordering::Relaxed);
            }
        }

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
            let total_downloaded = shared_downloaded.fetch_add(chunk_len, Ordering::Relaxed) + chunk_len;

            let now = std::time::Instant::now();
            let elapsed = now.duration_since(last_update_time).as_secs_f64();

            // Throttle IPC messaging to ~2Hz to prevent Chrome Extension freezing.
            // Emit progress even when the origin didn't send Content-Length (batch_total < 0),
            // so the popup shows live byte/speed movement instead of appearing frozen —
            // percent is only meaningful once we know the total, but bytes/speed are not.
            if elapsed >= 0.5 {
                let mb_per_sec = (bytes_since_last_update as f64 / 1024.0 / 1024.0) / elapsed;
                // Percent is computed against the WHOLE batch of pending artifacts, not just
                // this one file — this is what stops the progress bar from appearing to
                // "loop" back to 0% every time one artifact finishes and the next starts.
                // -1.0 signals "indeterminate" to the frontend (unknown total size) rather
                // than falsely reporting 0%. batch_total can still be growing (see the
                // size_already_known top-up above), so it's read fresh here each tick.
                let known_batch_total = batch_total.load(Ordering::Relaxed);
                // Self-correct: Xet-backed storage's Content-Length isn't fully authoritative
                // even when present (see fetch_repo_file_sizes' doc comment) — it can
                // under-report a resumed download's true remaining size. If real bytes ever
                // exceed what we currently think the total is, grow the total to match
                // instead of letting percent run past 100% (confirmed happening in practice:
                // 104% climbing to 129% before this fix).
                let known_batch_total = if known_batch_total > 0 && total_downloaded > known_batch_total {
                    batch_total.store(total_downloaded, Ordering::Relaxed);
                    total_downloaded
                } else {
                    known_batch_total
                };
                let percent = if known_batch_total > 0 { (total_downloaded as f64 / known_batch_total as f64) * 100.0 } else { -1.0 };

                if let Some(channel) = &tx {
                    let _ = channel.send(DaemonResponse::DownloadProgress {
                        request_id: req_id.clone(),
                        file: name.to_string(),
                        pct: percent,
                        mb_per_sec,
                    }).await;
                }

                if known_batch_total > 0 {
                    if percent as u64 >= last_log_percent + 5 {
                        info!("Downloading '{}': {:.1}% overall ({} MB / {} MB total)", name, percent, total_downloaded / 1024 / 1024, known_batch_total / 1024 / 1024);
                        last_log_percent = percent as u64;
                    }
                } else {
                    info!("Downloading '{}': {} MB (total size unknown)", name, downloaded / 1024 / 1024);
                }

                last_update_time = now;
                bytes_since_last_update = 0;
            }

            // Cooperative pause: finish this chunk (already written above), flush what
            // we have, and stop — the `.part` file on disk is a valid resume point.
            if pause_signal.load(Ordering::SeqCst) {
                file.flush().await?;
                return Ok(DownloadOutcome::Paused);
            }
        }

        file.flush().await?;
        Ok(DownloadOutcome::Completed)
    }
}