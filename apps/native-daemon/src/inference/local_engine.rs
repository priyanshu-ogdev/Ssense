// apps/native-daemon/src/inference/local_engine.rs

use anyhow::{Context, Result};
use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{error, info, warn};

use crate::messaging::protocol::DpdpAuditReport;
use super::grammar::{DPDP_AUDIT_GRAMMAR, validate_grammar};

const MAX_CONTEXT_TOKENS: i32 = 8192;
const MAX_POLICY_CHARS: usize = 16000; 
const MAX_GENERATE_TOKENS: i32 = 2048;
const GPU_LAYERS: i32 = 9999;  

pub struct LocalEngine {
    model_path: PathBuf,
    model: Arc<Mutex<Option<LLama>>>,
    is_loaded: Arc<Mutex<bool>>,
    metrics: Arc<Mutex<InferenceMetrics>>,
    optimal_threads: i32, // 🚀 SOTA: Dynamic hardware handshake
}

#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    pub total_inferences: u64,
    pub total_tokens_generated: u64,
    pub total_inference_time_ms: u64,
    pub avg_tokens_per_second: f64,
    pub last_inference_time_ms: u64,
}

impl LocalEngine {
    pub fn new(model_path: &Path, optimal_threads: i32) -> Result<Self> {
        validate_grammar(DPDP_AUDIT_GRAMMAR)
            .map_err(|e| anyhow::anyhow!("Grammar validation failed: {}", e))?;
        
        info!("✅ GBNF grammar validated successfully");

        Ok(Self {
            model_path: model_path.to_path_buf(),
            model: Arc::new(Mutex::new(None)),
            is_loaded: Arc::new(Mutex::new(false)),
            metrics: Arc::new(Mutex::new(InferenceMetrics::default())),
            optimal_threads,
        })
    }

    // 🚀 SOTA FIX: Safe boolean extraction from MutexGuard
    pub fn is_loaded(&self) -> bool {
        self.is_loaded.lock().map(|guard| *guard).unwrap_or(false)
    }

    pub fn get_metrics(&self) -> InferenceMetrics {
        self.metrics.lock().map(|m| m.clone()).unwrap_or_default()
    }

    fn load_model(&self) -> Result<()> {
        info!("🧠 Loading model from {:?}", self.model_path);

        if !self.model_path.exists() {
            return Err(anyhow::anyhow!("Model file not found at {:?}", self.model_path));
        }

        let metadata = std::fs::metadata(&self.model_path)?;
        let size_gb = metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0);

        let mut model_options = ModelOptions::default();
        model_options.n_ctx = MAX_CONTEXT_TOKENS;
        model_options.n_gpu_layers = GPU_LAYERS;
        model_options.use_mmap = true;
        model_options.use_mlock = false;

        let model_path_str = self.model_path.to_string_lossy();
        
        info!("Loading model with {} GPU layers, {} context window", GPU_LAYERS, MAX_CONTEXT_TOKENS);

        let model = LLama::new(&model_path_str, &model_options)
            .context("Failed to load GGUF model. File may be corrupted.")?;

        let mut model_guard = self.model.lock().map_err(|_| anyhow::anyhow!("Model mutex poisoned during load"))?;
        *model_guard = Some(model);
        
        let mut loaded_guard = self.is_loaded.lock().map_err(|_| anyhow::anyhow!("Loaded mutex poisoned"))?;
        *loaded_guard = true;

        info!("✅ Model loaded successfully ({:.2}GB)", size_gb);
        Ok(())
    }

    fn truncate_to_token_limit<'a>(&self, text: &'a str, max_chars: usize) -> &'a str {
        if text.len() <= max_chars {
            return text;
        }

        let mut end = max_chars;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }

        if end == 0 && !text.is_char_boundary(max_chars) {
            warn!("Truncation limit falls inside a multi-byte character; returning empty slice.");
            return "";
        }

        &text[..end]
    }

    pub fn audit_policy(&self, domain: &str, policy_text: &str) -> Result<DpdpAuditReport> {
        let start_time = Instant::now();

        if policy_text.trim().is_empty() {
            return Err(anyhow::anyhow!("Policy text is empty"));
        }

        if !self.is_loaded() {
            self.load_model()?;
        }

        info!("🔍 Running inference for domain: {}", domain);
        let truncated_policy = self.truncate_to_token_limit(policy_text, MAX_POLICY_CHARS);
        let prompt = self.build_audit_prompt(domain, truncated_policy);

        let mut predict_options = PredictOptions::default();
        predict_options.temperature = 0.1;
        predict_options.top_p = 0.9;
        predict_options.top_k = 40;
        predict_options.n_predict = MAX_GENERATE_TOKENS;
        predict_options.threads = self.optimal_threads; // 🚀 SOTA: Dynamic thread scaling
        predict_options.grammar = Some(DPDP_AUDIT_GRAMMAR.to_string());

        let output = self.run_inference(prompt, predict_options)?;
        let report = self.parse_audit_response(&output)?;

        let inference_time_ms = start_time.elapsed().as_millis() as u64;
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.total_inferences += 1;
            metrics.total_inference_time_ms += inference_time_ms;
            metrics.last_inference_time_ms = inference_time_ms;
            
            let estimated_tokens = output.len() as u64 / 4;
            metrics.total_tokens_generated += estimated_tokens;
            let avg_time_secs = metrics.total_inference_time_ms as f64 / 1000.0;
            if avg_time_secs > 0.0 {
                metrics.avg_tokens_per_second = metrics.total_tokens_generated as f64 / avg_time_secs;
            }
        }

        info!("✅ Inference complete in {}ms", inference_time_ms);
        Ok(report)
    }

    fn run_inference(&self, prompt: String, options: PredictOptions) -> Result<String> {
        let model_guard = self.model.lock()
            .map_err(|_| anyhow::anyhow!("Model mutex poisoned during inference"))?;
            
        let model = model_guard.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded (internal error)"))?;

        let output = model.predict(prompt, options)
            .context("Model predict call failed")?;

        Ok(output)
    }

    pub fn chat_with_context(
        &self,
        domain: &str,
        user_prompt: &str,
        audit_context: &DpdpAuditReport,
    ) -> Result<String> {
        if !self.is_loaded() {
            self.load_model()?;
        }

        info!("💬 Running chat inference for domain: {}", domain);
        let prompt = self.build_chat_prompt(domain, user_prompt, audit_context);

        let mut predict_options = PredictOptions::default();
        predict_options.temperature = 0.7;
        predict_options.top_p = 0.9;
        predict_options.top_k = 40;
        predict_options.n_predict = 512;
        predict_options.threads = self.optimal_threads; // 🚀 SOTA: Dynamic thread scaling

        let output = self.run_inference(prompt, predict_options)?;
        Ok(output.trim().to_string())
    }

    fn build_audit_prompt(&self, domain: &str, policy_text: &str) -> String {
        format!(
            "<|im_start|>system\n\
You are a strict DPDP (Digital Personal Data Protection Act 2023, India) Regulatory Auditor. Your job is to analyze privacy policies and identify violations of Indian data protection law.

You must output ONLY valid JSON matching this exact schema:
{{
  \"global_legal_reasoning\": \"string - Your chain-of-thought analysis\",
  \"violations\": [
    {{
      \"statute_reference\": \"string - e.g., 'Section 8(7)'\",
      \"violation_type\": \"string - MUST BE ONE OF THE EXACT ENUM VALUES BELOW\",
      \"evidence_quote\": \"string - Exact quote from the policy\",
      \"network_action\": \"string - MUST BE ONE OF THE EXACT ENUM VALUES BELOW\",
      \"offending_entities\": [\"string\"]
    }}
  ],
  \"dpdp_trust_score\": number (0-100)
}}

Violation Types (CHOOSE EXACTLY ONE):
- PURPOSE_LIMITATION_VIOLATION
- CONSENT_NOT_FREE_OR_SPECIFIC
- NOTICE_INADEQUATE
- DATA_RETENTION_LIMIT_EXCEEDED
- CHILD_CONSENT_VIOLATION
- SECURITY_SAFEGUARDS_MISSING
- GRIEVANCE_REDRESSAL_INADEQUATE
- BREACH_NOTIFICATION_FAILURE
- SDF_OBLIGATIONS_MISSING
- CROSS_BORDER_TRANSFER_VIOLATION
- UNKNOWN_VIOLATION

Network Actions (CHOOSE EXACTLY ONE):
- BLOCK_THIRD_PARTY
- STRIP_TELEMETRY_HEADER
- SPOOF_HARDWARE_API
- INJECT_GPC_SIGNAL
- WARN_USER_ONLY
- UNKNOWN_ACTION

Be strict. If the policy is clear and compliant, give a high trust score (80-100). If there are violations, give a low score (0-50).
<|im_end|>\n\
<|im_start|>user\n\
Audit the following privacy policy for domain: {domain}\n\n\
{policy_text}\n\
<|im_end|>\n\
<|im_start|>assistant\n"
        )
    }

    fn build_chat_prompt(
        &self,
        domain: &str,
        user_prompt: &str,
        audit_context: &DpdpAuditReport,
    ) -> String {
        let context_json = serde_json::to_string(audit_context).unwrap_or_else(|_| "{}".to_string());

        format!(
            "<|im_start|>system\n\
You are the Ssense Co-Pilot, a helpful AI assistant that explains DPDP compliance issues to users.
Use this audit report context to answer questions: {context_json}
<|im_end|>\n\
<|im_start|>user\n\
Current domain: {domain}\n\
User Question: {user_prompt}\n\
<|im_end|>\n\
<|im_start|>assistant\n"
        )
    }

    fn parse_audit_response(&self, output: &str) -> Result<DpdpAuditReport> {
        let start = output.find('{');
        let end = output.rfind('}');

        let json_str = match (start, end) {
            (Some(s), Some(e)) if s < e => &output[s..=e],
            _ => {
                error!("No valid JSON object found in LLM output:\n{}", output);
                return Err(anyhow::anyhow!("LLM output did not contain a valid JSON object"));
            }
        };

        let mut report: DpdpAuditReport = serde_json::from_str(json_str)
            .context(format!("Failed to parse extracted JSON. Raw extracted string: {}", json_str))?;

        if report.dpdp_trust_score < 0 { report.dpdp_trust_score = 0; }
        if report.dpdp_trust_score > 100 { report.dpdp_trust_score = 100; }

        Ok(report)
    }
}