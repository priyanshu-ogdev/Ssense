use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::grammar::LlamaGrammar;
use llama_cpp_2::token::LlamaToken;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{error, info, warn};

use crate::messaging::protocol::DpdpAuditReport;
use super::grammar::{DPDP_AUDIT_GRAMMAR, validate_grammar};
use crate::rag_engine::RagEngine;

const MAX_CONTEXT_TOKENS: u32 = 8192;
const MAX_POLICY_CHARS: usize = 16000; 
const MAX_GENERATE_TOKENS: u32 = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Auditor,
    Chatbot,
    None,
}

pub struct LocalEngine {
    audit_model_path: PathBuf,
    chat_model_path: PathBuf,
    backend: Arc<LlamaBackend>,
    model: Arc<Mutex<Option<LlamaModel>>>,
    current_model_type: Arc<Mutex<ModelType>>,
    metrics: Arc<Mutex<InferenceMetrics>>,
    optimal_threads: i32,
    dynamic_gpu_layers: i32,
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
    pub fn new(audit_model_path: &Path, chat_model_path: &Path, optimal_threads: i32) -> Result<Self> {
        validate_grammar(DPDP_AUDIT_GRAMMAR)
            .map_err(|e| anyhow::anyhow!("Grammar validation failed: {}", e))?;
        info!("✅ [LocalEngine] GBNF grammar validated successfully");

        let backend = LlamaBackend::init().context("Failed to initialize llama.cpp backend")?;

        let dynamic_gpu_layers = if cfg!(feature = "cublas") || cfg!(feature = "metal") {
            info!("⚡ [LocalEngine] Hardware Acceleration detected. Enabling GPU offloading.");
            33 // Full offload for 7B models
        } else {
            info!("💻 [LocalEngine] Running in pure CPU mode.");
            0
        };

        Ok(Self {
            audit_model_path: audit_model_path.to_path_buf(),
            chat_model_path: chat_model_path.to_path_buf(),
            backend: Arc::new(backend),
            model: Arc::new(Mutex::new(None)),
            current_model_type: Arc::new(Mutex::new(ModelType::None)),
            metrics: Arc::new(Mutex::new(InferenceMetrics::default())),
            optimal_threads,
            dynamic_gpu_layers,
        })
    }

    pub fn is_loaded(&self) -> bool {
        self.current_model_type.lock().map(|guard| *guard != ModelType::None).unwrap_or(false)
    }

    pub fn switch_context(&self, required_model: ModelType) -> Result<()> {
        let mut type_guard = self.current_model_type.lock().map_err(|_| anyhow::anyhow!("Type mutex poisoned"))?;
        
        if *type_guard != required_model {
            info!("🔄 [VRAM Airlock] Unloading current context...");
            
            {
                let mut model_guard = self.model.lock().map_err(|_| anyhow::anyhow!("Model mutex poisoned"))?;
                *model_guard = None; 
            }
            
            let path = match required_model {
                ModelType::Auditor => &self.audit_model_path,
                ModelType::Chatbot => &self.chat_model_path,
                ModelType::None => return Ok(()),
            };
            
            if !path.exists() {
                return Err(anyhow::anyhow!("Model file missing at {:?}", path));
            }

            let mut model_params = LlamaModelParams::default();
            model_params.with_n_gpu_layers(self.dynamic_gpu_layers);
            model_params.with_use_mmap(true);
            model_params.with_use_mlock(false);

            let model = LlamaModel::load_from_file(&self.backend, path, &model_params)
                .context("Failed to map GGUF to memory.")?;

            let mut model_guard = self.model.lock().unwrap();
            *model_guard = Some(model);
            *type_guard = required_model;

            info!("✅ [VRAM Airlock] Model {:?} mapped to OS successfully.", required_model);
        }
        Ok(())
    }

    pub fn audit_policy(&self, domain: &str, policy_text: &str) -> Result<DpdpAuditReport> {
        let start_time = Instant::now();
        if policy_text.trim().is_empty() { return Err(anyhow::anyhow!("Policy text empty")); }

        self.switch_context(ModelType::Auditor)?;

        let truncated_policy = Self::truncate_to_token_limit(policy_text, MAX_POLICY_CHARS);
        let prompt = self.build_audit_prompt(domain, truncated_policy);

        let grammar = LlamaGrammar::from_str(DPDP_AUDIT_GRAMMAR).context("Failed to parse grammar")?;
        
        // Audit requires low temp to prevent JSON structural drift
        let output = self.run_inference(prompt, 0.1, 1.1, Some(grammar))?;
        
        let report = self.parse_audit_response(&output)?;

        let inference_time_ms = start_time.elapsed().as_millis() as u64;
        self.update_metrics(output.len() as u64 / 4, inference_time_ms);

        Ok(report)
    }

    pub fn chat_with_context(
        &self,
        domain: &str,
        user_prompt: &str,
        audit_context: &DpdpAuditReport,
        rag_engine: Option<&RagEngine>,
    ) -> Result<String> {
        let start_time = Instant::now();

        self.switch_context(ModelType::Chatbot)?;

        let mut retrieved_context = String::new();
        if let Some(rag) = rag_engine {
            if let Ok(hits) = rag.search(user_prompt, false) {
                for hit in hits {
                    retrieved_context.push_str(&format!("<document>\n  <metadata>{}</metadata>\n  <text>{}</text>\n</document>\n", hit.metadata, hit.text));
                }
            }
        }

        let prompt = self.build_chat_prompt(domain, user_prompt, audit_context, &retrieved_context);

        // Chat requires higher temp (0.4) for natural flow, and higher repetition penalty (1.15) to prevent loops
        let output = self.run_inference(prompt, 0.4, 1.15, None)?;
        
        let inference_time_ms = start_time.elapsed().as_millis() as u64;
        self.update_metrics(output.len() as u64 / 4, inference_time_ms);

        Ok(output.trim().to_string())
    }

    // ─────────────────────────────────────────────────────────────────
    // SOTA SENSE: Hallucination-Protected Inference Loop
    // ─────────────────────────────────────────────────────────────────
    fn run_inference(&self, prompt: String, temp: f32, rep_pen: f32, grammar: Option<LlamaGrammar>) -> Result<String> {
        let model_guard = self.model.lock().map_err(|_| anyhow::anyhow!("Mutex poisoned"))?;
        let model = model_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Model not mapped"))?;

        let mut ctx_params = LlamaContextParams::default();
        ctx_params.with_n_ctx(MAX_CONTEXT_TOKENS.try_into().unwrap());
        ctx_params.with_n_threads(self.optimal_threads as u32);
        
        let mut ctx = model.create_context(&self.backend, ctx_params)
            .context("Failed to create context")?;

        let tokens = model.str_to_token(&prompt, AddBos::Always)?;
        let mut batch = LlamaBatch::new(MAX_CONTEXT_TOKENS as usize, 1);
        
        for (i, &token) in tokens.iter().enumerate() {
            batch.add(token, i as i32, &[0], i == tokens.len() - 1);
        }

        ctx.decode(&mut batch).context("Prefill decode failed")?;

        let mut output_text = String::new();
        let mut current_pos = tokens.len() as i32;
        let mut history: Vec<LlamaToken> = Vec::new();

        for _ in 0..MAX_GENERATE_TOKENS {
            let mut candidates = ctx.candidates_ith(batch.n_tokens() - 1);
            
            // 1. Apply Repetition Penalty to prevent Hallucination Loops
            if !history.is_empty() {
                ctx.sample_repetition_penalties(&mut candidates, &history, history.len(), rep_pen, 0.0, 0.0);
            }

            // 2. Apply Grammar (JSON Enforcement)
            if let Some(ref g) = grammar {
                ctx.sample_grammar(&mut candidates, g);
            }

            // 3. Apply Temperature Scaling
            ctx.sample_temp(&mut candidates, temp);

            // 4. Sample Token (Top-P = 0.9)
            ctx.sample_top_p(&mut candidates, 0.9, 1);
            let id = ctx.sample_token(&mut candidates);
            
            if let Some(ref g) = grammar {
                ctx.grammar_accept_token(g, id);
            }

            if id == model.token_eos() || id == model.token_eot() {
                break;
            }

            history.push(id);
            let text = model.token_to_str(id).unwrap_or_default();
            output_text.push_str(&text);

            if output_text.ends_with("<|im_end|>") {
                output_text = output_text.trim_end_matches("<|im_end|>").to_string();
                break;
            }

            batch.clear();
            batch.add(id, current_pos, &[0], true);
            ctx.decode(&mut batch).context("Token decode failed")?;
            current_pos += 1;
        }

        Ok(output_text)
    }

    fn truncate_to_token_limit<'a>(text: &'a str, max_chars: usize) -> &'a str {
        if text.len() <= max_chars { return text; }
        let mut end = max_chars;
        while end > 0 && !text.is_char_boundary(end) { end -= 1; }
        if end == 0 { return ""; }
        &text[..end]
    }

    // ─────────────────────────────────────────────────────────────────
    // SOTA SENSE: Natural Language Attention Structuring
    // ─────────────────────────────────────────────────────────────────
    fn build_audit_prompt(&self, domain: &str, policy_text: &str) -> String {
        format!(
            "<|im_start|>system\n\
You are a strict DPDP Regulatory Auditor. Identify violations of Indian data protection law.
You must output ONLY valid JSON.\n<|im_end|>\n\
<|im_start|>user\n\
Audit domain: {}\n\n[PRIVACY POLICY TEXT]\n{}\n<|im_end|>\n\
<|im_start|>assistant\n", domain, policy_text
        )
    }

    fn build_chat_prompt(&self, domain: &str, user_prompt: &str, audit_context: &DpdpAuditReport, rag_context: &str) -> String {
        // SOTA FIX: Translate raw JSON into Natural Language bullet points so the LLM's attention mechanism doesn't misfire.
        let mut translated_audit = format!("Trust Score: {}/100.\n", audit_context.dpdp_trust_score);
        if audit_context.violations.is_empty() {
            translated_audit.push_str("Status: Compliant. No critical violations found.\n");
        } else {
            translated_audit.push_str("Identified DPDP Violations:\n");
            for (i, v) in audit_context.violations.iter().enumerate() {
                translated_audit.push_str(&format!("{}. {} (Reference: {})\n", i+1, v.violation_type, v.statute_reference));
            }
        }

        format!(
            "<|im_start|>system\n\
You are the Ssense DPDP Co-Pilot. Base your answers strictly on the retrieved context below.

[AUDIT REPORT SUMMARY FOR {}]
{}

[RETRIEVED DPDP ACT STATUTES]
{}
<|im_end|>\n\
<|im_start|>user\n\
Question: {}\n<|im_end|>\n\
<|im_start|>assistant\n", domain, translated_audit, rag_context, user_prompt
        )
    }

    fn parse_audit_response(&self, output: &str) -> Result<DpdpAuditReport> {
        let start = output.find('{');
        let end = output.rfind('}');
        let json_str = match (start, end) {
            (Some(s), Some(e)) if s < e => &output[s..=e],
            _ => return Err(anyhow::anyhow!("No valid JSON object found in LLM output.")),
        };
        let mut report: DpdpAuditReport = serde_json::from_str(json_str)?;
        report.dpdp_trust_score = report.dpdp_trust_score.clamp(0, 100);
        Ok(report)
    }

    fn update_metrics(&self, estimated_tokens: u64, inference_time_ms: u64) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.total_inferences += 1;
            metrics.total_inference_time_ms += inference_time_ms;
            metrics.last_inference_time_ms = inference_time_ms;
            metrics.total_tokens_generated += estimated_tokens;
            
            let avg_time_secs = metrics.total_inference_time_ms as f64 / 1000.0;
            if avg_time_secs > 0.0 {
                metrics.avg_tokens_per_second = metrics.total_tokens_generated as f64 / avg_time_secs;
            }
        }
    }
}