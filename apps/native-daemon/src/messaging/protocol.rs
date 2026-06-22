use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════
// REQUESTS (Extension -> Daemon)
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DaemonRequest {
    AuditPolicy(AuditPolicyRequest),
    Chat(ChatRequest),
    GetTrustScore(GetTrustScoreRequest),
    HealthCheck(HealthCheckRequest),
}

impl DaemonRequest {
    pub fn request_id(&self) -> &str {
        match self {
            Self::AuditPolicy(r) => &r.request_id,
            Self::Chat(r) => &r.request_id,
            Self::GetTrustScore(r) => &r.request_id,
            Self::HealthCheck(r) => &r.request_id,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuditPolicyRequest {
    pub request_id: String,
    pub domain: String,
    pub policy_text: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatRequest {
    pub request_id: String,
    pub domain: String,
    pub user_prompt: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GetTrustScoreRequest {
    pub request_id: String,
    pub domain: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthCheckRequest {
    pub request_id: String,
}

// ═══════════════════════════════════════════════════════════════
// RESPONSES (Daemon -> Extension)
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DaemonResponse {
    AuditPolicyResult {
        #[serde(rename = "requestId")] request_id: String,
        success: bool,
        // CRITICAL FIX: The skip_serializing_if attribute is completely removed.
        // A successful audit will ALWAYS contain this report object.
        report: DpdpAuditReport,
        cached: bool,
    },
    ChatResult {
        #[serde(rename = "requestId")] request_id: String,
        success: bool,
        message: String,
    },
    TrustScoreResult {
        #[serde(rename = "requestId")] request_id: String,
        success: bool,
        score: Option<i32>,
    },
    HealthCheckResult {
        #[serde(rename = "requestId")] request_id: String,
        success: bool,
        #[serde(rename = "modelLoaded")] model_loaded: bool,
        #[serde(rename = "cacheSize")] cache_size: usize,
    },
    Error {
        #[serde(rename = "requestId")] request_id: String,
        success: bool,
        error: String,
    },
}

// ═══════════════════════════════════════════════════════════════
// SHARED TYPES (The Legal Payload)
// CRITICAL: Defaults to snake_case to perfectly mirror the ML schema
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DpdpAuditReport {
    pub global_legal_reasoning: String,
    pub violations: Vec<Violation>,
    pub dpdp_trust_score: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Violation {
    pub statute_reference: String,
    pub violation_type: String,
    pub evidence_quote: String,
    pub network_action: String,
    pub offending_entities: Vec<String>,
}