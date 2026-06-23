// apps/native-daemon/src/messaging/protocol.rs

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

// 🚀 SOTA FIX: The Fallback Envelope
// If full deserialization fails, we use this to rescue the requestId
// so we can send an error back to Chrome and unlock the UI.
#[derive(Debug, Deserialize)]
pub struct RawEnvelope {
    #[serde(rename = "requestId")]
    pub request_id: Option<String>,
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
        // 🚀 SOTA FIX: Removed skip_serializing_if. 
        // Serde will now natively serialize None as `null`, preventing V8 `undefined` crashes.
        score: Option<i32>,
    },
    HealthCheckResult {
        #[serde(rename = "requestId")] request_id: String,
        success: bool,
        #[serde(rename = "modelLoaded")] model_loaded: bool,
        #[serde(rename = "cacheSize")] cache_size: usize,
        #[serde(rename = "totalInferences")] total_inferences: u64,
        #[serde(rename = "avgTokensPerSecond")] avg_tokens_per_second: u32,
    },
    Error {
        #[serde(rename = "requestId")] request_id: String,
        success: bool,
        error: String,
    },
}

// ═══════════════════════════════════════════════════════════════
// SHARED TYPES (Strictly Typed & Schema Mirrored)
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
    pub violation_type: ViolationType,
    pub evidence_quote: String,
    pub network_action: NetworkAction,
    pub offending_entities: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ViolationType {
    PurposeLimitationViolation,
    ConsentNotFreeOrSpecific,
    NoticeInadequate,
    DataRetentionLimitExceeded,
    ChildConsentViolation,
    SecuritySafeguardsMissing,
    GrievanceRedressalInadequate,
    BreachNotificationFailure,
    SdfObligationsMissing,
    CrossBorderTransferViolation,
    #[serde(other)]
    UnknownViolation,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NetworkAction {
    BlockThirdParty,
    StripTelemetryHeader,
    SpoofHardwareApi,
    InjectGpcSignal,
    WarnUserOnly,
    #[serde(other)]
    UnknownAction,
}