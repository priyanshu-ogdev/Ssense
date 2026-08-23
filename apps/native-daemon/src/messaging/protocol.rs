use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════
// REQUESTS (Chrome Extension -> Rust Daemon)
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DaemonRequest {
    AuditPolicy(AuditPolicyRequest),
    Chat(ChatRequest),
    GetTrustScore(GetTrustScoreRequest),
    HealthCheck(HealthCheckRequest),
    DownloadModels(DownloadModelsRequest), // SOTA FIX: Explicit UI control over network usage
    PauseDownload(PauseDownloadRequest), // Lets the user pause an in-flight download; partial files are kept for resume
}

impl DaemonRequest {
    pub fn request_id(&self) -> &str {
        match self {
            Self::AuditPolicy(r) => &r.request_id,
            Self::Chat(r) => &r.request_id,
            Self::GetTrustScore(r) => &r.request_id,
            Self::HealthCheck(r) => &r.request_id,
            Self::DownloadModels(r) => &r.request_id,
            Self::PauseDownload(r) => &r.request_id,
        }
    }
}

// Fallback envelope to rescue request_ids if JSON parsing fails structurally
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadModelsRequest {
    pub request_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PauseDownloadRequest {
    pub request_id: String,
}

// ═══════════════════════════════════════════════════════════════
// RESPONSES (Rust Daemon -> Chrome Extension)
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
    // SOTA FIX: Added real-time SSE-style streaming for the Native Messaging pipe
    ChatStreamChunk {
        #[serde(rename = "requestId")] request_id: String,
        token: String,
        is_final: bool,
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
        #[serde(rename = "totalInferences")] total_inferences: u64,
        #[serde(rename = "avgTokensPerSecond")] avg_tokens_per_second: u32,
        #[serde(rename = "hasGpuAcceleration")] has_gpu_acceleration: bool,
    },
    Status {
        #[serde(rename = "requestId")] request_id: Option<String>,
        status: String,
        message: String,
    },
    Error {
        #[serde(rename = "requestId")] request_id: String,
        success: bool,
        error: String,
    },
    DownloadProgress {
        #[serde(rename = "requestId")] request_id: Option<String>,
        file: String,
        pct: f64,
        #[serde(rename = "mbPerSec")] mb_per_sec: f64,
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
    pub subtlety_score: i32, 
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Violation {
    pub step_1_active_claim_analysis: String,
    pub step_2_statute_match: String,
    pub omission_check: bool,
    pub step_3_semantic_justification: String,
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
    LegitimateUsesAbuse,
    NoticeInadequate,
    DataRetentionLimitExceeded,
    ErasureNoticePeriodViolation,
    LogRetentionMandateViolation,
    ChildConsentViolation,
    SecuritySafeguardsMissing,
    GrievanceRedressalInadequate,
    BreachNotificationFailure,
    ProcessorAccountabilityViolation,
    SdfObligationsMissing,
    SdfDataLocalizationViolation,
    CrossBorderTransferViolation,
    ConsentManagerObstruction,
    LanguageAccessibility,
    AlgorithmicProfilingSdf,
    RightsImplementationViolation,
    DataAccuracyCompletenessViolation,
    BoardComplianceViolation,
    PenaltyAvoidance,
    AppealProcessViolation,
    ScopeApplicationEvasion,
    IllegalExemptionClaim,
    ConsentMechanicsViolation,
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