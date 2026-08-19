/**
 * SLM Server & Native Daemon Protocol Types
 * MUST stay in sync with:
 * 1. apps/slm-server/security.py (get_dpdp_schema / validate_and_repair_report)
 * 2. apps/native-daemon/src/messaging/protocol.rs
 * 3. libs/contracts/schemas/dpdp_schema.json
 *
 * This unified protocol dictates the JSON structure for both the Cloud API (SSE)
 * and the Local Edge Daemon (Native Messaging IPC).
 */

// ═══════════════════════════════════════════════════════════════
// REQUESTS (Extension -> SLM Server / Rust Daemon)
// ═══════════════════════════════════════════════════════════════

export type DaemonRequest =
  | AuditPolicyRequest
  | ChatRequest
  | GetTrustScoreRequest
  | HealthCheckRequest
  | DownloadModelsRequest;

export interface AuditPolicyRequest {
  type: "AUDIT_POLICY";
  requestId: string;
  domain: string;
  policyText: string;
}

export interface ChatRequest {
  type: "CHAT";
  requestId: string;
  domain: string;
  userPrompt: string;
}

export interface GetTrustScoreRequest {
  type: "GET_TRUST_SCORE";
  requestId: string;
  domain: string;
}

export interface HealthCheckRequest {
  type: "HEALTH_CHECK";
  requestId: string;
}

export interface DownloadModelsRequest {
  type: "DOWNLOAD_MODELS";
  requestId: string;
}

// ═══════════════════════════════════════════════════════════════
// RESPONSES (SLM Server / Rust Daemon -> Extension)
// ═══════════════════════════════════════════════════════════════

export type DaemonResponse =
  | AuditPolicyResponse
  | ChatStreamChunk
  | GetTrustScoreResponse
  | HealthCheckResponse
  | StatusResponse
  | DownloadProgressResponse
  | ErrorResponse;

export interface AuditPolicyResponse {
  type: "AUDIT_POLICY_RESULT";
  requestId: string;
  success: true;
  report: DpdpAuditReport;
  cached: boolean;
}

export interface ChatStreamChunk {
  type: "CHAT_STREAM_CHUNK";
  requestId: string;
  token: string;
  is_final: boolean;
}

export interface GetTrustScoreResponse {
  type: "TRUST_SCORE_RESULT";
  requestId: string;
  success: true;
  score: number | null;
}

export interface HealthCheckResponse {
  type: "HEALTH_CHECK_RESULT";
  requestId: string;
  success: boolean;
  modelLoaded: boolean;
  cacheSize: number;
  totalInferences: number;
  avgTokensPerSecond: number;
}

export interface StatusResponse {
  type: "STATUS";
  requestId: string | null;
  status: string;
  message: string;
}

export interface DownloadProgressResponse {
  type: "DOWNLOAD_PROGRESS";
  requestId: string | null;
  file: string;
  pct: number;
  mbPerSec: number;
}

export interface ErrorResponse {
  type: "ERROR";
  requestId: string;
  success: false;
  error: string;
}

// ═══════════════════════════════════════════════════════════════
// SHARED TYPES (Strictly matching dpdp_schema.json & Rust Structs)
// ═══════════════════════════════════════════════════════════════

export interface DpdpAuditReport {
  global_legal_reasoning: string;
  violations: Violation[];
  dpdp_trust_score: number;
  subtlety_score: number;
}

export interface Violation {
  step_1_active_claim_analysis: string;
  step_2_statute_match: string;
  omission_check: boolean;
  step_3_semantic_justification: string;
  statute_reference: string;
  violation_type: ViolationType;
  evidence_quote: string;
  network_action: NetworkAction;
  offending_entities: string[];
}

// 🚀 SOTA FIX: Fully synchronized with all 27 DPDP Violation targets from grammar.rs
export type ViolationType =
  | "PURPOSE_LIMITATION_VIOLATION"
  | "CONSENT_NOT_FREE_OR_SPECIFIC"
  | "LEGITIMATE_USES_ABUSE"
  | "NOTICE_INADEQUATE"
  | "DATA_RETENTION_LIMIT_EXCEEDED"
  | "ERASURE_NOTICE_PERIOD_VIOLATION"
  | "LOG_RETENTION_MANDATE_VIOLATION"
  | "CHILD_CONSENT_VIOLATION"
  | "SECURITY_SAFEGUARDS_MISSING"
  | "GRIEVANCE_REDRESSAL_INADEQUATE"
  | "BREACH_NOTIFICATION_FAILURE"
  | "PROCESSOR_ACCOUNTABILITY_VIOLATION"
  | "SDF_OBLIGATIONS_MISSING"
  | "SDF_DATA_LOCALIZATION_VIOLATION"
  | "CROSS_BORDER_TRANSFER_VIOLATION"
  | "CONSENT_MANAGER_OBSTRUCTION"
  | "LANGUAGE_ACCESSIBILITY"
  | "ALGORITHMIC_PROFILING_SDF"
  | "RIGHTS_IMPLEMENTATION_VIOLATION"
  | "DATA_ACCURACY_COMPLETENESS_VIOLATION"
  | "BOARD_COMPLIANCE_VIOLATION"
  | "PENALTY_AVOIDANCE"
  | "APPEAL_PROCESS_VIOLATION"
  | "SCOPE_APPLICATION_EVASION"
  | "ILLEGAL_EXEMPTION_CLAIM"
  | "CONSENT_MECHANICS_VIOLATION"
  | "UNKNOWN_VIOLATION";

// 🚀 SOTA FIX: Added UNKNOWN_ACTION to align with Rust enums
export type NetworkAction =
  | "BLOCK_THIRD_PARTY"
  | "STRIP_TELEMETRY_HEADER"
  | "SPOOF_HARDWARE_API"
  | "INJECT_GPC_SIGNAL"
  | "WARN_USER_ONLY"
  | "UNKNOWN_ACTION";