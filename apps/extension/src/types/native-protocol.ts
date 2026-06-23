/**
 * Native Messaging Protocol Types
 * MUST stay in 1000% sync with:
 * 1. apps/native-daemon/src/messaging/protocol.rs
 * 2. libs/contracts/schemas/dpdp_schema.json
 */

// ═══════════════════════════════════════════════════════════════
// REQUESTS (Extension -> Daemon)
// ═══════════════════════════════════════════════════════════════

export type DaemonRequest =
  | AuditPolicyRequest
  | ChatRequest
  | GetTrustScoreRequest
  | HealthCheckRequest;

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
  // 🚀 FIX: Removed siteAuditContext. The Rust daemon fetches this 
  // from the SQLite cache internally. Sending it over IPC is redundant.
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

// ═══════════════════════════════════════════════════════════════
// RESPONSES (Daemon -> Extension)
// ═══════════════════════════════════════════════════════════════

export type DaemonResponse =
  | AuditPolicyResponse
  | ChatResponse
  | GetTrustScoreResponse
  | HealthCheckResponse
  | ErrorResponse;

export interface AuditPolicyResponse {
  type: "AUDIT_POLICY_RESULT";
  requestId: string;
  success: true;
  report: DpdpAuditReport;
  cached: boolean;
}

export interface ChatResponse {
  type: "CHAT_RESULT";
  requestId: string;
  success: true;
  message: string;
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
  success: true;
  modelLoaded: boolean;
  cacheSize: number;
  // 🚀 FIX: Added telemetry metrics to match the final Rust HealthCheckResult
  totalInferences: number;
  avgTokensPerSecond: number;
}

export interface ErrorResponse {
  type: "ERROR";
  requestId: string;
  success: false;
  error: string;
}

// ═══════════════════════════════════════════════════════════════
// SHARED TYPES (Strictly matching dpdp_schema.json)
// ═══════════════════════════════════════════════════════════════

export interface DpdpAuditReport {
  global_legal_reasoning: string;
  violations: Violation[];
  dpdp_trust_score: number;
}

// 🚀 FIX: Strictly type the 10 allowed violation types to match 
// the JSON schema enum and the Rust ViolationType enum.
export type ViolationType =
  | "PURPOSE_LIMITATION_VIOLATION"
  | "CONSENT_NOT_FREE_OR_SPECIFIC"
  | "NOTICE_INADEQUATE"
  | "DATA_RETENTION_LIMIT_EXCEEDED"
  | "CHILD_CONSENT_VIOLATION"
  | "SECURITY_SAFEGUARDS_MISSING"
  | "GRIEVANCE_REDRESSAL_INADEQUATE"
  | "BREACH_NOTIFICATION_FAILURE"
  | "SDF_OBLIGATIONS_MISSING"
  | "CROSS_BORDER_TRANSFER_VIOLATION";

export type NetworkAction =
  | "BLOCK_THIRD_PARTY"
  | "STRIP_TELEMETRY_HEADER"
  | "SPOOF_HARDWARE_API"
  | "INJECT_GPC_SIGNAL"
  | "WARN_USER_ONLY";

export interface Violation {
  statute_reference: string;
  violation_type: ViolationType; // 🚀 FIX: Changed from generic 'string' to strict enum
  evidence_quote: string;
  network_action: NetworkAction;
  offending_entities: string[];
}