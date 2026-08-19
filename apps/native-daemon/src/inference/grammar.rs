use std::fmt;

// ─────────────────────────────────────────────────────────────────
// SOTA SENSE: Forced Structural Chain-of-Thought (CoT) Grammar
// ─────────────────────────────────────────────────────────────────
// This GBNF strictly orders keys to prevent state-machine explosion in llama.cpp.
// It explicitly forces the LLM to output Step 1 -> 2 -> 3 reasoning BEFORE 
// the final violation_type, rescuing reasoning accuracy for INT4 models.

pub const DPDP_AUDIT_GRAMMAR: &str = r#"
root ::= object

object ::= 
  "{" ws 
  "\"global_legal_reasoning\"" ws ":" ws string ws "," ws
  "\"violations\"" ws ":" ws violation-array ws "," ws
  "\"dpdp_trust_score\"" ws ":" ws trust-score ws "," ws
  "\"subtlety_score\"" ws ":" ws trust-score ws 
  "}"

violation-array ::= "[" ws (violation ws ("," ws violation ws)*)? "]"

violation ::= 
  "{" ws
  "\"step_1_active_claim_analysis\"" ws ":" ws string ws "," ws
  "\"step_2_statute_match\"" ws ":" ws string ws "," ws
  "\"omission_check\"" ws ":" ws boolean ws "," ws
  "\"step_3_semantic_justification\"" ws ":" ws string ws "," ws
  "\"statute_reference\"" ws ":" ws string ws "," ws
  "\"violation_type\"" ws ":" ws violation-type ws "," ws
  "\"evidence_quote\"" ws ":" ws string ws "," ws
  "\"network_action\"" ws ":" ws network-action ws "," ws
  "\"offending_entities\"" ws ":" ws string-array ws
  "}"

violation-type ::= 
  "\"PURPOSE_LIMITATION_VIOLATION\"" |
  "\"CONSENT_NOT_FREE_OR_SPECIFIC\"" |
  "\"LEGITIMATE_USES_ABUSE\"" |
  "\"NOTICE_INADEQUATE\"" |
  "\"DATA_RETENTION_LIMIT_EXCEEDED\"" |
  "\"ERASURE_NOTICE_PERIOD_VIOLATION\"" |
  "\"LOG_RETENTION_MANDATE_VIOLATION\"" |
  "\"CHILD_CONSENT_VIOLATION\"" |
  "\"SECURITY_SAFEGUARDS_MISSING\"" |
  "\"GRIEVANCE_REDRESSAL_INADEQUATE\"" |
  "\"BREACH_NOTIFICATION_FAILURE\"" |
  "\"PROCESSOR_ACCOUNTABILITY_VIOLATION\"" |
  "\"SDF_OBLIGATIONS_MISSING\"" |
  "\"SDF_DATA_LOCALIZATION_VIOLATION\"" |
  "\"CROSS_BORDER_TRANSFER_VIOLATION\"" |
  "\"CONSENT_MANAGER_OBSTRUCTION\"" |
  "\"LANGUAGE_ACCESSIBILITY\"" |
  "\"ALGORITHMIC_PROFILING_SDF\"" |
  "\"RIGHTS_IMPLEMENTATION_VIOLATION\"" |
  "\"DATA_ACCURACY_COMPLETENESS_VIOLATION\"" |
  "\"BOARD_COMPLIANCE_VIOLATION\"" |
  "\"PENALTY_AVOIDANCE\"" |
  "\"APPEAL_PROCESS_VIOLATION\"" |
  "\"SCOPE_APPLICATION_EVASION\"" |
  "\"ILLEGAL_EXEMPTION_CLAIM\"" |
  "\"CONSENT_MECHANICS_VIOLATION\"" |
  "\"UNKNOWN_VIOLATION\""

network-action ::= 
  "\"BLOCK_THIRD_PARTY\"" |
  "\"STRIP_TELEMETRY_HEADER\"" |
  "\"SPOOF_HARDWARE_API\"" |
  "\"INJECT_GPC_SIGNAL\"" |
  "\"WARN_USER_ONLY\"" |
  "\"UNKNOWN_ACTION\""

string-array ::= "[" ws (string ws ("," ws string ws)*)? "]"

string ::= "\"" ( [^"\\\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] ) )* "\""

boolean ::= "true" | "false"

trust-score ::= "0" | [1-9] [0-9]? | "100"

ws ::= [ \t\n]*
"#;

// ─────────────────────────────────────────────────────────────────
// SOTA SENSE: Grammar Drift Prevention Validator
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarValidationError {
    EmptyGrammar,
    MissingRule(String),
    MissingEnumVariant(String),
}

impl fmt::Display for GrammarValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGrammar => write!(f, "Grammar string is empty"),
            Self::MissingRule(rule) => write!(f, "Grammar missing required structural rule: {}", rule),
            Self::MissingEnumVariant(variant) => write!(f, "Grammar drift detected! Missing enum variant: {}", variant),
        }
    }
}

impl std::error::Error for GrammarValidationError {}

pub fn validate_grammar(grammar: &str) -> Result<(), GrammarValidationError> {
    if grammar.trim().is_empty() {
        return Err(GrammarValidationError::EmptyGrammar);
    }
    
    // 1. Structural Validation
    let required_rules = [
        "root ::=",
        "object ::=",
        "violation-array ::=",
        "violation ::=",
        "violation-type ::=",
        "network-action ::=",
        "string-array ::=",
        "string ::=",
        "boolean ::=",
        "trust-score ::=",
    ];

    for rule in required_rules {
        if !grammar.contains(rule) {
            return Err(GrammarValidationError::MissingRule(rule.to_string()));
        }
    }

    // 2. Enum Synchronization Check 
    // (Ensures the GBNF matches the Rust structs in messaging/protocol.rs)
    let required_network_actions = [
        "BLOCK_THIRD_PARTY",
        "STRIP_TELEMETRY_HEADER",
        "SPOOF_HARDWARE_API",
        "INJECT_GPC_SIGNAL",
        "WARN_USER_ONLY",
        "UNKNOWN_ACTION",
    ];

    for action in required_network_actions {
        if !grammar.contains(&format!("\"\\\"{}\\\"\"", action)) {
            return Err(GrammarValidationError::MissingEnumVariant(action.to_string()));
        }
    }
    
    Ok(())
}