//! Shared utilities for the Ssense workspace.
//! Ensures consistent domain normalization, hashing, and data sanitization 
//! across the Rust Native Daemon, the SQLite cache, and auxiliary tooling.

use sha2::{Digest, Sha256};

/// Normalizes a domain by stripping "www." and lowercasing.
/// This ensures 'www.amazon.com' and 'amazon.com' share the exact same SQLite cache key.
pub fn normalize_domain(domain: &str) -> String {
    domain.to_lowercase().trim_start_matches("www.").to_string()
}

/// Generates a cryptographically stable SHA-256 hash of a normalized domain.
/// This is used as the PRIMARY KEY in the SQLite cache.
pub fn hash_domain(domain: &str) -> String {
    let normalized = normalize_domain(domain);
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Sanitizes and clamps the trust score to strictly enforce the 0-100 boundary
/// defined in the dpdp_schema.json, protecting against LLM hallucinations.
pub fn sanitize_trust_score(score: i32) -> i32 {
    if score < 0 {
        0
    } else if score > 100 {
        100
    } else {
        score
    }
}

/// Validates that a string is not empty, satisfying the `minLength: 1` 
/// constraint of the `evidence_quote` field in the strict JSON schema.
pub fn is_valid_evidence_quote(quote: &str) -> bool {
    !quote.trim().is_empty()
}

/// Validates that a violation type string exactly matches the 10 allowed enums
/// in the dpdp_schema.json.
pub fn is_valid_violation_type(v_type: &str) -> bool {
    matches!(
        v_type,
        "PURPOSE_LIMITATION_VIOLATION"
            | "CONSENT_NOT_FREE_OR_SPECIFIC"
            | "NOTICE_INADEQUATE"
            | "DATA_RETENTION_LIMIT_EXCEEDED"
            | "CHILD_CONSENT_VIOLATION"
            | "SECURITY_SAFEGUARDS_MISSING"
            | "GRIEVANCE_REDRESSAL_INADEQUATE"
            | "BREACH_NOTIFICATION_FAILURE"
            | "SDF_OBLIGATIONS_MISSING"
            | "CROSS_BORDER_TRANSFER_VIOLATION"
    )
}

/// Validates that a network action string exactly matches the 5 allowed enums
/// in the dpdp_schema.json.
pub fn is_valid_network_action(action: &str) -> bool {
    matches!(
        action,
        "BLOCK_THIRD_PARTY"
            | "STRIP_TELEMETRY_HEADER"
            | "SPOOF_HARDWARE_API"
            | "INJECT_GPC_SIGNAL"
            | "WARN_USER_ONLY"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_normalization() {
        assert_eq!(normalize_domain("WWW.Swiggy.com"), "swiggy.com");
        assert_eq!(normalize_domain("Amazon.com"), "amazon.com");
        assert_eq!(normalize_domain("example.com"), "example.com");
    }

    #[test]
    fn test_domain_hashing_consistency() {
        let hash1 = hash_domain("www.example.com");
        let hash2 = hash_domain("example.com");
        assert_eq!(hash1, hash2, "Hashes must match regardless of 'www.' prefix");
        
        // Ensure it's a valid 64-char hex string (SHA-256)
        assert_eq!(hash1.len(), 64);
    }

    #[test]
    fn test_trust_score_clamping() {
        assert_eq!(sanitize_trust_score(-50), 0);
        assert_eq!(sanitize_trust_score(150), 100);
        assert_eq!(sanitize_trust_score(85), 85);
    }

    #[test]
    fn test_evidence_quote_validation() {
        assert!(!is_valid_evidence_quote(""));
        assert!(!is_valid_evidence_quote("   "));
        assert!(is_valid_evidence_quote("We collect your data."));
    }

    #[test]
    fn test_schema_enum_validation() {
        assert!(is_valid_violation_type("NOTICE_INADEQUATE"));
        assert!(!is_valid_violation_type("UNKNOWN_VIOLATION")); // Strict schema enforcement
        
        assert!(is_valid_network_action("BLOCK_THIRD_PARTY"));
        assert!(!is_valid_network_action("UNKNOWN_ACTION")); // Strict schema enforcement
    }
}