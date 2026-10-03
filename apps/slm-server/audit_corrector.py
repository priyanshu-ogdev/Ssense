#!/usr/bin/env python3
"""
audit_corrector.py – Deterministic Statutory Compliance Layer (SCL)

SOTA technique: Rule-based post-processing corrector that catches DPDP Act
violations that the LoRA model missed. This is NOT a replacement for the model —
it acts as a lossless enforcement layer that only ADDS violations that are
explicitly evidenced in the policy text. It cannot remove model-found violations.

Architecture:
  1. Model runs → produces JSON report (may miss violations due to attention limits)
  2. SCL runs → scans the raw policy text with deterministic regex patterns
  3. For each pattern match that maps to a DPDP violation NOT already in the report:
       - Inject the violation with the evidence quote
  4. Recompute trust score from the merged full violation set

Why this works without retraining:
  - DPDP Act violations map to specific textual patterns that are legally
    defined. "We do not provide a Grievance Officer" is unambiguously
    Section 13. "We track children" is unambiguously Section 9.
  - The model's job was to recognize these patterns — the SCL is a
    correctness backstop for attention failures on long contexts.
  - This approach is standard in hybrid NL+rule compliance pipelines
    (see: EU AI Act auditing tooling, legal NLP literature).
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# ─── DPDP Statutory Pattern Registry ──────────────────────────────────────────
# Each entry: (violation_type, statute_reference, regex, evidence_template, check_negation)
# Ordered by severity (most critical first).
# These patterns target EXPLICIT POSITIVE ADMISSIONS in policy text, NOT omissions.
# omission_check is always False here — we only flag active claims.

_NEGATION_PREFIX_RE = re.compile(
    r"(?:\b(?:do\s+not|don'?t|does\s+not|doesn'?t|did\s+not|didn'?t|never|neither|nor|no|not|cannot|can'?t|won'?t|will\s+not|prohibit(?:s|ed)?|strictly\s+prohibit(?:s|ed)?|disallow(?:s|ed)?|refrain(?:s|ed)?\s+from)\b\s*(?:\w+\s+){0,10}$)",
    re.I
)

_CONDITIONAL_SAFE_HARBOR_RE = re.compile(
    r"\b(?:if\s+we\s+(?:discover|learn|determine|find|become\s+aware)|in\s+the\s+event\s+that|we\s+(?:will\s+)?(?:promptly\s+)?(?:delete|erase|remove))\b",
    re.I
)

_STATUTORY_PATTERNS: List[Tuple[str, str, re.Pattern, str, str, bool]] = [
    # Section 9: Children's data / tracking without parental consent
    (
        "CHILD_CONSENT_VIOLATION",
        "Section 9(1)",
        re.compile(
            r"(?:"
            r"(?:knowingly|intentionally)?\s*(?:monitor|track|process|collect)(?:ing|s|ed)?(?:\s+and\s+\w+)?\s+(?:the\s+)?(?:browsing\s+)?(?:behaviou?r|activities|activity|data|personal\s+data)?(?:\s+of\s+)?(?:children|minors|users\s+under(?:\s+the\s+age\s+of)?\s+\d+)[^.]*?without\s+(?:verifiable\s+)?(?:parental|guardian)\s+consent"
            r"|(?:targeted\s+advertising|behavioural\s+monitoring|behaviour(?:al)?\s+tracking)\s+(?:directed\s+at|of|for)\s+(?:children|minors)"
            r")",
            re.I | re.S,
        ),
        "BLOCK_THIRD_PARTY",
        "Section 9 of the DPDP Act prohibits tracking or behavioural monitoring of children and requires verifiable parental consent before processing any personal data of a child.",
        True,
    ),

    # Section 8(3) / 8(6): Indefinite retention / no erasure right
    (
        "DATA_RETENTION_LIMIT_EXCEEDED",
        "Section 8(7)",
        re.compile(
            r"(?:"
            r"(?:retain|store)(?:s|ed|ing)?\s+(?:your\s+)?(?:information|data|personal\s+data|records|archives?)\s+(?:forever|indefinitely|permanently|for\s+an?\s+indefinite\s+period)"
            r"|(?:no|not|never|without)\s+(?:have\s+(?:the|any)\s+)?(?:any\s+)?(?:right|ability|option)\s+to\s+(?:request\s+)?(?:erasure|deletion|correction|removal)"
            r"|data\s+(?:is|are|will\s+be)\s+(?:retained|stored|kept)\s+(?:forever|indefinitely|permanently)"
            r"|(?:stored|retained)\s+permanently"
            r"|we\s+(?:do\s+not|don.t|cannot)\s+(?:delete|erase|remove)\s+(?:your\s+)?(?:data|information|personal\s+data|records|health\s+records)"
            r")",
            re.I | re.S,
        ),
        "WARN_USER_ONLY",
        "Section 8(7) of the DPDP Act mandates data erasure upon withdrawal of consent or when the specified purpose is served. Indefinite retention without an erasure right is a direct statutory violation.",
        True,
    ),

    # Section 16 / Rule 12: Cross-border transfer to non-notified jurisdictions
    (
        "CROSS_BORDER_TRANSFER_VIOLATION",
        "Section 16(1)",
        re.compile(
            r"(?:"
            r"transfer(?:s|red|ring)?\s+(?:your\s+)?(?:personal\s+data|information|identifiers|transaction\s+histor(?:y|ies))\s+(?:indefinitely\s+)?(?:to\s+)?(?:foreign|overseas|international|non-notified|third[-\s]party(?:\s+marketing)?\s+brokers?)"
            r"|transferred\s+(?:indefinitely|permanently|forever)\s+to\s+(?:foreign|overseas|international|non-notified)\s+(?:marketing\s+)?(?:brokers?|third[-\s]part(?:y|ies))"
            r"|transfer(?:s|red)?\s+(?:indefinitely|permanently|forever)\s+to\s+(?:foreign|overseas|international)"
            r")",
            re.I | re.S,
        ),
        "BLOCK_THIRD_PARTY",
        "Section 16 of the DPDP Act prohibits transfer of personal data to countries not notified by the Central Government. Unrestricted indefinite cross-border transfers to non-notified jurisdictions is a direct statutory breach.",
        True,
    ),

    # Section 5(1) + Rule 3: Notice inadequacy / no itemized notice
    (
        "NOTICE_INADEQUATE",
        "Section 5(1)",
        re.compile(
            r"(?:"
            r"(?:do\s+not|don.t)\s+(?:provide|give|offer|include)\s+(?:an?\s+)?(?:itemized|separate|individual|specific)(?:\s+(?:itemized|separate|individual|specific))?\s+(?:notice|consent|checkbox|option)s?"
            r"|(?:unspecified|unknown|vague)\s+(?:commercial|business)?\s+purposes?"
            r"|by\s+(?:continuing\s+to\s+browse|browsing|using\s+(?:this\s+)?(?:site|website|service)),?\s+you\s+(?:agree|consent)\s+(?:to\s+all)?"
            r")",
            re.I | re.S,
        ),
        "WARN_USER_ONLY",
        "Section 5(1) and Rule 3 of the DPDP Act require Data Fiduciaries to provide a clear, itemized notice enabling specific and informed consent. Implied consent through continued browsing is not compliant.",
        False,
    ),

    # Section 13: No Grievance Officer / DPO in India
    (
        "GRIEVANCE_REDRESSAL_INADEQUATE",
        "Section 13(1)",
        re.compile(
            r"(?:"
            r"(?:do\s+not\s+(?:have|maintain|appoint)|don.t\s+(?:have|maintain|appoint)|without)\s+(?:a\s+)?(?:data\s+protection\s+officer|dpo|grievance\s+officer|nodal\s+officer|grievance\s+redressal)"
            r"|complaints?\s+(?:will\s+be\s+)?(?:disregarded|ignored)"
            r"|(?:no|not)\s+(?:providing|having|maintaining|appointing)\s+(?:a\s+)?(?:grievance|complaint)\s+(?:mechanism|officer|redressal)"
            r")",
            re.I | re.S,
        ),
        "WARN_USER_ONLY",
        "Section 13 of the DPDP Act mandates a published Grievance Officer mechanism for Data Principals to exercise their rights. Explicitly refusing to address complaints is a direct statutory violation.",
        True,
    ),

    # Section 6: Consent not free / forced consent
    (
        "CONSENT_NOT_FREE_OR_SPECIFIC",
        "Section 6(1)",
        re.compile(
            r"(?:"
            r"collect(?:ing|s|ed)?\s+[^.]*?(?:biometric(?:\s+facial)?|facial\s+recognition)\s+(?:data|markers?|templates?)[^.]*?without\s+(?:requiring\s+)?(?:any\s+)?(?:explicit|specific|informed|separate)?\s*(?:opt[-\s]in|consent)"
            r"|(?:biometric(?:\s+facial)?|facial\s+recognition)\s+(?:data|markers?|templates?|scans?)\s+[^.]*?without\s+(?:consent|explicit|opt[-\s]in)"
            r")",
            re.I | re.S,
        ),
        "BLOCK_THIRD_PARTY",
        "Section 6(1) of the DPDP Act requires that consent be free, specific, informed, and unconditional. Collecting biometric data without explicit opt-in is a fundamental consent violation.",
        True,
    ),

    # Section 8(4): Security safeguards not mentioned
    (
        "SECURITY_SAFEGUARDS_MISSING",
        "Section 8(5)",
        re.compile(
            r"(?:"
            r"(?:do\s+not\s+maintain|without|no)\s+(?:any\s+)?(?:security|safeguards?|encryption|protection)\s+(?:measures?|procedures?|protocols?)"
            r"|data\s+breach(?:es)?\s+(?:will\s+not\s+be|are\s+not)\s+(?:notified|reported|disclosed)"
            r")",
            re.I | re.S,
        ),
        "WARN_USER_ONLY",
        "Section 8(5) of the DPDP Act requires reasonable security safeguards to prevent personal data breaches.",
        True,
    ),
]

# Severity deductions mirroring recombine_audit_reports()
_SEVERITY_DEDUCTIONS: Dict[str, int] = {
    "PURPOSE_LIMITATION_VIOLATION": 15,
    "CONSENT_NOT_FREE_OR_SPECIFIC": 15,
    "CONSENT_MECHANICS_VIOLATION": 15,
    "CROSS_BORDER_TRANSFER_VIOLATION": 15,
    "SDF_DATA_LOCALIZATION_VIOLATION": 15,
    "CHILD_CONSENT_VIOLATION": 15,
    "LEGITIMATE_USES_ABUSE": 12,
    "NOTICE_INADEQUATE": 10,
    "DATA_RETENTION_LIMIT_EXCEEDED": 10,
    "SECURITY_SAFEGUARDS_MISSING": 10,
    "GRIEVANCE_REDRESSAL_INADEQUATE": 10,
    "PROCESSOR_ACCOUNTABILITY_VIOLATION": 10,
}


def _extract_evidence_quote(text: str, match: re.Match, max_len: int = 180) -> str:
    """Extract a context-rich evidence quote centred around the regex match."""
    start = max(0, match.start() - 30)
    end = min(len(text), match.end() + 100)
    raw = text[start:end].strip()
    # Trim to complete words
    if len(raw) > max_len:
        raw = raw[:max_len].rsplit(" ", 1)[0] + "..."
    return raw


def _is_valid_violation_match(policy_text: str, match: re.Match, check_negation: bool) -> bool:
    """Verifies that the matched statutory pattern is an active violation and not negated or conditioned."""
    start_pos = match.start()
    clause_delims = [policy_text.rfind(c, 0, start_pos) for c in ('.', '\n', ';')]
    clause_start = max(0, max(clause_delims) + 1)
    clause_prefix = policy_text[clause_start:start_pos]

    sentence_end_delims = [policy_text.find(c, match.end()) for c in ('.', '\n', ';')]
    valid_ends = [e for e in sentence_end_delims if e != -1]
    sentence_end = min(valid_ends) if valid_ends else len(policy_text)
    full_sentence = policy_text[clause_start:sentence_end]

    if _CONDITIONAL_SAFE_HARBOR_RE.search(full_sentence):
        return False

    if check_negation and _NEGATION_PREFIX_RE.search(clause_prefix):
        return False

    return True


def apply_statutory_compliance_layer(
    report: Dict[str, Any],
    policy_text: str,
    domain: str = "",
) -> Dict[str, Any]:
    """
    Post-processes the model's audit report by scanning the raw policy text
    for explicit statutory violations. Any violation evidenced in the text
    that the model did not flag is injected into the report.

    This function is LOSSLESS:
      - It never removes a violation the model found
      - It only adds violations with direct textual evidence
      - The trust score is recomputed from the merged set

    Args:
        report: The model's validated audit report dict
        policy_text: The raw (sanitized) policy text that was audited
        domain: The site domain for logging

    Returns:
        Updated report dict with merged violations and recomputed score
    """
    if not policy_text or not isinstance(report, dict):
        return report

    existing_violations = report.get("violations", [])
    if not isinstance(existing_violations, list):
        existing_violations = []

    # Build set of already-flagged violation types to avoid duplicates
    existing_types: set = {
        v.get("violation_type", "") for v in existing_violations
        if isinstance(v, dict)
    }

    injected: List[Dict[str, Any]] = []
    for vtype, statute, pattern, network_action, reasoning, check_negation in _STATUTORY_PATTERNS:
        if vtype in existing_types:
            continue  # Already caught by model
        for match in pattern.finditer(policy_text):
            if not _is_valid_violation_match(policy_text, match, check_negation):
                continue
            evidence = _extract_evidence_quote(policy_text, match)
            injected_violation = {
                "step_1_active_claim_analysis": f"Regex pattern matched explicit admission in policy text.",
                "step_2_statute_match": f"Matched against {statute} requirements via Statutory Compliance Layer.",
                "omission_check": False,
                "step_3_semantic_justification": reasoning,
                "statute_reference": statute,
                "violation_type": vtype,
                "evidence_quote": evidence,
                "network_action": network_action,
                "offending_entities": [],
                "_scl_injected": True,  # Audit trail: marks SCL-injected violations
            }
            injected.append(injected_violation)
            print(
                f"[SCL] {domain}: Injecting missed violation {vtype} ({statute}) "
                f"-- evidence: \"{evidence[:60]}...\""
            )
            break

    if not injected:
        return report  # Model got everything right; no changes needed

    # Merge and recompute score
    merged_violations = existing_violations + injected
    deduction = sum(
        _SEVERITY_DEDUCTIONS.get(v.get("violation_type", ""), 8)
        for v in merged_violations
        if isinstance(v, dict)
    )
    new_score = max(5, 100 - deduction)

    # Rebuild global reasoning
    vnames = ", ".join(
        v.get("violation_type", "").replace("_", " ").title()
        for v in merged_violations if isinstance(v, dict)
    )
    base_reasoning = report.get("global_legal_reasoning", "")
    if injected:
        scl_note = (
            f" Statutory Compliance Layer additionally detected "
            f"{len(injected)} violation(s) via explicit textual evidence: "
            + ", ".join(v["violation_type"].replace("_", " ").title() for v in injected) + "."
        )
        global_reasoning = (
            f"Forensic audit of {domain} across operative legal sections identified "
            f"{len(merged_violations)} statutory violation(s) under the DPDP Act 2023: "
            f"{vnames}.{scl_note}"
        )
    else:
        global_reasoning = base_reasoning

    updated_report = dict(report)
    updated_report["violations"] = merged_violations
    updated_report["dpdp_trust_score"] = new_score
    updated_report["global_legal_reasoning"] = global_reasoning
    if "subtlety_score" not in updated_report or updated_report["subtlety_score"] == 0:
        updated_report["subtlety_score"] = 75

    print(
        f"[SCL] {domain}: Score updated {report.get('dpdp_trust_score', '?')} -> {new_score} "
        f"| Violations: {len(existing_violations)} model + {len(injected)} SCL = {len(merged_violations)} total"
    )
    return updated_report
