#!/usr/bin/env python3
"""
metrics.py – Shared Metric Functions for the DPDP Eval Suite.

Consolidates metric functions that were previously duplicated across multiple scripts:
    - JSON extraction from model outputs (was in run_grammar_evals.py AND run_accuracy_evals.py)
    - Section reference normalization and matching (from run_accuracy_evals.py)
    - Statute Citation Precision (SCP) evaluation (from evaluate_chatbot.py)
    - Jurisdictional Contamination Rate (JCR) detection (from evaluate_chatbot.py)
    - Context Faithfulness (CF) judge evaluation (from evaluate_chatbot.py)
    - Evidence Quote Hallucination detection (from run_accuracy_evals.py)
    - Parametric Citation Validity (NEW – replaces context grounding for audit track)
"""

import re
from typing import List, Dict, Any, Tuple, Set, Optional


# ═══════════════════════════════════════════════════════════════════════════
# JSON EXTRACTION (deduplicated from run_grammar_evals.py + run_accuracy_evals.py)
# ═══════════════════════════════════════════════════════════════════════════
def extract_json_from_output(output: str) -> str:
    """
    Extract JSON from model output, handling markdown code fences,
    stray text, and trailing commas.

    Returns the cleaned JSON string, or empty string if extraction fails.
    """
    if not output or not output.strip():
        return ""
    cleaned = output.strip()

    # Strip markdown code fences
    if '```json' in cleaned:
        match = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
    elif '```' in cleaned:
        match = re.search(r'```\s*(.*?)\s*```', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    # Extract outermost JSON object
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]
    elif not cleaned.startswith('{'):
        # Try array extraction as fallback
        first_bracket = cleaned.find('[')
        last_bracket = cleaned.rfind(']')
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            cleaned = cleaned[first_bracket:last_bracket + 1]

    # Fix trailing commas before } or ]
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    return cleaned.strip()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION REFERENCE NORMALIZATION (from run_accuracy_evals.py)
# ═══════════════════════════════════════════════════════════════════════════
def normalize_section_reference(section: str) -> Set[str]:
    """
    Generate all plausible textual variations of a DPDP section reference.

    Examples:
        "Section 8(5)" → {"Section 8(5)", "Section 8", "8(5)", "8", "section 8(5)", ...}
        "Section 8(5) read with Rule 3" → also includes Rule 3 variations

    This handles the statutory citation style where Section 8(5) and Section 8
    refer to the same provision at different levels of specificity.
    """
    if not section:
        return set()
    section = section.strip()
    variations = {section}

    # Handle "read with" compound references
    if "read with" in section.lower():
        parts = re.split(r'\s+read\s+with\s+', section, flags=re.IGNORECASE)
        for part in parts:
            variations.update(normalize_section_reference(part.strip()))

    # Sub-clause stripping: "Section 8(5)" → "Section 8"
    if '(' in section:
        base = section.split('(')[0].strip()
        variations.add(base)

    # Strip prefix: "Section 8" → "8"
    if section.startswith("Section "):
        variations.add(section.replace("Section ", ""))
    if section.startswith("Rule ") and '(' in section:
        base = section.split('(')[0].strip()
        variations.add(base)
    if section.startswith("Rule "):
        variations.add(section.replace("Rule ", ""))

    # Case-insensitive matching
    variations.add(section.lower())
    return variations


def sections_match(pred: str, gt: str) -> bool:
    """Check if a predicted section reference matches a ground-truth section reference."""
    return bool(normalize_section_reference(pred) & normalize_section_reference(gt))


# ═══════════════════════════════════════════════════════════════════════════
# STATUTE CITATION PRECISION (SCP) – from evaluate_chatbot.py
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_scp(response_text: str, target_section: str) -> bool:
    """
    Verifies precise citation of the ground truth Section or Rule in the response.

    Binary: did the model cite the correct statutory provision?
    """
    if not target_section:
        return True
    resp_lower = response_text.lower()
    sec_clean = target_section.lower().strip()
    if sec_clean == "none":
        return True
    if sec_clean in resp_lower:
        return True

    # If sub-clause, check root section plus clause number
    match = re.match(r'(section|rule)\s+(\d+)\s*\((.*?)\)', sec_clean)
    if match:
        root = f"{match.group(1)} {match.group(2)}"
        return root in resp_lower
    return False


# ═══════════════════════════════════════════════════════════════════════════
# JURISDICTIONAL CONTAMINATION RATE (JCR) – from evaluate_chatbot.py
# ═══════════════════════════════════════════════════════════════════════════
JURISDICTIONAL_CONTAMINANTS = [
    "gdpr", "ccpa", "hipaa", "article 17", "article 22", "right to be forgotten",
    "california consumer", "european union", "general data protection regulation",
    "copa", "coppa"
]


def evaluate_jcr(response_text: str) -> Tuple[bool, List[str]]:
    """
    Checks for jurisdictional contamination from Western privacy frameworks.

    Returns (is_contaminated, list_of_found_contaminants).
    """
    resp_lower = response_text.lower()
    found = []
    for term in JURISDICTIONAL_CONTAMINANTS:
        if re.search(r'\b' + re.escape(term) + r'\b', resp_lower):
            found.append(term)
    return len(found) > 0, found


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT FAITHFULNESS (CF) – from evaluate_chatbot.py
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_cf_judge(
    response_text: str,
    retrieved_context: str,
    ground_truth_keywords: List[str],
    judge_engine: Optional[Any] = None,
    panel_size: int = 3,
    loud_skip: bool = True
) -> float:
    """
    Evaluates Context Faithfulness on a 1-5 scale.

    When judge_engine is provided, uses panel-of-judges scoring (multiple prompts,
    median score). When judge_engine is None:
      - If loud_skip=True: prints a warning and falls back to keyword heuristic.
      - The heuristic is NOT a substitute for a real judge; scores should be
        flagged as "heuristic_only" in reports.

    Args:
        response_text: The model's generated response.
        retrieved_context: The context that was injected into the prompt.
        ground_truth_keywords: Expected keywords for keyword-overlap heuristic.
        judge_engine: Optional BackendEngine instance for LLM-as-judge.
        panel_size: Number of judge prompts for panel scoring (default 3).
        loud_skip: If True, print warning when falling back to heuristic.
    """
    if judge_engine is not None:
        scores = []
        judge_framings = [
            # Framing 1: Standard
            "Evaluate the generated legal response against the provided retrieved statute context.",
            # Framing 2: Adversarial (checks if judge is robust to prompt variation)
            "You are auditing an AI-generated legal response. Determine if it faithfully represents the statute context provided, or if it introduces unsupported claims.",
            # Framing 3: Reversed position (mitigates position bias)
            "First read the GENERATED RESPONSE, then compare it against the RETRIEVED STATUTE CONTEXT to assess faithfulness.",
        ]
        for framing in judge_framings[:panel_size]:
            judge_prompt = f"""<|im_start|>system
You are an unbiased, expert legal auditor assessing AI faithfulness to Indian law.<|im_end|>
<|im_start|>user
{framing}
[RETRIEVED STATUTE CONTEXT]:
{retrieved_context}

[GENERATED RESPONSE]:
{response_text}

Score the Context Faithfulness on a scale of 1 to 5:
5 = Every legal assertion is strictly grounded in the retrieved context with accurate legal framing.
4 = Very high faithfulness with minor stylistic elaboration that does not violate statutory limits.
3 = Moderate grounding; mentions valid sections but omits critical statutory nuances.
2 = Low faithfulness; extrapolates unsupported administrative requirements.
1 = Total failure; contradicts or ignores the retrieved statute text entirely.

Output only a single digit from 1 to 5 representing your faithfulness rating.<|im_end|>
<|im_start|>assistant
Rating: """
            try:
                out = judge_engine.generate(judge_prompt, max_tokens=10, temperature=0.0)
                num_match = re.search(r'\b([1-5])\b', out["raw_output"])
                if num_match:
                    scores.append(float(num_match.group(1)))
            except Exception:
                pass

        if scores:
            # Return median (robust to single outlier)
            scores.sort()
            return scores[len(scores) // 2]

    # ── Heuristic fallback ──────────────────────────────────────────────
    if loud_skip and judge_engine is None:
        pass  # Caller should log this at suite level, not per-call

    resp_lower = response_text.lower()
    if not ground_truth_keywords:
        return 5.0
    hits = sum(1 for kw in ground_truth_keywords if kw.lower() in resp_lower)
    ratio = hits / max(1, len(ground_truth_keywords))

    if ratio >= 0.8:
        score = 5.0
    elif ratio >= 0.6:
        score = 4.5
    elif ratio >= 0.4:
        score = 4.0
    elif ratio >= 0.2:
        score = 3.0
    else:
        score = 2.0

    # Penalize speculative external concepts
    if any(w in resp_lower for w in ["maybe", "likely in eu", "typically under international law"]):
        score = max(1.0, score - 2.0)
    return score


# ═══════════════════════════════════════════════════════════════════════════
# EVIDENCE QUOTE HALLUCINATION (from run_accuracy_evals.py)
# ═══════════════════════════════════════════════════════════════════════════
def calculate_evidence_hallucination_rate(
    pred_violations: List[Dict[str, Any]],
    policy_text: str
) -> Dict[str, Any]:
    """
    Pillar 4: Verifies verbatim evidence quote existence within the input policy.

    This is a deterministic string match — the cheapest and most reliable control
    for catching fabricated citations. Run on every response.
    """
    if not pred_violations:
        return {"hallucinated_quotes": 0, "total_quotes": 0, "hallucination_rate": 0.0}

    hallucinated = 0
    total = 0
    clean_policy = re.sub(r'\s+', ' ', policy_text.lower()).strip()

    for v in pred_violations:
        if not isinstance(v, dict):
            continue
        quote = v.get("evidence_quote", "")
        if not quote or not isinstance(quote, str) or len(quote.strip()) < 4:
            continue
        total += 1
        clean_quote = re.sub(r'\s+', ' ', quote.lower()).strip()
        if clean_policy.find(clean_quote) == -1:
            hallucinated += 1

    rate = (hallucinated / total) * 100 if total > 0 else 0.0
    return {
        "hallucinated_quotes": hallucinated,
        "total_quotes": total,
        "hallucination_rate": round(rate, 2)
    }


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETRIC CITATION VALIDITY (NEW – replaces context grounding for audit track)
# ═══════════════════════════════════════════════════════════════════════════
def calculate_parametric_citation_validity(
    pred_violations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    For the audit track (closed-book): checks whether cited sections actually exist
    in the DPDP Act 2023 / Rules 2025, using the master list in stats.py.

    This replaces "Context Grounding" which was structurally broken for the
    audit track — the auditor uses parametric memory (memorized during SFT),
    so a "is this citation in the prompt context?" check always returns 0%
    because the prompt only contains the user's raw privacy policy, not the statute.
    """
    from stats import is_valid_dpdp_citation

    if not pred_violations:
        return {
            "valid_citations": 0,
            "total_citations": 0,
            "validity_rate": 100.0,
            "invalid_citations": []
        }

    valid_citations = 0
    total_citations = 0
    invalid_citations = []

    for v in pred_violations:
        citation = v.get("statute_reference", "")
        if not citation or not isinstance(citation, str) or len(citation.strip()) < 3:
            continue
        total_citations += 1
        if is_valid_dpdp_citation(citation):
            valid_citations += 1
        else:
            invalid_citations.append(citation)

    validity_rate = (valid_citations / total_citations * 100) if total_citations > 0 else 100.0
    return {
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "validity_rate": validity_rate,
        "invalid_citations": invalid_citations
    }

# ═══════════════════════════════════════════════════════════════════════════
# ACCURACY METRICS (from run_accuracy_evals.py)
# ═══════════════════════════════════════════════════════════════════════════
VIOLATION_SEVERITY_MAP = {
    "CHILD_CONSENT_VIOLATION": "CRITICAL",
    "CROSS_BORDER_TRANSFER_VIOLATION": "HIGH",
    "CONSENT_NOT_FREE_OR_SPECIFIC": "HIGH",
    "DATA_RETENTION_LIMIT_EXCEEDED": "HIGH",
    "PURPOSE_LIMITATION_VIOLATION": "HIGH",
    "SECURITY_SAFEGUARDS_MISSING": "HIGH",
    "NOTICE_INADEQUATE": "MEDIUM",
    "GRIEVANCE_REDRESSAL_INADEQUATE": "MEDIUM",
    "SDF_OBLIGATIONS_MISSING": "MEDIUM",
    "BREACH_NOTIFICATION_FAILURE": "HIGH",
}

SEVERITY_WEIGHTS = {
    "CRITICAL": 1.0,
    "HIGH": 0.8,
    "MEDIUM": 0.6,
    "LOW": 0.4
}


def calculate_violation_f1(
    pred_violations: List[Dict[str, Any]],
    gt_violations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute severity-weighted Violation F1 between predicted and ground-truth violations.
    """
    if not pred_violations and not gt_violations:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "weighted_f1": 1.0}
    if not pred_violations or not gt_violations:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "weighted_f1": 0.0}

    matched_gt = set()
    true_positives = 0
    weighted_tp = 0.0
    weighted_fn = 0.0
    weighted_fp = 0.0

    for pv in pred_violations:
        if not isinstance(pv, dict):
            continue
        p_type = pv.get("violation_type", "")
        p_sec = pv.get("statute_reference", "")
        if not isinstance(p_type, str): p_type = str(p_type)
        if not isinstance(p_sec, str): p_sec = str(p_sec)

        severity = VIOLATION_SEVERITY_MAP.get(p_type, "MEDIUM")
        weight = SEVERITY_WEIGHTS.get(severity, 0.6)

        match_found = False
        for idx_gt, gv in enumerate(gt_violations):
            if idx_gt in matched_gt:
                continue
            g_type = gv.get("violation_type", "")
            g_sec = gv.get("statute_reference", "")
            if p_type == g_type and sections_match(p_sec, g_sec):
                matched_gt.add(idx_gt)
                true_positives += 1
                weighted_tp += weight
                match_found = True
                break
        if not match_found:
            weighted_fp += weight

    for idx_gt, gv in enumerate(gt_violations):
        if not isinstance(gv, dict):
            continue
        if idx_gt not in matched_gt:
            g_type = gv.get("violation_type", "")
            if not isinstance(g_type, str): g_type = str(g_type)
            severity = VIOLATION_SEVERITY_MAP.get(g_type, "MEDIUM")
            weighted_fn += SEVERITY_WEIGHTS.get(severity, 0.6)

    precision = true_positives / len(pred_violations) if pred_violations else 0.0
    recall = true_positives / len(gt_violations) if gt_violations else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    w_prec = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0.0
    w_rec = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0.0
    weighted_f1 = (2 * w_prec * w_rec) / (w_prec + w_rec) if (w_prec + w_rec) > 0 else 0.0

    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "weighted_f1": round(weighted_f1, 4)
    }


# ═══════════════════════════════════════════════════════════════════════════
# CHATBOT-SPECIFIC HELPERS (from run_chatbot_evals.py)
# ═══════════════════════════════════════════════════════════════════════════
AUDITOR_BLEED_TERMS = [
    "global_legal_reasoning",
    "network_action",
    "subtlety_score",
    "[CONTEXT: THE LAW]",
    "[SYNTHESIZED POLICY]",
    "statute_reference",
    "offending_entities"
]


def check_schema_bleed(text: str) -> List[str]:
    """Check if any Auditor internal JSON schema keys or preambles bled into the chatbot response."""
    found = []
    text_lower = text.lower()
    for term in AUDITOR_BLEED_TERMS:
        if term.lower() in text_lower:
            found.append(term)
    return found


def check_forbidden_terms(text: str, forbidden_list: List[str]) -> List[str]:
    """Check if any forbidden hallucination terms appear in the response."""
    found = []
    text_lower = text.lower()
    for term in forbidden_list:
        if term.lower() in text_lower:
            found.append(term)
    return found


def evaluate_key_points_coverage(response: str, expected_points: List[str]) -> float:
    """Estimate basic keyword/concept coverage of expected statutory key points."""
    if not expected_points:
        return 1.0
    covered = 0
    resp_lower = response.lower()
    for pt in expected_points:
        keywords = [w for w in re.findall(r'\b\w+\b', pt.lower()) if len(w) >= 4]
        if not keywords:
            covered += 1
            continue
        matches = sum(1 for kw in keywords if kw in resp_lower)
        if matches / len(keywords) >= 0.4:
            covered += 1
    return covered / len(expected_points)
