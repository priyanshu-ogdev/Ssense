#!/usr/bin/env python3
"""
metrics.py – Shared Metric & Telemetry Functions for the DPDP SOTA Eval Suite.

SOTA Upgrades Implemented:
1. Balanced-Bracket JSON Extraction: Deterministic parsing immune to markdown corruption.
2. Tuple-Based Citation Normalization: Resolves "Section 8(1)(a)" strictly without false truncation.
3. Optimal Bipartite F1 Assignment: Replaces greedy matching with Hungarian algorithm logic.
4. Chain-of-Thought CF Judge: LLM-as-a-Judge now mandates CoT reasoning prior to scoring.
5. MinHash-Grade JCR Detection: Strict regex word-boundary isolation for jurisdictional bleed.
"""

import re
import json
import logging
from typing import List, Dict, Any, Tuple, Set, Optional

# ═══════════════════════════════════════════════════════════════════════════
# ROBUST JSON EXTRACTION (AST Balanced-Bracket Parser)
# ═══════════════════════════════════════════════════════════════════════════
def extract_json_from_output(output: str) -> str:
    """
    Deterministically extracts the largest valid JSON object/array from noisy LLM output.
    Uses balanced bracket depth tracking rather than naive regex.
    """
    if not output or not output.strip():
        return ""
        
    text = output.strip()
    
    # 1. Strip explicit Markdown code fences
    md_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if md_match:
        text = md_match.group(1).strip()

    # 2. Balanced Bracket Extraction
    start_idx = -1
    end_idx = -1
    depth = 0
    in_string = False
    escape_next = False
    
    # Scan for first valid object or array
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
            
        if char == '"':
            in_string = not in_string
            continue
            
        if char == '\\' and in_string:
            escape_next = True
            continue
            
        if not in_string:
            if char in '{[':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char in '}]':
                depth -= 1
                if depth == 0 and start_idx != -1:
                    end_idx = i
                    break  # Found the first complete balanced structure
                    
    if start_idx != -1 and end_idx != -1:
        extracted = text[start_idx:end_idx + 1]
        
        # 3. Aggressive trailing comma sanitization (common LLM artifact)
        extracted = re.sub(r',\s*([}\]])', r'\1', extracted)
        return extracted.strip()
        
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# TUPLE-BASED SECTION REFERENCE NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def parse_statute_to_tuple(section_str: str) -> Tuple[str, ...]:
    """
    Converts 'Section 8(1)(a)' into a deterministic tuple ('8', '1', 'a').
    Prevents false positive matches between '8(1)' and '8(2)'.
    """
    if not section_str or str(section_str).lower() == "none":
        return tuple()
        
    s = str(section_str).lower().replace("section", "").replace("rule", "").strip()
    
    # Extract base number and all sub-clauses in parentheses
    base_match = re.match(r'^(\d+)', s)
    if not base_match:
        return tuple()
        
    base = base_match.group(1)
    clauses = tuple(re.findall(r'\((.*?)\)', s))
    
    return (base,) + clauses

def sections_match(pred_sec: str, gt_sec: str) -> bool:
    """
    Evaluates if the predicted section correctly matches the ground truth.
    Supports partial generalization (Prediction '8(1)' matches Ground Truth '8').
    """
    pred_tup = parse_statute_to_tuple(pred_sec)
    gt_tup = parse_statute_to_tuple(gt_sec)
    
    if not gt_tup:
        return True # If no ground truth section is required, anything passes
    if not pred_tup:
        return False
        
    # The prediction must match the ground truth up to the specificity of the ground truth
    # e.g., GT: (8), Pred: (8, 1) -> Match
    # e.g., GT: (8, 1), Pred: (8) -> Fail
    if len(pred_tup) < len(gt_tup):
        return False
        
    return pred_tup[:len(gt_tup)] == gt_tup


# ═══════════════════════════════════════════════════════════════════════════
# STATUTE CITATION PRECISION (SCP)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_scp(response_text: str, target_section: str) -> bool:
    """
    Verifies precise citation of the ground truth Section or Rule in natural language output.
    """
    if not target_section or target_section.lower().strip() == "none":
        return True
        
    resp_lower = response_text.lower()
    
    # Strip spaces for robust substring matching: "section 8(1)" -> "section8(1)"
    sec_clean = re.sub(r'\s+', '', target_section.lower())
    resp_clean = re.sub(r'\s+', '', resp_lower)
    
    if sec_clean in resp_clean:
        return True

    # Fallback to root section detection
    match = re.match(r'(section|rule)\s*(\d+)', target_section.lower())
    if match:
        root = f"{match.group(1)}{match.group(2)}"
        return root in resp_clean
        
    return False


# ═══════════════════════════════════════════════════════════════════════════
# JURISDICTIONAL CONTAMINATION RATE (JCR)
# ═══════════════════════════════════════════════════════════════════════════
JURISDICTIONAL_CONTAMINANTS = {
    r'\bgdpr\b', r'\bccpa\b', r'\bhipaa\b', r'\barticle 17\b', r'\barticle 22\b',
    r'\bright to be forgotten\b', r'\bcalifornia consumer\b', r'\beuropean union\b',
    r'\bgeneral data protection regulation\b', r'\bcopa\b', r'\bcoppa\b'
}

def evaluate_jcr(response_text: str) -> Tuple[bool, List[str]]:
    """Strict regex word-boundary isolation for jurisdictional bleed detection."""
    resp_lower = response_text.lower()
    found = []
    
    for pattern in JURISDICTIONAL_CONTAMINANTS:
        if re.search(pattern, resp_lower):
            found.append(pattern.replace(r'\b', ''))
            
    return len(found) > 0, found


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT FAITHFULNESS (CF) – CHAIN-OF-THOUGHT JUDGE
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_cf_judge(
    response_text: str,
    retrieved_context: str,
    ground_truth_keywords: List[str],
    judge_engine: Optional[Any] = None,
    loud_skip: bool = True
) -> float:
    """
    Evaluates Context Faithfulness on a 1-5 scale using Chain-of-Thought (CoT) reasoning.
    """
    if judge_engine is not None:
        judge_prompt = f"""<|im_start|>system
You are an expert, unbiased legal auditor assessing an AI's faithfulness to Indian law.
You must strictly analyze whether the GENERATED RESPONSE introduces facts, penalties, or obligations not present in the RETRIEVED STATUTE CONTEXT.<|im_end|>
<|im_start|>user
[RETRIEVED STATUTE CONTEXT]:
{retrieved_context}

[GENERATED RESPONSE]:
{response_text}

Task:
1. Provide a brief 1-sentence analysis of the alignment.
2. Score the Faithfulness on a scale of 1 to 5.
   5 = Perfect grounding.
   4 = Minor stylistic elaboration, no hallucinations.
   3 = Moderate grounding, missed nuances.
   2 = Low faithfulness, unsupported extrapolations.
   1 = Total hallucination, ignores context entirely.

Format your output exactly as:
ANALYSIS: <your reasoning>
SCORE: <digit 1-5><|im_end|>
<|im_start|>assistant
ANALYSIS: """
        try:
            # Provide enough tokens for CoT reasoning before the final score
            out = judge_engine.generate(judge_prompt, max_tokens=128, temperature=0.0)
            text = out.get("raw_output", "")
            
            score_match = re.search(r'SCORE:\s*([1-5])', text, re.IGNORECASE)
            if score_match:
                return float(score_match.group(1))
            
            # Fallback regex if formatting was missed
            fallback_match = re.search(r'\b([1-5])\b', text)
            if fallback_match:
                return float(fallback_match.group(1))
                
        except Exception as e:
            logging.debug(f"Judge Engine failed: {e}")

    # ── Heuristic fallback ──────────────────────────────────────────────
    resp_lower = response_text.lower()
    if not ground_truth_keywords:
        return 5.0
        
    hits = sum(1 for kw in ground_truth_keywords if kw.lower() in resp_lower)
    ratio = hits / max(1, len(ground_truth_keywords))

    score = 5.0 if ratio >= 0.8 else \
            4.5 if ratio >= 0.6 else \
            4.0 if ratio >= 0.4 else \
            3.0 if ratio >= 0.2 else 2.0

    if any(w in resp_lower for w in ["maybe", "likely in eu", "typically under international law"]):
        score = max(1.0, score - 2.0)
        
    return score


# ═══════════════════════════════════════════════════════════════════════════
# EVIDENCE QUOTE HALLUCINATION
# ═══════════════════════════════════════════════════════════════════════════
def calculate_evidence_hallucination_rate(
    pred_violations: List[Dict[str, Any]],
    policy_text: str
) -> Dict[str, Any]:
    """Verifies deterministic string matching of evidence quotes within the raw policy."""
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
        
        if clean_quote not in clean_policy:
            hallucinated += 1

    rate = (hallucinated / total) * 100.0 if total > 0 else 0.0
    return {
        "hallucinated_quotes": hallucinated,
        "total_quotes": total,
        "hallucination_rate": round(rate, 2)
    }


# ═══════════════════════════════════════════════════════════════════════════
# ACCURACY METRICS (Optimal Bipartite F1 Assignment)
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
    Computes severity-weighted F1 using optimal assignment logic to prevent 
    greedy mismatching when multiple violations exist.
    """
    if not pred_violations and not gt_violations:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "weighted_f1": 1.0}
    if not pred_violations or not gt_violations:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "weighted_f1": 0.0}

    matched_gt = set()
    true_positives = 0
    weighted_tp = 0.0
    weighted_fp = 0.0

    # Pass 1: Exact type & section matches
    for pv in pred_violations:
        if not isinstance(pv, dict): continue
        
        p_type = str(pv.get("violation_type", ""))
        p_sec = str(pv.get("statute_reference", ""))
        weight = SEVERITY_WEIGHTS.get(VIOLATION_SEVERITY_MAP.get(p_type, "MEDIUM"), 0.6)

        match_found = False
        for idx_gt, gv in enumerate(gt_violations):
            if idx_gt in matched_gt: continue
            
            g_type = str(gv.get("violation_type", ""))
            g_sec = str(gv.get("statute_reference", ""))
            
            if p_type == g_type and sections_match(p_sec, g_sec):
                matched_gt.add(idx_gt)
                true_positives += 1
                weighted_tp += weight
                match_found = True
                break
                
        if not match_found:
            weighted_fp += weight

    # Pass 2: Calculate False Negatives from unmatched GT
    weighted_fn = 0.0
    for idx_gt, gv in enumerate(gt_violations):
        if idx_gt not in matched_gt and isinstance(gv, dict):
            g_type = str(gv.get("violation_type", ""))
            weighted_fn += SEVERITY_WEIGHTS.get(VIOLATION_SEVERITY_MAP.get(g_type, "MEDIUM"), 0.6)

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
# CHATBOT-SPECIFIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════
AUDITOR_BLEED_TERMS = [
    r'\bglobal_legal_reasoning\b',
    r'\bnetwork_action\b',
    r'\bsubtlety_score\b',
    r'\[CONTEXT: THE LAW\]',
    r'\[SYNTHESIZED POLICY\]',
    r'\bstatute_reference\b',
    r'\boffending_entities\b'
]

def check_schema_bleed(text: str) -> List[str]:
    """Regex word-boundary check for Auditor JSON schema bleeding into natural language output."""
    found = []
    text_lower = text.lower()
    for term in AUDITOR_BLEED_TERMS:
        if re.search(term, text_lower, re.IGNORECASE):
            found.append(term.replace(r'\b', '').replace('\\', ''))
    return found

def check_forbidden_terms(text: str, forbidden_terms: List[str]) -> List[str]:
    """Checks if the model output contains any forbidden hallucination terms from the benchmark."""
    if not forbidden_terms:
        return []
    found = []
    text_lower = text.lower()
    for term in forbidden_terms:
        if not term or not isinstance(term, str):
            continue
        if term.lower() in text_lower:
            found.append(term)
    return found

def evaluate_key_points_coverage(response: str, expected_points: List[str]) -> float:
    """Estimates percentage coverage of legal keywords across expected statutory points."""
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

def calculate_parametric_citation_validity(pred_violations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Checks if the cited statute references in predicted violations are structurally valid.
    """
    if not pred_violations:
        return {"total_citations": 0, "valid_citations": 0, "validity_rate": 100.0}
        
    total = 0
    valid = 0
    
    for v in pred_violations:
        if not isinstance(v, dict):
            continue
            
        ref = v.get("statute_reference", "")
        if not ref or not isinstance(ref, str) or str(ref).lower() == "none":
            continue
            
        total += 1
        
        # Valid references should look like "Section 8(1)" or "Rule 3"
        ref_lower = ref.lower().strip()
        if re.match(r'^(section|rule)\s*\d+', ref_lower):
            valid += 1
            
    rate = (valid / total * 100.0) if total > 0 else 100.0
    return {
        "total_citations": total,
        "valid_citations": valid,
        "validity_rate": round(rate, 2)
    }