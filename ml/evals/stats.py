#!/usr/bin/env python3
"""
stats.py – Statistical Utilities for the DPDP Eval Suite.

SOTA Upgrades Implemented:
1. Wilson Score Confidence Intervals (Lower & Upper) for binomial rate metrics.
2. MTLD (Measure of Textual Lexical Diversity) – length-unbiased replacement for TTR.
3. Bootstrap Confidence Intervals for continuous distributions.
4. Smart Boundary Evaluation: Solves the "Wilson Paradox" where finite sample sizes 
   make strict 0.0% or 100.0% targets mathematically impossible to pass under CI bounds.
"""

import math
import re
from typing import List, Tuple, Optional, Dict, Any

# ═══════════════════════════════════════════════════════════════════════════
# WILSON SCORE CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════
def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson score confidence interval for a binomial proportion.
    Behaves correctly at extreme proportions (0% or 100%) and small N.
    Returns (lower_bound_pct, upper_bound_pct).
    """
    if total <= 0:
        return (0.0, 100.0)

    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(confidence, 1.96)

    p_hat = successes / total
    denominator = 1 + z * z / total

    centre = p_hat + z * z / (2 * total)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total)

    lower = max(0.0, (centre - margin) / denominator)
    upper = min(1.0, (centre + margin) / denominator)

    return (round(lower * 100, 2), round(upper * 100, 2))

def wilson_ci_from_pct(point_pct: float, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Helper to compute Wilson CI directly from a percentage and N."""
    successes = int((max(0.0, min(100.0, point_pct)) / 100.0) * total)
    return wilson_ci(successes, total, confidence)

# Alias for backward compatibility (used by compare_sota_models.py)
wilson_score_interval = wilson_ci

# ═══════════════════════════════════════════════════════════════════════════
# SMART STATISTICAL GATING (Solving the Wilson Paradox)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_metric_against_target(
    point_val: float, 
    n_val: Optional[int], 
    target: float, 
    op: str, 
    is_rate: bool = True
) -> Tuple[bool, float, str]:
    """
    Evaluates a metric against a target threshold.
    
    The Wilson Paradox Fix: 
    If a target is an absolute boundary (0.0 or 100.0), a Wilson CI mathematically 
    cannot touch the boundary unless N is infinite. Thus, for 0.0 or 100.0 targets, 
    we evaluate the Point Estimate. For all fuzzy targets (e.g., 95.0), we strictly 
    enforce the Wilson Lower/Upper bounds to guarantee statistical significance.
    
    Returns: (Passed, Evaluated_Value, Display_String)
    """
    if not is_rate or n_val is None or n_val <= 0:
        # Continuous metric (e.g., Latency, F1, MAE) -> Use Point Estimate
        passed = (point_val >= target) if op == ">=" else (point_val <= target)
        return passed, point_val, f"{point_val:.2f} (Point)"

    # Compute bounds
    lower_bound, upper_bound = wilson_ci_from_pct(point_val, n_val)
    
    # Absolute Boundary Handling (0.0 or 100.0)
    if (op == "<=" and target == 0.0) or (op == ">=" and target == 100.0):
        passed = (point_val <= target) if op == "<=" else (point_val >= target)
        ci_str = f"{point_val:.1f}% (CI: {lower_bound:.1f}-{upper_bound:.1f})"
        return passed, point_val, ci_str
        
    # Fuzzy Boundary Handling -> Rigorous Wilson CI Enforcement
    if op == ">=":
        passed = lower_bound >= target
        eval_val = lower_bound
        display = f"{point_val:.1f}% (Wilson L-Bound: {lower_bound:.1f}%)"
    else:  # op == "<="
        passed = upper_bound <= target
        eval_val = upper_bound
        display = f"{point_val:.1f}% (Wilson U-Bound: {upper_bound:.1f}%)"
        
    return passed, eval_val, display


# ═══════════════════════════════════════════════════════════════════════════
# MTLD – MEASURE OF TEXTUAL LEXICAL DIVERSITY
# ═══════════════════════════════════════════════════════════════════════════
def _mtld_forward(tokens: List[str], threshold: float = 0.72) -> float:
    if not tokens: return 0.0
    factors = 0.0
    factor_start = 0

    for i in range(len(tokens)):
        segment = tokens[factor_start:i + 1]
        ttr = len(set(segment)) / len(segment)
        if ttr <= threshold:
            factors += 1.0
            factor_start = i + 1

    if factor_start < len(tokens):
        remaining = tokens[factor_start:]
        ttr = len(set(remaining)) / len(remaining) if remaining else 1.0
        if ttr < 1.0:
            factors += (1.0 - ttr) / (1.0 - threshold)

    return len(tokens) / factors if factors > 0 else float(len(tokens))

def mtld(text: str, threshold: float = 0.72) -> float:
    """Computes MTLD (Measure of Textual Lexical Diversity) - independent of length."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    if len(tokens) < 10:
        return len(set(tokens)) / len(tokens) if tokens else 0.0
    forward = _mtld_forward(tokens, threshold)
    backward = _mtld_forward(list(reversed(tokens)), threshold)
    return (forward + backward) / 2.0

# ═══════════════════════════════════════════════════════════════════════════
# VALID DPDP SECTIONS (Parametric Memory Audit)
# ═══════════════════════════════════════════════════════════════════════════
VALID_DPDP_SECTIONS = {
    *{f"Section {i}" for i in range(1, 45)},
    *{f"Rule {i}" for i in range(1, 23)},
    "Section 8(1)", "Section 8(2)", "Section 8(5)", "Section 8(7)",
    "Rule 3(1)", "Rule 3(2)", "Rule 8(1)", "Rule 8(2)", "Rule 8(3)"
}

def is_valid_dpdp_citation(citation: str) -> bool:
    """Verifies citation exists in the statutory universe."""
    clean = citation.strip()
    if clean in VALID_DPDP_SECTIONS: return True
    match = re.match(r'(Section|Rule)\s+(\d+)', clean, re.IGNORECASE)
    if match:
        root = f"{match.group(1).capitalize()} {match.group(2)}"
        return root in VALID_DPDP_SECTIONS
    return False