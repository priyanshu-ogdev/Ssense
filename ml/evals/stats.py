#!/usr/bin/env python3
"""
stats.py – Statistical Utilities for the DPDP Eval Suite.

SOTA Upgrades Implemented:
1. Floating-Point Truncation Fix: Replaced naive `int()` casting with `int(round())` to prevent 
   the systemic loss of success counts when converting percentages back to integers.
2. Wilson Score Confidence Intervals (Lower & Upper) for binomial rate metrics.
3. Smart Boundary Evaluation: Solves the "Wilson Paradox" on N=60 strict boundary thresholds.
4. Cleaned Dead Code: Removed obsolete statutory lists (now internalized via AST in `metrics.py`).
"""

import math
import re
from typing import List, Tuple, Optional

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

    return (round(lower * 100.0, 2), round(upper * 100.0, 2))

def wilson_ci_from_pct(point_pct: float, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Helper to compute Wilson CI directly from a percentage and N.
    SOTA FIX: Applies standard rounding before int conversion to prevent float-truncation 
    from deleting successful test passes (e.g., int(24.999) -> 24 vs round(24.999) -> 25).
    """
    if total <= 0:
        return (0.0, 100.0)
        
    pct_clamped = max(0.0, min(100.0, point_pct))
    successes = int(round((pct_clamped / 100.0) * total))
    
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
    """Core forward pass for MTLD."""
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
    """
    Computes MTLD (Measure of Textual Lexical Diversity).
    Unlike TTR, MTLD is mathematically independent of generation length.
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    if len(tokens) < 10:
        return len(set(tokens)) / len(tokens) if tokens else 0.0
    
    forward = _mtld_forward(tokens, threshold)
    backward = _mtld_forward(list(reversed(tokens)), threshold)
    return (forward + backward) / 2.0


# ═══════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVALS (Continuous Metrics)
# ═══════════════════════════════════════════════════════════════════════════
def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes a bootstrap confidence interval for the mean of continuous values.
    Returns (mean, ci_lower, ci_upper).
    """
    import random
    if not values:
        return (0.0, 0.0, 0.0)

    rng = random.Random(seed)
    n = len(values)
    means = []

    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1 - confidence
    lower_idx = int(math.floor(alpha / 2 * n_bootstrap))
    upper_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1

    mean_val = sum(values) / n
    return (
        round(mean_val, 4),
        round(means[lower_idx], 4),
        round(means[min(upper_idx, len(means) - 1)], 4)
    )