#!/usr/bin/env python3
"""
stats.py – Statistical Utilities for the DPDP Eval Suite.

Provides:
    - Wilson score confidence intervals for rate metrics (binary proportions)
    - MTLD (Measure of Textual Lexical Diversity) – length-unbiased replacement for TTR
    - Bootstrap confidence intervals for continuous metrics
    - Threshold gating with CI lower bounds

All certification gating should use `wilson_ci_lower()` for rate metrics,
not bare point estimates. At N<30, a "98%" point estimate has a Wilson 95% CI
spanning 60-100% — reporting only the point estimate materially overstates certainty.
"""

import math
import re
from typing import List, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════
# WILSON SCORE CONFIDENCE INTERVAL
# ═══════════════════════════════════════════════════════════════════════════
def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a binomial proportion.

    Returns (lower_bound, upper_bound) as percentages (0-100).

    The Wilson interval is preferred over the normal approximation (Wald interval)
    because it behaves correctly at extreme proportions and small N.

    Args:
        successes: Number of successes (e.g., passed tests).
        total: Total number of trials.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound_pct, upper_bound_pct).
    """
    if total == 0:
        return (0.0, 100.0)

    # Z-score for common confidence levels (avoid scipy dependency)
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(confidence, 1.96)

    p_hat = successes / total
    denominator = 1 + z * z / total

    centre = p_hat + z * z / (2 * total)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total)

    lower = max(0.0, (centre - margin) / denominator)
    upper = min(1.0, (centre + margin) / denominator)

    return (round(lower * 100, 2), round(upper * 100, 2))


def wilson_ci_lower(successes: int, total: int, confidence: float = 0.95) -> float:
    """
    Return the lower bound of the Wilson CI as a percentage.

    Use this for threshold gating: compare this value (not the point estimate)
    against the certification threshold.
    """
    lower, _ = wilson_ci(successes, total, confidence)
    return lower


def rate_with_ci(successes: int, total: int, confidence: float = 0.95) -> dict:
    """
    Compute a rate metric with full CI metadata for inclusion in reports.

    Returns a dict with:
        - point_estimate_pct: bare percentage
        - ci_lower_pct: Wilson CI lower bound
        - ci_upper_pct: Wilson CI upper bound
        - n: sample size
        - confidence: confidence level used
    """
    point = (successes / total * 100) if total > 0 else 0.0
    lower, upper = wilson_ci(successes, total, confidence)
    return {
        "point_estimate_pct": round(point, 2),
        "ci_lower_pct": lower,
        "ci_upper_pct": upper,
        "n": total,
        "confidence": confidence
    }


# ═══════════════════════════════════════════════════════════════════════════
# MTLD – MEASURE OF TEXTUAL LEXICAL DIVERSITY
# ═══════════════════════════════════════════════════════════════════════════
def _mtld_forward(tokens: List[str], threshold: float = 0.72) -> float:
    """
    One-directional MTLD pass.

    MTLD counts the number of "factors" — contiguous sub-sequences of tokens
    where the TTR drops below a threshold. The final MTLD score is
    total_tokens / number_of_factors.

    A higher MTLD indicates greater lexical diversity.
    """
    if not tokens:
        return 0.0

    factors = 0.0
    factor_start = 0

    for i in range(len(tokens)):
        segment = tokens[factor_start:i + 1]
        ttr = len(set(segment)) / len(segment)
        if ttr <= threshold:
            factors += 1.0
            factor_start = i + 1

    # Partial factor at end
    if factor_start < len(tokens):
        remaining = tokens[factor_start:]
        ttr = len(set(remaining)) / len(remaining) if remaining else 1.0
        if ttr < 1.0:
            factors += (1.0 - ttr) / (1.0 - threshold)

    if factors == 0:
        return float(len(tokens))

    return len(tokens) / factors


def mtld(text: str, threshold: float = 0.72) -> float:
    """
    Compute MTLD (Measure of Textual Lexical Diversity).

    Unlike TTR, MTLD is length-independent: it does not systematically decrease
    with longer texts, making it suitable for comparing responses of varying length.

    Args:
        text: The text to analyze.
        threshold: The TTR threshold for factor boundaries (standard: 0.72).

    Returns:
        MTLD score (higher = more diverse vocabulary).
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    if len(tokens) < 10:
        # Too short for meaningful MTLD; fall back to simple TTR
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    forward = _mtld_forward(tokens, threshold)
    backward = _mtld_forward(list(reversed(tokens)), threshold)

    return (forward + backward) / 2.0


def calculate_ttr(text: str) -> float:
    """
    Legacy TTR calculation – kept for backward compatibility and comparison.
    Prefer mtld() for new code.
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


# ═══════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVALS (for continuous metrics)
# ═══════════════════════════════════════════════════════════════════════════
def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for the mean of continuous values.

    Returns (mean, ci_lower, ci_upper).

    Args:
        values: List of observed values.
        confidence: Confidence level.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
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


# ═══════════════════════════════════════════════════════════════════════════
# VALID DPDP SECTIONS (for Parametric Citation Validity)
# ═══════════════════════════════════════════════════════════════════════════
# Master list of all valid sections in the DPDP Act 2023 and Rules 2025.
# Used by the audit track to verify citations are plausible without
# requiring the statute text in-context (closed-book parametric memory).
VALID_DPDP_SECTIONS = {
    # DPDP Act 2023 — Sections 1 through 44
    *{f"Section {i}" for i in range(1, 45)},
    # Common sub-clauses
    "Section 4(1)", "Section 4(2)",
    "Section 5(1)", "Section 5(2)", "Section 5(3)",
    "Section 6(1)", "Section 6(2)", "Section 6(3)", "Section 6(4)",
    "Section 6(5)", "Section 6(6)", "Section 6(7)", "Section 6(8)", "Section 6(9)",
    "Section 7(1)", "Section 7(2)", "Section 7(3)", "Section 7(4)",
    "Section 7(5)", "Section 7(6)", "Section 7(7)", "Section 7(8)", "Section 7(9)",
    "Section 8(1)", "Section 8(2)", "Section 8(3)", "Section 8(4)",
    "Section 8(5)", "Section 8(6)", "Section 8(7)", "Section 8(8)", "Section 8(9)",
    "Section 9(1)", "Section 9(2)", "Section 9(3)", "Section 9(4)", "Section 9(5)",
    "Section 10(1)", "Section 10(2)", "Section 10(3)",
    "Section 11(1)", "Section 11(2)", "Section 11(3)", "Section 11(4)",
    "Section 12(1)", "Section 12(2)", "Section 12(3)",
    "Section 13(1)", "Section 13(2)", "Section 13(3)",
    "Section 14(1)", "Section 14(2)",
    "Section 15(1)", "Section 15(2)", "Section 15(3)",
    "Section 16(1)", "Section 16(2)", "Section 16(3)",
    "Section 17(1)", "Section 17(2)", "Section 17(3)", "Section 17(4)", "Section 17(5)",
    "Section 18(1)", "Section 18(2)", "Section 18(3)",
    "Section 33(1)", "Section 33(2)",
    # DPDP Rules 2025 — Rules 1 through 22
    *{f"Rule {i}" for i in range(1, 23)},
    # Common rule sub-clauses
    "Rule 3(1)", "Rule 3(2)", "Rule 3(3)",
    "Rule 4(1)", "Rule 4(2)", "Rule 4(3)", "Rule 4(4)",
    "Rule 5(1)", "Rule 5(2)",
    "Rule 6(1)", "Rule 6(2)", "Rule 6(3)",
    "Rule 7(1)", "Rule 7(2)", "Rule 7(3)",
    "Rule 8(1)", "Rule 8(2)", "Rule 8(3)", "Rule 8(4)",
    "Rule 9(1)", "Rule 9(2)",
    "Rule 10(1)", "Rule 10(2)", "Rule 10(3)",
    "Rule 12(1)", "Rule 12(2)",
}


def is_valid_dpdp_citation(citation: str) -> bool:
    """
    Check if a statute citation is a valid DPDP Act 2023 or Rules 2025 reference.

    Used for the audit track's "Parametric Citation Validity" metric —
    since the auditor uses parametric memory (no statute text in the prompt),
    we check whether cited sections actually exist, not whether they were in-context.

    Args:
        citation: A citation string like "Section 8(5)" or "Rule 3(2)".

    Returns:
        True if the citation matches a known valid section/rule.
    """
    citation_clean = citation.strip()
    if citation_clean in VALID_DPDP_SECTIONS:
        return True

    # Try root section (e.g., "Section 8(5)" → "Section 8")
    match = re.match(r'(Section|Rule)\s+(\d+)', citation_clean, re.IGNORECASE)
    if match:
        root = f"{match.group(1)} {match.group(2)}"
        # Capitalize consistently
        root = root[0].upper() + root[1:]
        return root in VALID_DPDP_SECTIONS

    return False
