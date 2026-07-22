#!/usr/bin/env python3
"""
run_accuracy_evals.py – Legal Reasoning Accuracy & Hallucination Evaluation (Universal Dual-Backend Grade)

Measures Pillars 2, 3, 4, and 5:
1. Violation F1 Score (Severity-Weighted & Section-Normalized)
2. Trust Score & Subtlety Score MAE
3. Evidence Quote Hallucination Rate (Exact Verbatim Substring Check)
4. Hardware Efficiency (TTFT & Throughput)
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import json
import time
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
from datetime import datetime, timezone
from tqdm import tqdm
import numpy as np

try:
    from backend_loader import BackendEngine
except ImportError:
    from ml.evals.backend_loader import BackendEngine

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_SCHEMA_PATH = Path("libs/contracts/schemas/dpdp_schema.json")
DEFAULT_GROUND_TRUTH_PATH = Path("ml/evals/holdout_policies/ground_truth.json")
DEFAULT_LAW_FILE_PATH = Path("ml/data-forge/dpdp_act_and_rules_2025.txt")
DEFAULT_MODEL_PATH = Path("../models/audit-model-final") if Path("../models/audit-model-final").exists() else Path("../models/Qwen3.5-9B")
REPORT_DIR = Path("ml/evals/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def load_law_context(law_path: Path) -> str:
    if not law_path.exists():
        return "Digital Personal Data Protection Act 2023 [Law context text loaded]"
    with open(law_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_test_data(gt_path: Path) -> List[Dict[str, Any]]:
    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    test_data = []
    for item in ground_truth:
        if 'policy_text_snippet' in item:
            content = item['policy_text_snippet']
        else:
            policy_file = gt_path.parent / item['filename']
            if not policy_file.exists():
                continue
            with open(policy_file, 'r', encoding='utf-8') as f:
                content = f.read()
        test_data.append({
            "case_id": item.get('case_id', item['filename']),
            "filename": item['filename'],
            "category": item.get('category', 'unknown'),
            "description": item.get('description', ''),
            "content": content,
            "expected_output": item.get('expected_output', {}),
            "evaluation_targets": item.get('evaluation_targets', {})
        })
    return test_data

def normalize_section_reference(section: str) -> Set[str]:
    if not section:
        return set()
    section = section.strip()
    variations = {section}
    if "read with" in section.lower():
        parts = re.split(r'\s+read\s+with\s+', section, flags=re.IGNORECASE)
        for part in parts:
            variations.update(normalize_section_reference(part.strip()))
    if '(' in section:
        base = section.split('(')[0].strip()
        variations.add(base)
    if section.startswith("Section "):
        variations.add(section.replace("Section ", ""))
    if section.startswith("Rule ") and '(' in section:
        base = section.split('(')[0].strip()
        variations.add(base)
    if section.startswith("Rule "):
        variations.add(section.replace("Rule ", ""))
    variations.add(section.lower())
    return variations

def sections_match(pred: str, gt: str) -> bool:
    return bool(normalize_section_reference(pred) & normalize_section_reference(gt))

def extract_json_from_output(output: str) -> str:
    if not output or not output.strip():
        return ""
    cleaned = output.strip()
    if '```json' in cleaned:
        match = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
    elif '```' in cleaned:
        match = re.search(r'```\s*(.*?)\s*```', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    return cleaned.strip()

# ═══════════════════════════════════════════════════════════════════════════
# METRIC CALCULATION ENGINES
# ═══════════════════════════════════════════════════════════════════════════
def calculate_violation_f1(pred_violations: List[Dict[str, Any]], gt_violations: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not pred_violations and not gt_violations:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "weighted_f1": 1.0}
    if not pred_violations or not gt_violations:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "weighted_f1": 0.0}

    matched_gt = set()
    true_positives = 0
    weighted_tp = 0.0
    weighted_fn = 0.0
    weighted_fp = 0.0

    for idx_pred, pv in enumerate(pred_violations):
        p_type = pv.get("violation_type", "")
        p_sec = pv.get("statute_reference", "")
        severity = VIOLATION_SEVERITY_MAP.get(p_type, "MEDIUM")
        weight = SEVERITY_WEIGHTS[severity]

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
        if idx_gt not in matched_gt:
            g_type = gv.get("violation_type", "")
            severity = VIOLATION_SEVERITY_MAP.get(g_type, "MEDIUM")
            weighted_fn += SEVERITY_WEIGHTS[severity]

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

def calculate_evidence_hallucination_rate(pred_violations: List[Dict[str, Any]], policy_text: str) -> Dict[str, Any]:
    """Pillar 4: Verifies verbatim quote existence within the input policy."""
    if not pred_violations:
        return {"hallucinated_quotes": 0, "total_quotes": 0, "hallucination_rate": 0.0}

    hallucinated = 0
    total = 0
    clean_policy = re.sub(r'\s+', ' ', policy_text.lower()).strip()

    for v in pred_violations:
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
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Pillar 2-5: Accuracy, Trust Score, Hallucination, & Latency Evals")
    parser.add_argument("--backend", type=str, default="llamacpp", choices=["unsloth", "vllm", "llamacpp"])
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--ground-truth-path", type=str, default=str(DEFAULT_GROUND_TRUTH_PATH))
    parser.add_argument("--law-path", type=str, default=str(DEFAULT_LAW_FILE_PATH))
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--lora-name", type=str, default="audit")
    args = parser.parse_args()

    test_data = load_test_data(Path(args.ground_truth_path))
    law_context = load_law_context(Path(args.law_path))
    if not test_data:
        print("⚠️ No test data found.")
        return

    print(f"🚀 Running Legal Reasoning & Hallucination Evals on {len(test_data)} policies across backend: {args.backend}...")
    engine = BackendEngine(
        backend_type=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        vllm_url=args.vllm_url,
        lora_name=args.lora_name
    )

    results = []
    f1_scores = []
    weighted_f1_scores = []
    trust_errors = []
    subtlety_errors = []
    total_quotes = 0
    total_hallucinated = 0

    for item in tqdm(test_data, desc="Evaluating Accuracy & Hallucination"):
        prompt = f"""<|im_start|>system
You are a strict DPDP Regulatory Auditor enforcing the Indian Digital Personal Data Protection (DPDP) Act 2023 and Rules 2025. Output ONLY valid JSON matching the dpdp_schema.<|im_end|>
<|im_start|>user
[CONTEXT: THE LAW]
{law_context[:8000]}

[SYNTHESIZED POLICY]
{item['content']}<|im_end|>
<|im_start|>assistant
"""
        out = engine.generate(prompt, max_tokens=2048, temperature=0.0)
        extracted = extract_json_from_output(out["raw_output"])
        parsed = {}
        try:
            parsed = json.loads(extracted) if extracted else {}
        except json.JSONDecodeError:
            pass

        pred_violations = parsed.get("violations", []) if isinstance(parsed, dict) else []
        gt_violations = item["expected_output"].get("violations", [])

        f1_metrics = calculate_violation_f1(pred_violations, gt_violations)
        f1_scores.append(f1_metrics["f1"])
        weighted_f1_scores.append(f1_metrics["weighted_f1"])

        pred_trust = parsed.get("dpdp_trust_score", 50) if isinstance(parsed, dict) else 50
        gt_trust = item["expected_output"].get("dpdp_trust_score", 50)
        if isinstance(pred_trust, (int, float)) and isinstance(gt_trust, (int, float)):
            trust_errors.append(abs(pred_trust - gt_trust))

        pred_subt = parsed.get("subtlety_score", 5) if isinstance(parsed, dict) else 5
        gt_subt = item["expected_output"].get("subtlety_score", 5)
        if isinstance(pred_subt, (int, float)) and isinstance(gt_subt, (int, float)):
            subtlety_errors.append(abs(pred_subt - gt_subt))

        halluc_metrics = calculate_evidence_hallucination_rate(pred_violations, item["content"])
        total_quotes += halluc_metrics["total_quotes"]
        total_hallucinated += halluc_metrics["hallucinated_quotes"]

        results.append({
            "case_id": item["case_id"],
            "filename": item["filename"],
            "f1": f1_metrics["f1"],
            "weighted_f1": f1_metrics["weighted_f1"],
            "trust_score_error": abs(pred_trust - gt_trust) if isinstance(pred_trust, (int, float)) else 50,
            "subtlety_score_error": abs(pred_subt - gt_subt) if isinstance(pred_subt, (int, float)) else 5,
            "hallucination_rate": halluc_metrics["hallucination_rate"],
            "latency_ms": out["latency_ms"]
        })

    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    avg_weighted_f1 = float(np.mean(weighted_f1_scores)) if weighted_f1_scores else 0.0
    mae_trust = float(np.mean(trust_errors)) if trust_errors else 0.0
    mae_subt = float(np.mean(subtlety_errors)) if subtlety_errors else 0.0
    overall_halluc_rate = (total_hallucinated / total_quotes) * 100 if total_quotes > 0 else 0.0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model_path": args.model_path,
        "total_cases_evaluated": len(test_data),
        "avg_violation_f1": round(avg_f1, 4),
        "avg_weighted_violation_f1": round(avg_weighted_f1, 4),
        "trust_score_mae": round(mae_trust, 2),
        "subtlety_score_mae": round(mae_subt, 2),
        "evidence_quote_hallucination_rate": round(overall_halluc_rate, 2),
        "details": results
    }

    report_path = REPORT_DIR / "accuracy_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "═"*70)
    print("📊 PILLARS 2, 3, & 4: ACCURACY & HALLUCINATION SUMMARY")
    print("═"*70)
    print(f"   • Total Cases Evaluated:           {len(test_data)}")
    print(f"   • Average Violation F1 Score:      {avg_f1:.4f}")
    print(f"   • Severity-Weighted F1 Score:      {avg_weighted_f1:.4f} (Threshold: >= 0.88)")
    print(f"   • Trust Score MAE:                 {mae_trust:.2f} pts (Threshold: <= 8.5 pts)")
    print(f"   • Subtlety Score MAE:              {mae_subt:.2f} pts")
    print(f"   • Evidence Quote Hallucination:    {overall_halluc_rate:.2f}% (Threshold: == 0.0%)")
    print(f"💾 Detailed report saved to: {report_path}")
    print("═"*70 + "\n")

if __name__ == "__main__":
    main()