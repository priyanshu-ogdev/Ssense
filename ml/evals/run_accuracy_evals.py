#!/usr/bin/env python3
"""
run_accuracy_evals.py – Legal Reasoning Accuracy & Hallucination Evaluation (Universal Dual-Backend Grade)

Measures Pillars 2, 3, and 4 (Diagnostic):
1. Violation F1 Score (Severity-Weighted & Sub-clause Section Normalized)
2. DPDP Trust Score & Subtlety Score MAE
3. Evidence Quote Hallucination Rate (Verbatim Substring Validation)
4. Parametric Statutory Citation Validity Rate
5. Sector/Category Breakdown & Telemetry (Latency, TTFT, Throughput)

SOTA Upgrades Implemented:
1. Production Guided Decoding: Injects `grammar=json.dumps(schema)` for vLLM structured decoding.
2. Token Window Balancing: Balanced 32k context envelope with 4096-token generation headroom.
3. Strict VRAM Airlock: Unloads model via `engine.unload()` and purges CUDA allocator caches.
4. Diagnostic Exit Codes: Concludes with return 0 for clean aggregation by `verify.py`.
5. Indestructible Paths: Utilizes `path_resolver.py` for absolute working directory independence.
"""

import os
import sys
import gc
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone
from tqdm import tqdm
import numpy as np
import torch

# Ensure terminal stdout/stderr uses UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Dynamic path resolution
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

try:
    from path_resolver import Paths
    DEFAULT_GROUND_TRUTH_PATH = Paths.GROUND_TRUTH
    DEFAULT_LAW_FILE_PATH = Paths.LAW_TEXT
    DEFAULT_MODEL_PATH = Paths.resolve_model_path(None, "audit-model-final")
    DEFAULT_SCHEMA_PATH = Paths.SCHEMA_PATH
    REPORT_DIR = Paths.ensure_reports_dir()
except ImportError:
    _ML_DIR = _CURRENT_DIR.parent
    DEFAULT_GROUND_TRUTH_PATH = _CURRENT_DIR / "holdout_policies" / "ground_truth.json"
    DEFAULT_LAW_FILE_PATH = _ML_DIR / "data-forge" / "dpdp_act_and_rules_2025.txt"
    DEFAULT_MODEL_PATH = _ML_DIR / "models" / "audit-model-final"
    DEFAULT_SCHEMA_PATH = _ML_DIR.parent / "libs" / "contracts" / "schemas" / "dpdp_schema.json"
    REPORT_DIR = _CURRENT_DIR / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

from backend_loader import BackendEngine, format_chatml_prompt
from metrics import (
    extract_json_from_output,
    calculate_violation_f1,
    calculate_evidence_hallucination_rate,
    calculate_parametric_citation_validity
)

try:
    from stats import wilson_ci_from_pct
except ImportError:
    def wilson_ci_from_pct(p: float, n: int):
        return (p, p)


def flush_gpu():
    """Forces garbage collection and clears CUDA allocator caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_schema(schema_path: Path) -> Dict[str, Any]:
    """Loads the schema for Guided Decoding injection."""
    if not schema_path.exists():
        return {}
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════
def load_test_data(gt_path: Path) -> List[Dict[str, Any]]:
    """Loads holdout evaluation policies and expected ground truth structures."""
    if not gt_path.exists():
        print(f"❌ Error: Ground truth dataset not found at {gt_path}")
        return []

    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    test_data = []
    for item in ground_truth:
        content = ""
        if 'policy_text_snippet' in item and item['policy_text_snippet']:
            content = item['policy_text_snippet']
        elif 'filename' in item:
            policy_file = gt_path.parent / item['filename']
            if policy_file.exists():
                with open(policy_file, 'r', encoding='utf-8') as pf:
                    content = pf.read()

        if not content.strip():
            continue

        test_data.append({
            "case_id": item.get('case_id', item.get('filename', 'unknown')),
            "filename": item.get('filename', 'embedded_snippet'),
            "category": item.get('category', 'General Commercial'),
            "description": item.get('description', ''),
            "content": content.strip(),
            "expected_output": item.get('expected_output', {}),
            "evaluation_targets": item.get('evaluation_targets', {})
        })
    return test_data


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ACCURACY EVALUATION HARNESS
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Pillars 2, 3, & 4: Legal Reasoning Accuracy & Hallucination Benchmark")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "unsloth", "llamacpp"])
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--ground-truth-path", type=str, default=str(DEFAULT_GROUND_TRUTH_PATH))
    parser.add_argument("--law-path", type=str, default=str(DEFAULT_LAW_FILE_PATH))
    parser.add_argument("--schema-path", type=str, default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--lora-name", type=str, default="audit")
    parser.add_argument("--inject-law-context", action="store_true", 
                        help="Inject raw statutory text in-context (default: False, tests parametric SFT memory)")
    args = parser.parse_args()

    test_data = load_test_data(Path(args.ground_truth_path))
    if not test_data:
        print("❌ Evaluation aborted: No valid test cases loaded.")
        return 1
        
    dpdp_schema = load_schema(Path(args.schema_path))
    grammar_payload = json.dumps(dpdp_schema) if dpdp_schema else None

    print("═══════════════════════════════════════════════════════════════════════")
    print(f"🚀 [PILLARS 2, 3, 4]: LEGAL REASONING, ACCURACY & HALLUCINATION ({args.backend.upper()})")
    print("═══════════════════════════════════════════════════════════════════════")
    
    # SOTA FIX: Push max_seq_length to 32k for production parity
    engine = BackendEngine(
        backend_type=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        vllm_url=args.vllm_url,
        lora_name=args.lora_name,
        max_seq_length=32768
    )

    law_context = ""
    if args.inject_law_context and Path(args.law_path).exists():
        with open(args.law_path, 'r', encoding='utf-8') as f:
            law_context = f.read()[:6000]

    results = []
    f1_scores = []
    weighted_f1_scores = []
    precision_scores = []
    recall_scores = []
    trust_errors = []
    subtlety_errors = []
    
    total_quotes = 0
    total_hallucinated_quotes = 0
    total_citations = 0
    valid_citations = 0

    category_breakdown: Dict[str, Dict[str, Any]] = {}

    try:
        for item in tqdm(test_data, desc="Auditing Policies"):
            sys_msg = (
                "You are an expert DPDP Act 2023 forensic legal auditor. "
                "Analyze the provided corporate privacy policy for statutory violations under the "
                "Digital Personal Data Protection Act 2023 and DPDP Rules 2025. "
                "Output ONLY a valid JSON object strictly matching the schema contract."
            )
            
            if args.inject_law_context and law_context:
                user_msg = f"[CONTEXT: STATUTORY PROVISIONS]\n{law_context}\n\n[POLICY TO AUDIT]\n{item['content']}"
            else:
                user_msg = f"[POLICY TO AUDIT]\n{item['content']}"

            prompt = format_chatml_prompt(sys_msg, user_msg)
            
            # SOTA FIX: max_tokens set to 4096; schema grammar injected for vLLM structured decoding
            out = engine.generate(
                prompt, 
                max_tokens=4096, 
                temperature=0.0,
                grammar=grammar_payload
            )
            
            extracted = extract_json_from_output(out.get("raw_output", ""))
            parsed = {}
            json_valid = False
            try:
                if extracted:
                    parsed = json.loads(extracted)
                    json_valid = isinstance(parsed, dict)
            except Exception:
                json_valid = False

            pred_violations = parsed.get("violations", []) if json_valid else []
            gt_violations = item["expected_output"].get("violations", [])

            # 1. Violation F1 Metrics
            f1_res = calculate_violation_f1(pred_violations, gt_violations)
            f1_scores.append(f1_res["f1"])
            weighted_f1_scores.append(f1_res["weighted_f1"])
            precision_scores.append(f1_res["precision"])
            recall_scores.append(f1_res["recall"])

            # 2. Score MAE Metrics
            pred_trust = parsed.get("dpdp_trust_score", None) if json_valid else None
            gt_trust = item["expected_output"].get("dpdp_trust_score", 50)
            if isinstance(pred_trust, (int, float)) and isinstance(gt_trust, (int, float)):
                t_err = abs(pred_trust - gt_trust)
            else:
                t_err = 50.0  # Schema penalty
            trust_errors.append(t_err)

            pred_subt = parsed.get("subtlety_score", None) if json_valid else None
            gt_subt = item["expected_output"].get("subtlety_score", 5)
            if isinstance(pred_subt, (int, float)) and isinstance(gt_subt, (int, float)):
                s_err = abs(pred_subt - gt_subt)
            else:
                s_err = 5.0
            subtlety_errors.append(s_err)

            # 3. Evidence Quote Hallucination Check
            halluc_metrics = calculate_evidence_hallucination_rate(pred_violations, item["content"])
            total_quotes += halluc_metrics["total_quotes"]
            total_hallucinated_quotes += halluc_metrics["hallucinated_quotes"]

            # 4. Parametric Citation Validity Check
            cit_metrics = calculate_parametric_citation_validity(pred_violations)
            total_citations += cit_metrics["total_citations"]
            valid_citations += cit_metrics["valid_citations"]

            # Track Sectoral/Category Performance
            cat = item["category"]
            if cat not in category_breakdown:
                category_breakdown[cat] = {"count": 0, "f1_sum": 0.0, "trust_err_sum": 0.0, "halluc_quotes": 0, "quotes": 0}
            category_breakdown[cat]["count"] += 1
            category_breakdown[cat]["f1_sum"] += f1_res["weighted_f1"]
            category_breakdown[cat]["trust_err_sum"] += t_err
            category_breakdown[cat]["halluc_quotes"] += halluc_metrics["hallucinated_quotes"]
            category_breakdown[cat]["quotes"] += halluc_metrics["total_quotes"]

            results.append({
                "case_id": item["case_id"],
                "category": cat,
                "json_valid": json_valid,
                "f1": f1_res["f1"],
                "weighted_f1": f1_res["weighted_f1"],
                "precision": f1_res["precision"],
                "recall": f1_res["recall"],
                "predicted_trust_score": pred_trust,
                "expected_trust_score": gt_trust,
                "trust_score_error": t_err,
                "subtlety_score_error": s_err,
                "hallucinated_quotes": halluc_metrics["hallucinated_quotes"],
                "total_quotes": halluc_metrics["total_quotes"],
                "parametric_validity_rate": cit_metrics["validity_rate"],
                "latency_ms": out.get("latency_ms", 0.0),
                "ttft_ms": out.get("ttft_ms", 0.0),
                "tokens_generated": out.get("tokens_generated", 0)
            })

    finally:
        # Strict VRAM Airlock: Clean up engine from GPU memory
        engine.unload()
        del engine
        flush_gpu()
        print("\n🧹 [VRAM Airlock] Auditor model purged from GPU memory.")

    # ═══════════════════════════════════════════════════════════════════
    # AGGREGATE METRICS & CONFIDENCE INTERVALS
    # ═══════════════════════════════════════════════════════════════════
    total_cases = len(test_data)
    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    avg_weighted_f1 = float(np.mean(weighted_f1_scores)) if weighted_f1_scores else 0.0
    avg_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    
    mae_trust = float(np.mean(trust_errors)) if trust_errors else 0.0
    mae_subt = float(np.mean(subtlety_errors)) if subtlety_errors else 0.0
    
    overall_halluc_rate = (total_hallucinated_quotes / total_quotes * 100.0) if total_quotes > 0 else 0.0
    halluc_low, halluc_high = wilson_ci_from_pct(overall_halluc_rate, total_quotes) if total_quotes > 0 else (0.0, 0.0)
    
    overall_cit_validity = (valid_citations / total_citations * 100.0) if total_citations > 0 else 100.0
    cit_low, cit_high = wilson_ci_from_pct(overall_cit_validity, total_citations) if total_citations > 0 else (100.0, 100.0)

    # Sector summary with division guards
    sector_summary = {}
    for cat, d in category_breakdown.items():
        cnt = max(1, d["count"])
        q_tot = d["quotes"]
        h_rate = (d["halluc_quotes"] / q_tot * 100.0) if q_tot > 0 else 0.0
        sector_summary[cat] = {
            "cases_evaluated": d["count"],
            "mean_weighted_f1": round(d["f1_sum"] / cnt, 4),
            "trust_mae": round(d["trust_err_sum"] / cnt, 2),
            "hallucination_rate": round(h_rate, 2)
        }

    # Master summary dictionary structured exactly for verify.py extraction
    summary_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model_path": str(args.model_path),
        "total_cases_evaluated": total_cases,
        "total_quotes": total_quotes,
        "total_citations": total_citations,
        
        # Primary Extraction Keys for verify.py
        "avg_violation_f1": round(avg_f1, 4),
        "avg_weighted_violation_f1": round(avg_weighted_f1, 4),
        "macro_precision": round(avg_precision, 4),
        "macro_recall": round(avg_recall, 4),
        "trust_score_mae": round(mae_trust, 2),
        "subtlety_score_mae": round(mae_subt, 2),
        "evidence_quote_hallucination_rate": round(overall_halluc_rate, 2),
        "evidence_quote_hallucination_wilson_ci": [round(halluc_low, 2), round(halluc_high, 2)],
        "parametric_citation_validity_rate": round(overall_cit_validity, 2),
        "parametric_citation_validity_wilson_ci": [round(cit_low, 2), round(cit_high, 2)],
        
        # Breakdown & Detailed Cases
        "sector_breakdown": sector_summary,
        "details": results
    }

    report_path = REPORT_DIR / "accuracy_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════
    # TERMINAL SCORECARD
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "═"*75)
    print("📊 PILLARS 2, 3, & 4: ACCURACY, REASONING & HALLUCINATION SUMMARY")
    print("═"*75)
    print(f"  • Total Test Policies:             {total_cases}")
    print(f"  • Severity-Weighted Violation F1:  {avg_weighted_f1:.4f}  (Target: >= 0.8800) -> {'✅ PASS' if avg_weighted_f1 >= 0.88 else '❌ FAIL'}")
    print(f"  • Macro Precision / Recall:        {avg_precision:.4f} / {avg_recall:.4f}")
    print(f"  • DPDP Trust Score MAE:            {mae_trust:.2f} pts (Target: <= 8.50 pts) -> {'✅ PASS' if mae_trust <= 8.5 else '❌ FAIL'}")
    print(f"  • Subtlety Score MAE:              {mae_subt:.2f} pts")
    print(f"  • Evidence Quote Hallucination:    {overall_halluc_rate:.2f}% (Wilson Upper: {halluc_high:.2f}%) -> {'✅ PASS' if overall_halluc_rate == 0.0 else '❌ FAIL'}")
    print(f"  • Parametric Citation Validity:    {overall_cit_validity:.2f}% (Wilson Lower: {cit_low:.2f}%) -> {'✅ PASS' if overall_cit_validity >= 95.0 else '❌ FAIL'}")
    
    print("\n📈 Sectoral Performance Breakdown:")
    for cat, sdata in sector_summary.items():
        print(f"    - {cat:<22}: Weighted F1 = {sdata['mean_weighted_f1']:.4f} | Trust MAE = {sdata['trust_mae']:.1f} pts | Halluc = {sdata['hallucination_rate']:.1f}%")

    print(f"\n💾 Detailed report saved to: {report_path}")
    print("═"*75 + "\n")

    # Diagnostic return code: Always returns 0 so verify.py handles threshold grading
    return 0


if __name__ == "__main__":
    sys.exit(main())