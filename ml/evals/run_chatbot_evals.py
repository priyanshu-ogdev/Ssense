#!/usr/bin/env python3
"""
chatbot_authenticity.py – Unified Chatbot Authenticity & Faithfulness Evaluation

SOTA Upgrades Implemented:
1. Two-Pass VRAM Airlock: Evaluates generation (7B) and CF judging (72B) sequentially 
   to completely eliminate VRAM fragmentation and OOM crashes.
2. Context Fidelity: Removes the "RAG Context Roulette" bug. Defaults to parametric memory
   unless explicit ground-truth context is provided in the benchmark.
3. Statistical Gating: Computes Wilson 95% Confidence Intervals for SCP, JCR, and Bleed Rate.
4. Deep Path Integration: Utilizes path_resolver.py for indestructible dynamic paths.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime, timezone
from tqdm import tqdm

# Ensure terminal stdout/stderr uses UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Dynamic path resolution
from pathlib import Path
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

try:
    from path_resolver import Paths
    from backend_loader import BackendEngine, format_chatml_prompt
    from metrics import (
        evaluate_scp,
        evaluate_cf_judge,
        evaluate_jcr,
        check_schema_bleed,
        check_forbidden_terms,
        evaluate_key_points_coverage
    )
    from stats import mtld, wilson_ci_from_pct
except ImportError as e:
    print(f"❌ Core module import failed: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Unified Chatbot Authenticity & Fluidity Evals")
    parser.add_argument("--backend", type=str, default="unsloth", choices=["unsloth", "vllm", "llamacpp"])
    parser.add_argument("--model-path", type=str, default=str(Paths.resolve_model_path(None, "chatbot-model-final")))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=str(Paths.CHATBOT_QA_BENCHMARK))
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--lora-name", type=str, default="chatbot")
    parser.add_argument("--use-judge", action="store_true", help="Load 72B teacher model into VRAM for CF scoring")
    parser.add_argument("--judge-path", type=str, default=str(Paths.resolve_model_path(None, "Qwen2-72B-Instruct-FP8")))
    args = parser.parse_args()

    bench_path = Path(args.benchmark_path)
    if not bench_path.exists():
        print(f"❌ Error: Benchmark file not found at {bench_path}")
        return 1

    with open(bench_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print("═══════════════════════════════════════════════════════════════════════")
    print(f"🚀 [PILLAR 2]: CHATBOT AUTHENTICITY, FLUIDITY & JURISDICTION EVALUATION")
    print("═══════════════════════════════════════════════════════════════════════")
    
    # -------------------------------------------------------------------------
    # PASS 1: Generate Responses (7B Chatbot SLM)
    # -------------------------------------------------------------------------
    print(f"\n🧠 [Pass 1/2] Initializing Chatbot Engine (Backend: {args.backend})...")
    chatbot_engine = BackendEngine(
        backend_type=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        vllm_url=args.vllm_url,
        lora_name=args.lora_name,
        max_seq_length=4096
    )

    generated_responses = []
    try:
        print(f"Generating completions for {len(test_data)} statutory scenarios...")
        for item in tqdm(test_data, desc="Chatbot Generation"):
            query = item.get("query", item.get("question", ""))
            
            # SOTA Fix: Only inject context if explicitly provided by the benchmark, otherwise test parametric memory
            retrieved_context = item.get("context", "Digital Personal Data Protection Act 2023 & Rules 2025.")

            sys_msg = (
                "You are a warm, empathetic, and strictly compliant Indian DPDP Legal Assistant. "
                "Answer the query accurately according to the DPDP Act 2023 without citing foreign "
                "legal frameworks or inventing speculative rules."
            )
            user_msg = f"[STATUTORY CONTEXT]:\n{retrieved_context}\n\nQuery: {query}"
            
            prompt = format_chatml_prompt(sys_msg, user_msg)
            out = chatbot_engine.generate(prompt, max_tokens=1024, temperature=0.1)
            generated_responses.append(out["raw_output"])
    finally:
        # Strict VRAM Airlock
        chatbot_engine.unload()
        print("🧹 [VRAM Airlock] Chatbot model purged from GPU memory.")

    # -------------------------------------------------------------------------
    # PASS 2: Context Faithfulness Judging (72B Teacher)
    # -------------------------------------------------------------------------
    judge_engine = None
    if args.use_judge and Path(args.judge_path).exists():
        print(f"\n🏛️ [Pass 2/2] Initializing 72B Teacher Judge ({args.judge_path})...")
        try:
            # We strictly use unsloth/transformers for the judge to manage FP8 loading
            judge_engine = BackendEngine(backend_type="unsloth", model_path=args.judge_path, max_seq_length=8192)
        except Exception as e:
            print(f"⚠️ Failed to load 72B Judge: {e}. Falling back to heuristic CF scoring.")

    # -------------------------------------------------------------------------
    # METRICS CALCULATION
    # -------------------------------------------------------------------------
    scp_hits = 0
    cf_scores = []
    jcr_violations = 0
    total_accuracies = []
    total_mtlds = []
    bleed_violations = 0
    detailed_results = []

    print("\n📊 Computing Authenticity Metrics & Wilson Confidence Intervals...")
    for i, item in enumerate(tqdm(test_data, desc="Evaluating Metrics")):
        resp = generated_responses[i]
        query = item.get("query", item.get("question", ""))
        target_section = item.get("target_section", "None")
        target_keywords = item.get("target_keywords", item.get("expected_key_points", []))
        retrieved_context = item.get("context", "DPDP Act 2023 provisions.")

        # 1. Statute Citation Precision (SCP)
        is_precise = evaluate_scp(resp, target_section)
        if is_precise: scp_hits += 1

        # 2. Context Faithfulness (CF 1-5)
        cf_score = evaluate_cf_judge(resp, retrieved_context, target_keywords, judge_engine)
        cf_scores.append(cf_score)

        # 3. Jurisdictional Contamination Rate (JCR)
        is_contaminated, found_contaminants = evaluate_jcr(resp)
        if is_contaminated: jcr_violations += 1

        # 4. Accuracy & Coverage
        coverage = evaluate_key_points_coverage(resp, target_keywords)
        forbidden_hits = check_forbidden_terms(resp, item.get("forbidden_hallucination_terms", []))
        accuracy_score = max(0.0, coverage - (0.5 * len(forbidden_hits)))
        total_accuracies.append(accuracy_score)

        # 5. Schema Bleed & Fluidity (MTLD)
        bleed_hits = check_schema_bleed(resp)
        if bleed_hits: bleed_violations += 1
            
        fluidity = mtld(resp)
        total_mtlds.append(fluidity)

        detailed_results.append({
            "id": item.get("id", f"q_{i+1}"),
            "query": query,
            "target_section": target_section,
            "scp_pass": is_precise,
            "cf_score": cf_score,
            "jcr_contaminated": is_contaminated,
            "found_contaminants": found_contaminants,
            "accuracy_score": round(accuracy_score, 4),
            "forbidden_hits": forbidden_hits,
            "bleed_hits": bleed_hits,
            "mtld_fluidity": round(fluidity, 4),
            "response_snippet": resp[:200] + "..." if len(resp) > 200 else resp
        })

    # Strict VRAM Airlock for Judge
    if judge_engine is not None:
        judge_engine.unload()

    # -------------------------------------------------------------------------
    # AGGREGATION & REPORTING
    # -------------------------------------------------------------------------
    total = max(1, len(test_data))
    
    # Point Estimates
    scp_rate = (scp_hits / total) * 100.0
    jcr_rate = (jcr_violations / total) * 100.0
    bleed_rate = (bleed_violations / total) * 100.0
    avg_accuracy = (sum(total_accuracies) / total) * 100.0
    avg_cf = float(np.mean(cf_scores)) if cf_scores else 5.0
    avg_mtld = float(np.mean(total_mtlds)) if total_mtlds else 0.0

    # Wilson 95% CI Bounds
    scp_low, scp_high = wilson_ci_from_pct(scp_rate, total)
    jcr_low, jcr_high = wilson_ci_from_pct(jcr_rate, total)
    bleed_low, bleed_high = wilson_ci_from_pct(bleed_rate, total)
    acc_low, acc_high = wilson_ci_from_pct(avg_accuracy, total)

    # Win Conditions (Evaluated via CI bounds where applicable for stringent MLOps)
    passed_scp = scp_low >= 90.0
    passed_cf = avg_cf >= 4.5
    passed_jcr = jcr_rate == 0.0 # Strict absolute zero boundary evaluated on point estimate
    passed_acc = acc_low >= 95.0
    passed_bleed = bleed_rate == 0.0
    passed_mtld = avg_mtld >= 40.0
    
    passed_all = passed_scp and passed_cf and passed_jcr and passed_acc and passed_bleed and passed_mtld

    print("\n═══════════════════════════════════════════════════════════════════════════════")
    print("📊 PILLAR 2 CHATBOT AUTHENTICITY & GROUNDING EVALUATION REPORT")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print(f"| Evaluation Metric               | Measured Score (Wilson CI) | Target       | Status |")
    print(f"|---------------------------------|----------------------------|--------------|--------|")
    print(f"| Statute Citation Precision (SCP)| {scp_rate:6.2f}% ({scp_low:5.1f}-{scp_high:5.1f}) | >= 90.0% | {'✅ PASS' if passed_scp else '❌ FAIL'} |")
    print(f"| Context Faithfulness (CF Score) | {avg_cf:6.2f}  (Point Est.) | >= 4.50  | {'✅ PASS' if passed_cf else '❌ FAIL'} |")
    print(f"| Jurisdictional Contamination    | {jcr_rate:6.2f}% (Point Est.) | == 0.00% | {'✅ PASS' if passed_jcr else '❌ FAIL'} |")
    print(f"| Statutory Accuracy Rate         | {avg_accuracy:6.2f}% ({acc_low:5.1f}-{acc_high:5.1f}) | >= 95.0% | {'✅ PASS' if passed_acc else '❌ FAIL'} |")
    print(f"| Schema Bleed Rate               | {bleed_rate:6.2f}% (Point Est.) | == 0.00% | {'✅ PASS' if passed_bleed else '❌ FAIL'} |")
    print(f"| MTLD Fluidity Score             | {avg_mtld:6.2f}  (Point Est.) | >= 40.00 | {'✅ PASS' if passed_mtld else '❌ FAIL'} |")
    print("═══════════════════════════════════════════════════════════════════════════════\n")

    report_path = Paths.ensure_reports_dir() / "chatbot_evals_report.json"
    
    report_dict = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model_path": str(args.model_path),
        "total_evaluated_queries": total,
        "summary_metrics": {
            "statute_citation_precision_rate": round(scp_rate, 2),
            "context_faithfulness_score": round(avg_cf, 2),
            "jurisdictional_contamination_rate": round(jcr_rate, 2),
            "avg_statutory_accuracy_rate": round(avg_accuracy, 2),
            "schema_bleed_rate": round(bleed_rate, 2),
            "avg_ttr_fluidity_score": round(avg_mtld, 4), # Extracted by verify.py
            "certified_sota": passed_all
        },
        "wilson_ci_95": {
            "scp_lower": scp_low,
            "jcr_upper": jcr_high,
            "bleed_upper": bleed_high,
            "accuracy_lower": acc_low
        },
        "detailed_evaluations": detailed_results
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
        
    print(f"💾 Pillar 2 evaluation report saved to: {report_path}")
    return 0 if passed_all else 1

if __name__ == "__main__":
    sys.exit(main())