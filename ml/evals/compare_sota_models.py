#!/usr/bin/env python3
"""
compare_sota_models.py – Pillar 4: SOTA Legal Model Comparison Suite

Executes the exact 50-query DPDP test set against both our fine-tuned conversational SLM and unadapted baseline LLMs (vanilla Qwen2.5-7B-Instruct / generic legal models) to prove our domain superiority and complete eradication of Western legal bias.

Win Condition Mandates:
- Fine-Tuned Chatbot: >90% Statute Citation Precision (SCP), >4.5/5 Context Faithfulness (CF), and 0.0% Jurisdictional Contamination Rate (JCR).
- Baseline Models: Show vulnerability to Western privacy biases (JCR > 15%, SCP < 60%).
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
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


# Fix import path for core
import sys

from backend_loader import BackendEngine, format_chatml_prompt
from metrics import evaluate_scp, evaluate_cf_judge, evaluate_jcr
# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION (paths anchored to __file__, not CWD)
# ═══════════════════════════════════════════════════════════════════════════
_EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = _EVALS_DIR / "benchmarks" / "dpdp_rag_testset.json"
FINETUNED_MODEL_PATH = Path("../models/chatbot-model-final")
BASELINE_MODEL_PATH = Path("../models/Qwen2.5-7B-Instruct")
REPORT_DIR = _EVALS_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "sota_legal_comparison_report.json"

# ═══════════════════════════════════════════════════════════════════════════
# COMPARATIVE BENCHMARK ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model_on_testset(engine: BackendEngine, queries: List[Dict[str, Any]], model_label: str) -> Dict[str, Any]:
    """Runs full SCP, CF, and JCR evaluations for a given target model engine."""
    scp_hits = 0
    cf_scores = []
    jcr_violations = 0
    latencies = []

    print(f"\n🚀 Evaluating [{model_label}] across {len(queries)} statutory scenarios...")
    for item in tqdm(queries, desc=f"Testing {model_label}"):
        query = item["query"]
        target_sec = item["target_section"]
        target_kws = item["target_keywords"]

        sys_msg = "You are a legal assistant. Answer the privacy law query accurately based on valid legal provisions."
        user_msg = f"Query: {query}"
        prompt = format_chatml_prompt(sys_msg, user_msg)
        t0 = time.perf_counter()
        out = engine.generate(prompt, max_tokens=512, temperature=0.1)
        t1 = time.perf_counter()
        
        resp = out["raw_output"]
        latencies.append((t1 - t0) * 1000.0)

        # Evaluate metrics
        if evaluate_scp(resp, target_sec):
            scp_hits += 1
        
        cf_scores.append(evaluate_cf_judge(resp, "Digital Personal Data Protection Act 2023 provisions.", target_kws, None))
        
        is_contaminated, _ = evaluate_jcr(resp)
        if is_contaminated:
            jcr_violations += 1

    total = len(queries)
    return {
        "model_label": model_label,
        "scp_rate": (scp_hits / total) * 100.0,
        "avg_cf_score": float(np.mean(cf_scores)),
        "jcr_rate": (jcr_violations / total) * 100.0,
        "avg_latency_ms": float(np.mean(latencies))
    }

def main():
    parser = argparse.ArgumentParser(description="Pillar 4: SOTA Legal Model Comparison Suite")
    parser.add_argument("--backend", type=str, default="unsloth", choices=["unsloth", "vllm", "llamacpp"])
    parser.add_argument("--finetuned-path", type=str, default=str(FINETUNED_MODEL_PATH))
    parser.add_argument("--baseline-path", type=str, default=str(BASELINE_MODEL_PATH))
    parser.add_argument("--benchmark-path", type=str, default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--lora-name", type=str, default="chatbot")
    parser.add_argument("--allow-simulated-baseline", action="store_true",
                        help="Allow using hardcoded simulated baseline metrics when baseline model is unavailable. "
                             "Without this flag, the script will HARD-FAIL if the baseline model is not found.")
    args = parser.parse_args()

    print("🏆 [PILLAR 4]: SOTA Legal Model Head-to-Head Comparative Benchmark")
    bench_path = Path(args.benchmark_path)
    if not bench_path.exists():
        print(f"❌ Error: Benchmark dataset not found at {bench_path}")
        return 1

    with open(bench_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # 1. Evaluate Fine-Tuned Chatbot Model
    print(f"📦 Loading Fine-Tuned Model Engine from: {args.finetuned_path}...")
    ft_engine = BackendEngine(backend_type=args.backend, model_path=args.finetuned_path, lora_name=args.lora_name)
    ft_results = evaluate_model_on_testset(ft_engine, queries, "RAFT-Trained DPDP Chatbot (9B)")

    # 2. Evaluate Base Model / Baseline
    baseline_is_simulated = False
    base_results = None

    if Path(args.baseline_path).exists() and not (args.finetuned_path == args.baseline_path):
        try:
            print(f"📦 Loading Vanilla Baseline Model Engine from: {args.baseline_path}...")
            base_engine = BackendEngine(backend_type=args.backend, model_path=args.baseline_path)
            base_results = evaluate_model_on_testset(base_engine, queries, "Vanilla Baseline")
        except Exception as e:
            print(f"⚠️ Baseline model failed to load: {e}")

    if base_results is None:
        if args.allow_simulated_baseline:
            print("⚠️ WARNING: Baseline model not available. Using SIMULATED baseline metrics.")
            print("   This means the comparative superiority claim is NOT empirically validated.")
            baseline_is_simulated = True
            base_results = {
                "model_label": "[SIMULATED] Vanilla Baseline (NOT REAL)",
                "scp_rate": 56.4,
                "avg_cf_score": 3.2,
                "jcr_rate": 18.0,
                "avg_latency_ms": ft_results["avg_latency_ms"] * 0.95
            }
        else:
            print("❌ HARD-FAIL: Baseline model not found and --allow-simulated-baseline not set.")
            print(f"   Looked for baseline at: {args.baseline_path}")
            print("   Either provide the baseline model or pass --allow-simulated-baseline to use fake numbers.")
            return 1

    # Win Condition Assertions
    win_scp = ft_results["scp_rate"] >= 90.0 and ft_results["scp_rate"] > base_results["scp_rate"]
    win_cf = ft_results["avg_cf_score"] >= 4.5 and ft_results["avg_cf_score"] > base_results["avg_cf_score"]
    win_jcr = ft_results["jcr_rate"] == 0.0 and base_results["jcr_rate"] > 0.0
    certified_superiority = win_scp and win_cf and win_jcr

    print("\n═══════════════════════════════════════════════════════════════════════")
    print("🏆 PILLAR 4 SOTA LEGAL LLM COMPARISON MATRIX (50-QUERY DPDP BENCHMARK)")
    print("═══════════════════════════════════════════════════════════════════════")
    print(f"| Evaluation Metric               | {base_results['model_label']:<24} | {ft_results['model_label']:<26} | Win Condition | Status |")
    print(f"|---------------------------------|--------------------------|----------------------------|---------------|--------|")
    print(f"| Statute Citation Precision (SCP)| {base_results['scp_rate']:22.2f}% | {ft_results['scp_rate']:24.2f}% | > 90.0% (FT)  | {'✅ PASS' if win_scp else '⚠️ NOTICE'} |")
    print(f"| Context Faithfulness (CF Score) | {base_results['avg_cf_score']:22.2f}/5 | {ft_results['avg_cf_score']:24.2f}/5 | > 4.50 / 5    | {'✅ PASS' if win_cf else '⚠️ NOTICE'} |")
    print(f"| Jurisdictional Contamination    | {base_results['jcr_rate']:22.2f}% | {ft_results['jcr_rate']:24.2f}% | 0.00% (FT)    | {'✅ PASS' if win_jcr else '⚠️ NOTICE'} |")
    print("═══════════════════════════════════════════════════════════════════════\n")
    print(f"🏁 Win Condition Verdict: {'✅ CERTIFIED SOTA SUPERIORITY' if certified_superiority else '❌ NOT CERTIFIED — WIN CONDITIONS NOT MET'}")

    report_dict = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "testset_queries": len(queries),
        "finetuned_model": ft_results,
        "baseline_model": base_results,
        "baseline_is_simulated": baseline_is_simulated,
        "win_condition_verified": certified_superiority
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"💾 Pillar 4 SOTA comparison report saved to: {REPORT_PATH}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
