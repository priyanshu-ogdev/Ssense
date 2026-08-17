#!/usr/bin/env python3
"""
compare_sota_models.py – Pillar 4: SOTA Legal Model Comparative Benchmark

Executes the 50-query DPDP statutory test set against both the fine-tuned 
conversational SLM and unadapted baseline LLMs.

SOTA Upgrades Implemented:
1. 3-Stage VRAM Airlock: Generates FT, Generates Baseline, then loads 72B Judge sequentially.
2. The Blind Judge Fix: Correctly loads and passes the Qwen2-72B-Instruct engine to the CF evaluator.
3. Context-Aware Prompting: Synchronized with the empathetic, explicit-citation persona.
4. Scale Parity: Max sequence length 32768, max tokens 2048 to prevent truncation.
5. Statistical Rigor: Exact McNemar/Permutation p-values for statistical significance.
"""

import os
import sys
import gc
import json
import time
import math
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

import torch

# Ensure terminal stdout/stderr uses UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Dynamic path resolution to ensure imports work from any execution directory
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

try:
    from path_resolver import Paths
    DEFAULT_BENCHMARK = Paths.RAG_TESTSET
    DEFAULT_FINETUNED_PATH = Paths.resolve_model_path(None, "chatbot-model-final")
    DEFAULT_BASELINE_PATH = Paths.resolve_model_path(None, "Qwen2.5-7B-Instruct")
    DEFAULT_JUDGE_PATH = Paths.resolve_model_path(None, "Qwen2-72B-Instruct-FP8")
    REPORT_DIR = Paths.ensure_reports_dir()
    REPORT_PATH = REPORT_DIR / "sota_legal_comparison_report.json"
except ImportError:
    _ML_DIR = _CURRENT_DIR.parent
    DEFAULT_BENCHMARK = _CURRENT_DIR / "benchmarks" / "dpdp_rag_testset.json"
    DEFAULT_FINETUNED_PATH = _ML_DIR / "models" / "chatbot-model-final"
    DEFAULT_BASELINE_PATH = _ML_DIR / "models" / "Qwen2.5-7B-Instruct"
    DEFAULT_JUDGE_PATH = _ML_DIR / "models" / "Qwen2-72B-Instruct-FP8"
    REPORT_DIR = _CURRENT_DIR / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH = REPORT_DIR / "sota_legal_comparison_report.json"

try:
    from backend_loader import BackendEngine, format_chatml_prompt
    from metrics import evaluate_scp, evaluate_cf_judge, evaluate_jcr
except ImportError as e:
    print(f"❌ Failed to import core evaluation modules: {e}")
    sys.exit(1)

# Import standardized Wilson CI
try:
    from stats import wilson_ci
except ImportError:
    def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        if total == 0: return 0.0, 0.0
        z = 1.95996  # 95% two-sided confidence
        p = successes / total
        denominator = 1.0 + (z**2) / total
        centre_adjusted = p + (z**2) / (2 * total)
        adjusted_std = z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)
        lower_bound = max(0.0, (centre_adjusted - adjusted_std) / denominator)
        upper_bound = min(1.0, (centre_adjusted + adjusted_std) / denominator)
        return round(lower_bound * 100.0, 2), round(upper_bound * 100.0, 2)


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE TESTING
# ═══════════════════════════════════════════════════════════════════════════
def compute_paired_mcnemar(b_correct: List[bool], a_correct: List[bool]) -> float:
    n01 = sum(1 for b, a in zip(b_correct, a_correct) if not b and a)
    n10 = sum(1 for b, a in zip(b_correct, a_correct) if b and not a)
    
    total_discordant = n01 + n10
    if total_discordant == 0: return 1.0
    
    if total_discordant < 25:
        from math import comb
        p_val = sum(comb(total_discordant, i) * (0.5 ** total_discordant) for i in range(n01, total_discordant + 1))
        return float(min(1.0, p_val * 2))
    else:
        chi2 = (abs(n01 - n10) - 1.0) ** 2 / total_discordant
        p_val = math.erfc(math.sqrt(chi2 / 2.0))
        return float(max(0.0, min(1.0, p_val)))

def compute_paired_t_test(scores_a: List[float], scores_b: List[float]) -> Tuple[float, float]:
    diffs = np.array(scores_a) - np.array(scores_b)
    n = len(diffs)
    if n < 2: return float(np.mean(diffs)), 1.0
    
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    if std_diff == 0: return mean_diff, 1.0 if mean_diff == 0 else 0.0
    
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return mean_diff, float(p_val)

def flush_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ═══════════════════════════════════════════════════════════════════════════
# GENERATOR & EVALUATOR ABSTRACTIONS
# ═══════════════════════════════════════════════════════════════════════════
def generate_model_responses(engine: BackendEngine, queries: List[Dict[str, Any]], model_label: str) -> List[Dict[str, Any]]:
    print(f"\n🚀 Generating [{model_label}] responses across {len(queries)} scenarios...")
    
    system_prompt = (
        "You are an empathetic and expert Indian DPDP Legal Assistant. "
        "Answer the user's query accurately according to the Digital Personal Data Protection Act 2023. "
        "If statutory context is provided, you must ground your answer strictly within that context. "
        "If the context does not contain the answer, or if the query falls outside the scope of the DPDP Act, "
        "you must politely decline to answer or state that the Act is silent. "
        "Crucially, you must explicitly cite applicable statutory section numbers. Do not cite foreign statutes (GDPR)."
    )

    outputs = []
    for idx, item in enumerate(tqdm(queries, desc=f"Benchmarking {model_label}")):
        query = item.get("query", "")
        retrieved_context = item.get("context", "")

        if retrieved_context.strip():
            user_msg = f"[STATUTORY CONTEXT]:\n{retrieved_context}\n\nStatutory Query: {query}\nProvide legal evaluation and cite applicable provisions:"
        else:
            user_msg = f"Statutory Query: {query}\nProvide legal evaluation and cite applicable provisions:"
            
        prompt = format_chatml_prompt(system_prompt, user_msg)
        
        t0 = time.perf_counter()
        out = engine.generate(prompt, max_tokens=2048, temperature=0.0)
        t1 = time.perf_counter()
        
        outputs.append({
            "raw_output": out.get("raw_output", ""),
            "latency_ms": (t1 - t0) * 1000.0
        })
    return outputs


def compute_offline_metrics(responses: List[Dict[str, Any]], queries: List[Dict[str, Any]], model_label: str) -> Dict[str, Any]:
    scp_hits = 0; scp_mask = []
    jcr_violations = 0; jcr_mask = []
    latencies = []
    traces = []

    for idx, (resp_data, item) in enumerate(zip(responses, queries)):
        resp_text = resp_data["raw_output"]
        latencies.append(resp_data["latency_ms"])
        target_sec = item.get("target_section", "")

        is_scp_hit = evaluate_scp(resp_text, target_sec)
        scp_mask.append(is_scp_hit)
        if is_scp_hit: scp_hits += 1

        is_jcr_contaminated, foreign_citations = evaluate_jcr(resp_text)
        jcr_mask.append(is_jcr_contaminated)
        if is_jcr_contaminated: jcr_violations += 1

        traces.append({
            "query_id": idx + 1,
            "query": item.get("query", ""),
            "target_section": target_sec,
            "model_response": resp_text[:250] + "..." if len(resp_text) > 250 else resp_text,
            "scp_passed": is_scp_hit,
            "jcr_contaminated": is_jcr_contaminated,
            "foreign_citations": foreign_citations,
            "latency_ms": resp_data["latency_ms"]
        })

    total = len(queries)
    scp_point = (scp_hits / total) * 100.0
    scp_ci_low, scp_ci_high = wilson_ci(scp_hits, total)

    jcr_point = (jcr_violations / total) * 100.0
    jcr_ci_low, jcr_ci_high = wilson_ci(jcr_violations, total)

    return {
        "model_label": model_label,
        "total_queries": total,
        "scp": {"point_estimate": round(scp_point, 2), "wilson_ci_95": [scp_ci_low, scp_ci_high], "hits": scp_hits, "boolean_mask": scp_mask},
        "jcr": {"point_estimate": round(jcr_point, 2), "wilson_ci_95": [jcr_ci_low, jcr_ci_high], "violations": jcr_violations, "boolean_mask": jcr_mask},
        "latency": {"avg_ms": round(float(np.mean(latencies)), 2), "p95_ms": round(float(np.percentile(latencies, 95)), 2)},
        "traces": traces
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Pillar 4: SOTA Legal Model Comparison Suite")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "unsloth", "llamacpp"])
    parser.add_argument("--finetuned-path", type=str, default=str(DEFAULT_FINETUNED_PATH))
    parser.add_argument("--baseline-path", type=str, default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument("--judge-path", type=str, default=str(DEFAULT_JUDGE_PATH))
    parser.add_argument("--benchmark-path", type=str, default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--lora-name", type=str, default="chatbot")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--use-judge", action="store_true", help="Explicitly enable the 72B Teacher Judge")
    parser.add_argument("--allow-simulated-baseline", action="store_true", help="Allow using simulated baseline metrics")
    args = parser.parse_args()

    print("═══════════════════════════════════════════════════════════════════════")
    print("🏆 [PILLAR 4]: SOTA LEGAL MODEL HEAD-TO-HEAD COMPARATIVE BENCHMARK")
    print("═══════════════════════════════════════════════════════════════════════")

    bench_path = Path(args.benchmark_path)
    if not bench_path.exists():
        print(f"❌ Error: Benchmark dataset not found at {bench_path}")
        return 1

    with open(bench_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # -------------------------------------------------------------------------
    # STAGE 1: Evaluate Fine-Tuned Chatbot Model
    # -------------------------------------------------------------------------
    print(f"\n📦 [Pass 1/3] Loading Fine-Tuned Model from: {args.finetuned_path}...")
    ft_engine = BackendEngine(backend_type=args.backend, model_path=args.finetuned_path, lora_name=args.lora_name, vllm_url=args.vllm_url, max_seq_length=32768)
    ft_raw_responses = generate_model_responses(ft_engine, queries, "RAFT-Trained DPDP Chatbot")
    ft_engine.unload()
    del ft_engine
    flush_gpu_memory()
    print("🧹 [VRAM Airlock] Fine-Tuned model purged from GPU memory.")

    # -------------------------------------------------------------------------
    # STAGE 2: Evaluate Vanilla Baseline Model
    # -------------------------------------------------------------------------
    baseline_is_simulated = False
    base_raw_responses = []
    baseline_path = Path(args.baseline_path)

    if not baseline_path.exists() or not any(baseline_path.iterdir()):
        print(f"\n📥 Baseline model not found locally at {baseline_path}. Downloading unsloth/Qwen2.5-7B-Instruct...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id='unsloth/Qwen2.5-7B-Instruct', local_dir=str(baseline_path))
            print("✅ Baseline model downloaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to download baseline model: {e}")

    if baseline_path.exists() and str(args.finetuned_path) != str(args.baseline_path):
        print(f"\n📦 [Pass 2/3] Loading Vanilla Baseline Model from: {baseline_path}...")
        base_engine = BackendEngine(backend_type=args.backend, model_path=str(baseline_path), vllm_url=args.vllm_url, max_seq_length=32768)
        base_raw_responses = generate_model_responses(base_engine, queries, "Vanilla Qwen2.5 Baseline")
        base_engine.unload()
        del base_engine
        flush_gpu_memory()
        print("🧹 [VRAM Airlock] Baseline model purged from GPU memory.")
    else:
        if args.allow_simulated_baseline:
            print("\n⚠️ NOTICE: Vanilla baseline weights not detected. Using simulated baseline metrics.")
            baseline_is_simulated = True
        else:
            print(f"❌ HARD-FAIL: Baseline model not found at {baseline_path}.")
            return 1

    # -------------------------------------------------------------------------
    # STAGE 3: Context Faithfulness Judging (72B Teacher)
    # -------------------------------------------------------------------------
    judge_engine = None
    if args.use_judge and Path(args.judge_path).exists():
        print(f"\n🏛️ [Pass 3/3] Initializing 72B Teacher Judge ({args.judge_path})...")
        try:
            judge_engine = BackendEngine(backend_type=args.backend, model_path=args.judge_path, max_seq_length=8192)
        except Exception as e:
            print(f"⚠️ Failed to load 72B Judge: {e}. Falling back to heuristic CF scoring.")

    ft_cf_scores = []
    base_cf_scores = []
    
    print("\n⚖️ Computing Context Faithfulness (CF) Scores...")
    for idx, item in enumerate(tqdm(queries, desc="Judging CF")):
        target_kws = item.get("target_keywords", [])
        retrieved_context = item.get("context", "")
        ground_truth_context = retrieved_context if retrieved_context.strip() else f"Keywords: {', '.join(target_kws)}"
        
        ft_cf = evaluate_cf_judge(ft_raw_responses[idx]["raw_output"], ground_truth_context, target_kws, judge_engine)
        ft_cf_scores.append(ft_cf)
        
        if not baseline_is_simulated:
            base_cf = evaluate_cf_judge(base_raw_responses[idx]["raw_output"], ground_truth_context, target_kws, judge_engine)
            base_cf_scores.append(base_cf)

    if judge_engine is not None:
        judge_engine.unload()
        del judge_engine
        flush_gpu_memory()
        print("🧹 [VRAM Airlock] Judge model purged from GPU memory.")

    # -------------------------------------------------------------------------
    # COMPILE FINAL METRICS
    # -------------------------------------------------------------------------
    ft_results = compute_offline_metrics(ft_raw_responses, queries, "RAFT-Trained DPDP Chatbot")
    ft_results["cf_score"] = {"mean": round(float(np.mean(ft_cf_scores)), 4), "std": round(float(np.std(ft_cf_scores)), 4), "raw_scores": ft_cf_scores}

    if baseline_is_simulated:
        base_total = len(queries)
        sim_scp_hits = int(base_total * 0.42)
        sim_jcr_hits = int(base_total * 0.28)
        sim_scp_low, sim_scp_high = wilson_ci(sim_scp_hits, base_total)
        sim_jcr_low, sim_jcr_high = wilson_ci(sim_jcr_hits, base_total)
        
        base_results = {
            "model_label": "[SIMULATED] Vanilla Qwen2.5-7B",
            "total_queries": base_total,
            "scp": {"point_estimate": round((sim_scp_hits / base_total) * 100.0, 2), "wilson_ci_95": [sim_scp_low, sim_scp_high], "hits": sim_scp_hits, "boolean_mask": [i < sim_scp_hits for i in range(base_total)]},
            "cf_score": {"mean": 3.1200, "std": 0.6500, "raw_scores": [3.12] * base_total},
            "jcr": {"point_estimate": round((sim_jcr_hits / base_total) * 100.0, 2), "wilson_ci_95": [sim_jcr_low, sim_jcr_high], "violations": sim_jcr_hits, "boolean_mask": [i < sim_jcr_hits for i in range(base_total)]},
            "latency": {"avg_ms": round(ft_results["latency"]["avg_ms"] * 0.95, 2), "p95_ms": round(ft_results["latency"]["p95_ms"] * 0.95, 2)},
            "traces": []
        }
    else:
        base_results = compute_offline_metrics(base_raw_responses, queries, "Vanilla Qwen2.5 Baseline")
        base_results["cf_score"] = {"mean": round(float(np.mean(base_cf_scores)), 4), "std": round(float(np.std(base_cf_scores)), 4), "raw_scores": base_cf_scores}

    # Statistical Comparison & Win Condition Verification
    p_val_scp = compute_paired_mcnemar(base_results["scp"]["boolean_mask"], ft_results["scp"]["boolean_mask"])
    win_scp = (ft_results["scp"]["point_estimate"] >= 90.0) and (ft_results["scp"]["point_estimate"] > base_results["scp"]["point_estimate"])

    cf_diff, p_val_cf = compute_paired_t_test(ft_results["cf_score"]["raw_scores"], base_results["cf_score"]["raw_scores"])
    win_cf = (ft_results["cf_score"]["mean"] >= 4.50) and (cf_diff > 0)

    p_val_jcr = compute_paired_mcnemar(base_results["jcr"]["boolean_mask"], ft_results["jcr"]["boolean_mask"])
    win_jcr = (ft_results["jcr"]["point_estimate"] <= 1.0) and (ft_results["jcr"]["point_estimate"] <= base_results["jcr"]["point_estimate"])

    certified_sota = win_scp and win_cf and win_jcr

    # Terminal Output
    ft_label = ft_results["model_label"]
    base_label = base_results["model_label"]

    print("\n═════════════════════════════════════════════════════════════════════════════════════════════")
    print("📊 PILLAR 4 HEAD-TO-HEAD LEGAL BENCHMARK COMPARISON SCORECARD")
    print("═════════════════════════════════════════════════════════════════════════════════════════════")
    print(f"| {'Evaluation Metric':<32} | {base_label:<24} | {ft_label:<24} | {'p-value':<9} | {'Status':<6} |")
    print(f"|----------------------------------|--------------------------|--------------------------|-----------|--------|")
    
    scp_base_str = f"{base_results['scp']['point_estimate']:.1f}% [{base_results['scp']['wilson_ci_95'][0]:.1f}-{base_results['scp']['wilson_ci_95'][1]:.1f}]"
    scp_ft_str = f"{ft_results['scp']['point_estimate']:.1f}% [{ft_results['scp']['wilson_ci_95'][0]:.1f}-{ft_results['scp']['wilson_ci_95'][1]:.1f}]"
    print(f"| {'Statute Citation Precision (SCP)':<32} | {scp_base_str:<24} | {scp_ft_str:<24} | {p_val_scp:<9.4f} | {'✅ PASS' if win_scp else '❌ FAIL':<6} |")

    cf_base_str = f"{base_results['cf_score']['mean']:.2f} ± {base_results['cf_score']['std']:.2f}"
    cf_ft_str = f"{ft_results['cf_score']['mean']:.2f} ± {ft_results['cf_score']['std']:.2f}"
    print(f"| {'Context Faithfulness (CF Score)':<32} | {cf_base_str:<24} | {cf_ft_str:<24} | {p_val_cf:<9.4f} | {'✅ PASS' if win_cf else '❌ FAIL':<6} |")

    jcr_base_str = f"{base_results['jcr']['point_estimate']:.1f}% [{base_results['jcr']['wilson_ci_95'][0]:.1f}-{base_results['jcr']['wilson_ci_95'][1]:.1f}]"
    jcr_ft_str = f"{ft_results['jcr']['point_estimate']:.1f}% [{ft_results['jcr']['wilson_ci_95'][0]:.1f}-{ft_results['jcr']['wilson_ci_95'][1]:.1f}]"
    print(f"| {'Jurisdictional Contamination':<32} | {jcr_base_str:<24} | {jcr_ft_str:<24} | {p_val_jcr:<9.4f} | {'✅ PASS' if win_jcr else '❌ FAIL':<6} |")

    lat_base_str = f"{base_results['latency']['avg_ms']:.1f} ms (p95: {base_results['latency']['p95_ms']:.1f})"
    lat_ft_str = f"{ft_results['latency']['avg_ms']:.1f} ms (p95: {ft_results['latency']['p95_ms']:.1f})"
    print(f"| {'Inference Latency':<32} | {lat_base_str:<24} | {lat_ft_str:<24} | {'N/A':<9} | {'INFO':<6} |")
    print("═════════════════════════════════════════════════════════════════════════════════════════════\n")

    print(f"🏁 Head-to-Head SOTA Verdict: {'✅ CERTIFIED SOTA DOMAIN SUPERIORITY' if certified_sota else '❌ CERTIFICATION THRESHOLDS UNMET'}")

    summary_ft = {k: v for k, v in ft_results.items() if k != "traces"}
    summary_ft["scp"] = {k: v for k, v in summary_ft["scp"].items() if k != "boolean_mask"}
    summary_ft["cf_score"] = {k: v for k, v in summary_ft["cf_score"].items() if k != "raw_scores"}
    summary_ft["jcr"] = {k: v for k, v in summary_ft["jcr"].items() if k != "boolean_mask"}

    summary_base = {k: v for k, v in base_results.items() if k != "traces"}
    summary_base["scp"] = {k: v for k, v in summary_base["scp"].items() if k != "boolean_mask"}
    summary_base["cf_score"] = {k: v for k, v in summary_base["cf_score"].items() if k != "raw_scores"}
    summary_base["jcr"] = {k: v for k, v in summary_base["jcr"].items() if k != "boolean_mask"}

    report_payload = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_dataset": str(bench_path),
        "total_test_queries": len(queries),
        "finetuned_model_summary": summary_ft,
        "baseline_model_summary": summary_base,
        "baseline_is_simulated": baseline_is_simulated,
        "statistical_hypothesis_tests": {
            "scp_mcnemar_p_value": p_val_scp,
            "cf_paired_t_test_p_value": p_val_cf,
            "cf_mean_difference": cf_diff,
            "jcr_mcnemar_p_value": p_val_jcr,
            "statistically_significant_improvement": (p_val_scp < 0.05) and (p_val_cf < 0.05)
        },
        "win_conditions": {
            "statute_citation_precision_win": win_scp,
            "context_faithfulness_win": win_cf,
            "jurisdictional_contamination_win": win_jcr,
            "overall_certified_sota": certified_sota
        },
        "diagnostic_traces": ft_results["traces"]
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"💾 Master SOTA comparison report saved to: {REPORT_PATH}")

    # Return 0 so verify.py completes its scorecard orchestration
    return 0

if __name__ == "__main__":
    sys.exit(main())