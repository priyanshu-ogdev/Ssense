#!/usr/bin/env python3
"""
chatbot_authenticity.py – Unified Chatbot Authenticity & Faithfulness Evaluation

Merges evaluate_chatbot.py and run_chatbot_evals.py into a single 5-axis rubric:
1. Statute Citation Precision (SCP)
2. Context Faithfulness (CF)
3. Jurisdictional Contamination Rate (JCR)
4. Statutory Accuracy & Key-Point Coverage
5. Schema Bleed Rate + Vocabulary Diversity (MTLD)
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import json
import time
import argparse
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

# Ensure core can be imported

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
from stats import mtld

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Unified Chatbot Authenticity & Fluidity Evals")
    parser.add_argument("--backend", type=str, default="unsloth", choices=["unsloth", "vllm", "llamacpp"])
    parser.add_argument("--model-path", type=str, default=str(Paths.EVALS_DIR.parent / "models" / "chatbot-model-final"))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=str(Paths.CHATBOT_QA_BENCHMARK))
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--use-judge", action="store_true", help="Load 72B teacher model into VRAM as LLM-as-a-Judge")
    parser.add_argument("--lora-name", type=str, default="chatbot")
    args = parser.parse_args()

    bench_path = Path(args.benchmark_path)
    if not bench_path.exists():
        print(f"❌ Error: Benchmark file not found at {bench_path}")
        return 1

    with open(bench_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Attempt loading RAG context index if accessible
    rag_chunks = []
    index_path = Paths.EVALS_DIR.parent / "data-forge" / "dpdp_hybrid_index.pkl"
    if index_path.exists():
        try:
            with open(index_path, "rb") as f:
                idx = pickle.load(f)
                rag_chunks = idx.get("chunks", [])
        except Exception:
            pass

    print(f"🧠 Initializing Chatbot Model Engine (Backend: {args.backend}, Path: {args.model_path})...")
    chatbot_engine = BackendEngine(
        backend_type=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        vllm_url=args.vllm_url,
        lora_name=args.lora_name
    )

    judge_engine = None
    teacher_path = Paths.EVALS_DIR.parent / "models" / "Qwen2-72B-Instruct-FP8"
    if args.use_judge and teacher_path.exists():
        print("🏛️ Initializing 72B Teacher Model offline as LLM-as-a-Judge...")
        judge_engine = BackendEngine(backend_type=args.backend, model_path=str(teacher_path))

    # Metric accumulators
    scp_hits = 0
    cf_scores = []
    jcr_violations = 0
    total_accuracies = []
    total_mtlds = []
    bleed_violations = 0

    detailed_results = []

    print(f"\n🚀 Executing 5-Axis Authenticity Rubric across {len(test_data)} statutory Q&A scenarios...")
    for i, item in enumerate(tqdm(test_data, desc="Evaluating Chatbot")):
        query = item.get("query", item.get("question", ""))
        target_section = item.get("target_section", "")
        target_keywords = item.get("target_keywords", item.get("expected_key_points", []))
        
        # Simulate RAG retrieval context injection
        retrieved_context = "Digital Personal Data Protection Act 2023 & Rules 2025 relevant statutory provisions."
        if rag_chunks and len(rag_chunks) > i:
            retrieved_context = rag_chunks[i % len(rag_chunks)]

        sys_msg = "You are a warm, empathetic, and strictly compliant Indian DPDP Legal Assistant. Answer the query accurately according to the DPDP Act 2023 without citing foreign legal frameworks or inventing speculative rules."
        user_msg = f"[RETRIEVED_LAW_CONTEXT]:\n{retrieved_context}\n\nQuery: {query}"
        
        prompt = format_chatml_prompt(sys_msg, user_msg)
        
        out = chatbot_engine.generate(prompt, max_tokens=1024, temperature=0.1)
        resp = out["raw_output"]

        # 1. Statute Citation Precision (SCP)
        is_precise = evaluate_scp(resp, target_section)
        if is_precise:
            scp_hits += 1

        # 2. Context Faithfulness (CF 1-5)
        cf_score = evaluate_cf_judge(resp, retrieved_context, target_keywords, judge_engine)
        cf_scores.append(cf_score)

        # 3. Jurisdictional Contamination Rate (JCR)
        is_contaminated, found_contaminants = evaluate_jcr(resp)
        if is_contaminated:
            jcr_violations += 1

        # 4. Accuracy & Coverage (minus forbidden hallucination penalty)
        coverage = evaluate_key_points_coverage(resp, target_keywords)
        forbidden_hits = check_forbidden_terms(resp, item.get("forbidden_hallucination_terms", []))
        accuracy_score = max(0.0, coverage - (0.5 * len(forbidden_hits)))
        total_accuracies.append(accuracy_score)

        # 5. Schema Bleed & Fluidity (MTLD)
        bleed_hits = check_schema_bleed(resp)
        if bleed_hits:
            bleed_violations += 1
            
        fluidity = mtld(resp)
        total_mtlds.append(fluidity)

        detailed_results.append({
            "id": item.get("id", f"q_{i}"),
            "query": query,
            "scp_pass": is_precise,
            "cf_score": cf_score,
            "jcr_contaminated": is_contaminated,
            "found_contaminants": found_contaminants,
            "accuracy_score": round(accuracy_score, 4),
            "forbidden_hits": forbidden_hits,
            "bleed_hits": bleed_hits,
            "mtld_fluidity": round(fluidity, 4),
            "response_snippet": resp[:250] + "..." if len(resp) > 250 else resp
        })

    total = max(1, len(test_data))
    scp_rate = (scp_hits / total) * 100.0
    avg_cf = float(np.mean(cf_scores)) if cf_scores else 5.0
    jcr_rate = (jcr_violations / total) * 100.0
    avg_accuracy = (sum(total_accuracies) / total) * 100.0 if total_accuracies else 0.0
    avg_mtld = float(np.mean(total_mtlds)) if total_mtlds else 0.0
    bleed_rate = (bleed_violations / total) * 100.0

    print("\n═══════════════════════════════════════════════════════════════════════")
    print("📊 PILLAR 2 CHATBOT AUTHENTICITY & GROUNDING EVALUATION REPORT")
    print("═══════════════════════════════════════════════════════════════════════")
    print(f"| Evaluation Metric               | Measured Value | Win Target  | Certification Status |")
    print(f"|---------------------------------|----------------|-------------|----------------------|")
    print(f"| Statute Citation Precision (SCP)| {scp_rate:13.2f}% | > 90.0%     | {'✅ PASS' if scp_rate >= 90.0 else '❌ FAIL'}             |")
    print(f"| Context Faithfulness (CF Score) | {avg_cf:13.2f}/5 | > 4.50 / 5  | {'✅ PASS' if avg_cf >= 4.5 else '❌ FAIL'}             |")
    print(f"| Jurisdictional Contamination    | {jcr_rate:13.2f}% | 0.00% (Strict)| {'✅ PASS' if jcr_rate == 0.0 else '❌ FAIL'}             |")
    print(f"| Statutory Accuracy Rate         | {avg_accuracy:13.2f}% | > 95.0%     | {'✅ PASS' if avg_accuracy >= 95.0 else '❌ FAIL'}             |")
    print(f"| Schema Bleed Rate               | {bleed_rate:13.2f}% | 0.00%       | {'✅ PASS' if bleed_rate == 0.0 else '❌ FAIL'}             |")
    print(f"| MTLD Fluidity Score             | {avg_mtld:13.4f} | >= 40.0     | {'✅ PASS' if avg_mtld >= 40.0 else '❌ FAIL'}             |")
    print("═══════════════════════════════════════════════════════════════════════\n")

    passed_all = (scp_rate >= 90.0) and (avg_cf >= 4.5) and (jcr_rate == 0.0) and (avg_accuracy >= 95.0) and (bleed_rate == 0.0)

    Paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = Paths.REPORTS_DIR / "chatbot_authenticity_report.json"
    
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
            "avg_ttr_fluidity_score": round(avg_mtld, 4),  # kept name as avg_ttr_fluidity_score for verify.py compatibility or update verify.py
            "certified_sota": passed_all
        },
        "detailed_evaluations": detailed_results[:10]
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
        
    print(f"💾 Pillar 2 evaluation report saved to: {report_path}")
    return 0 if passed_all else 1

if __name__ == "__main__":
    sys.exit(main())
