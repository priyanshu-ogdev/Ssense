#!/usr/bin/env python3
"""
run_chatbot_evals.py – Conversational Chatbot Evaluation Suite (Universal Dual-Backend Grade)

Evaluates `chatbot-model-final` across 3 primary conversational benchmarks using `dpdp_chatbot_qa.json`:
1. Statutory Accuracy & Factual Consistency (verifying correct explanations & zero forbidden hallucination terms)
2. Conversational Fluidity & Anti-Reward-Hacking (Type-Token Ratio vocabulary diversity & natural dialogue structure)
3. No-Bleed Schema Containment (verifying zero Auditor JSON / system preamble bleed into chat responses)
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
from typing import Dict, List, Any, Optional
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
DEFAULT_BENCHMARK_PATH = Path("ml/evals/benchmarks/dpdp_chatbot_qa.json")
DEFAULT_MODEL_PATH = Path("../models/chatbot-model-final")
REPORT_DIR = Path("ml/evals/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

AUDITOR_BLEED_TERMS = [
    "global_legal_reasoning",
    "network_action",
    "subtlety_score",
    "[CONTEXT: THE LAW]",
    "[SYNTHESIZED POLICY]",
    "statute_reference",
    "offending_entities"
]

# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════
def calculate_ttr(text: str) -> float:
    """Calculate Type-Token Ratio (vocabulary diversity) to verify anti-reward-hacking fluidity."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def check_forbidden_terms(text: str, forbidden_list: List[str]) -> List[str]:
    """Check if any forbidden hallucination terms appear in the response."""
    found = []
    text_lower = text.lower()
    for term in forbidden_list:
        if term.lower() in text_lower:
            found.append(term)
    return found

def check_schema_bleed(text: str) -> List[str]:
    """Check if any Auditor internal JSON schema keys or preambles bled into the chatbot response."""
    found = []
    text_lower = text.lower()
    for term in AUDITOR_BLEED_TERMS:
        if term.lower() in text_lower:
            found.append(term)
    return found

def evaluate_key_points_coverage(response: str, expected_points: List[str]) -> float:
    """Estimate basic keyword/concept coverage of expected statutory key points."""
    if not expected_points:
        return 1.0
    covered = 0
    resp_lower = response.lower()
    for pt in expected_points:
        # Extract significant keywords (length >= 4)
        keywords = [w for w in re.findall(r'\b\w+\b', pt.lower()) if len(w) >= 4]
        if not keywords:
            covered += 1
            continue
        # If at least 40% of key keywords appear, consider the point covered
        matches = sum(1 for kw in keywords if kw in resp_lower)
        if matches / len(keywords) >= 0.4:
            covered += 1
    return covered / len(expected_points)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Conversational Chatbot Statutory & Fluidity Evals")
    parser.add_argument("--backend", type=str, default="unsloth", choices=["unsloth", "vllm", "llamacpp"])
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--lora-name", type=str, default="chatbot")
    args = parser.parse_args()

    bench_path = Path(args.benchmark_path)
    if not bench_path.exists():
        print(f"⚠️ Benchmark file not found: {bench_path}")
        return

    with open(bench_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    print(f"🚀 Running Conversational Chatbot Evals across {len(qa_data)} statutory Q&A scenarios (backend: {args.backend})...")
    engine = BackendEngine(
        backend_type=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        vllm_url=args.vllm_url,
        lora_name=args.lora_name
    )

    results = []
    total_accuracies = []
    total_ttrs = []
    total_halluc_violations = 0
    total_bleed_violations = 0

    for item in tqdm(qa_data, desc="Evaluating Chatbot Responses"):
        prompt = f"""<|im_start|>system
You are a warm, empathetic, and expert Indian DPDP Legal Assistant. Answer the user's questions accurately according to the DPDP Act 2023 and Rules 2025 in a natural, helpful conversational tone.<|im_end|>
<|im_start|>user
{item['question']}<|im_end|>
<|im_start|>assistant
"""
        out = engine.generate(prompt, max_tokens=1024, temperature=0.1)
        resp = out["raw_output"]

        coverage = evaluate_key_points_coverage(resp, item.get("expected_key_points", []))
        forbidden_hits = check_forbidden_terms(resp, item.get("forbidden_hallucination_terms", []))
        bleed_hits = check_schema_bleed(resp)
        ttr = calculate_ttr(resp)

        # Accuracy is coverage minus penalty for forbidden hallucination terms
        accuracy_score = max(0.0, coverage - (0.5 * len(forbidden_hits)))
        total_accuracies.append(accuracy_score)
        total_ttrs.append(ttr)

        if forbidden_hits:
            total_halluc_violations += 1
        if bleed_hits:
            total_bleed_violations += 1

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "accuracy_score": round(accuracy_score, 4),
            "ttr_fluidity": round(ttr, 4),
            "forbidden_hits": forbidden_hits,
            "bleed_hits": bleed_hits,
            "latency_ms": out["latency_ms"]
        })

    avg_accuracy = (sum(total_accuracies) / len(total_accuracies)) * 100 if total_accuracies else 0.0
    avg_ttr = float(np.mean(total_ttrs)) if total_ttrs else 0.0
    bleed_rate = (total_bleed_violations / len(qa_data)) * 100 if qa_data else 0.0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model_path": args.model_path,
        "total_qa_scenarios": len(qa_data),
        "avg_statutory_accuracy_rate": round(avg_accuracy, 2),
        "avg_ttr_fluidity_score": round(avg_ttr, 4),
        "schema_bleed_rate": round(bleed_rate, 2),
        "details": results
    }

    report_path = REPORT_DIR / "chatbot_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "═"*70)
    print("📊 CONVERSATIONAL CHATBOT EVALUATION SUMMARY")
    print("═"*70)
    print(f"   • Total Scenarios Evaluated:       {len(qa_data)}")
    print(f"   • Statutory Accuracy Rate:         {avg_accuracy:.2f}% (Threshold: >= 95.0%)")
    print(f"   • Average TTR Fluidity Score:      {avg_ttr:.4f} (Threshold: >= 0.45)")
    print(f"   • No-Bleed Schema Containment:     {100.0 - bleed_rate:.2f}% (Bleed Rate: {bleed_rate:.2f}%)")
    print(f"💾 Detailed report saved to: {report_path}")
    print("═"*70 + "\n")

if __name__ == "__main__":
    main()
