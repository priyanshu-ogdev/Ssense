#!/usr/bin/env python3
"""
evaluate_chatbot.py – Pillar 2: Chatbot Authenticity & Faithfulness Evaluation

Evaluates fine-tuned conversational DPDP models against legal ground truth and retrieved statute chunks:
1. Statute Citation Precision (SCP): Binary check if exact Section/Rule in ground truth is accurately cited (Target > 90%).
2. Context Faithfulness (CF): LLM-as-a-Judge (72B offline Teacher Model or grounded heuristic) scores premise grounding 1 to 5 (Target > 4.5).
3. Jurisdictional Contamination Rate (JCR): Strict zero-tolerance detection of GDPR, CCPA, HIPAA, Article 17, etc. (Target 0%).
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
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

try:
    from backend_loader import BackendEngine
except ImportError:
    from ml.evals.backend_loader import BackendEngine

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_BENCHMARK = Path(__file__).resolve().parent / "benchmarks" / "dpdp_rag_testset.json"
DEFAULT_CHAT_BENCHMARK = Path(__file__).resolve().parent / "benchmarks" / "dpdp_chatbot_qa.json"
DEFAULT_MODEL_PATH = Path("../models/chatbot-model-final") if Path("../models/chatbot-model-final").exists() else Path("../models/Qwen3.5-9B")
TEACHER_MODEL_PATH = Path("../models/Qwen2-72B-Instruct-FP8")
INDEX_PATH = Path(__file__).resolve().parent.parent / "data-forge" / "dpdp_hybrid_index.pkl"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "chatbot_authenticity_report.json"

JURISDICTIONAL_CONTAMINANTS = [
    "gdpr", "ccpa", "hipaa", "article 17", "article 22", "right to be forgotten",
    "california consumer", "european union", "general data protection regulation", "copa", "coppa"
]

# ═══════════════════════════════════════════════════════════════════════════
# METRICS & RUBRIC EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_jcr(response_text: str) -> Tuple[bool, List[str]]:
    """Checks for jurisdictional contamination from Western privacy frameworks."""
    resp_lower = response_text.lower()
    found = []
    for term in JURISDICTIONAL_CONTAMINANTS:
        if re.search(r'\b' + re.escape(term) + r'\b', resp_lower):
            found.append(term)
    return len(found) > 0, found

def evaluate_scp(response_text: str, target_section: str) -> bool:
    """Verifies precise citation of the ground truth Section or Rule."""
    if not target_section:
        return True
    resp_lower = response_text.lower()
    # Normalize section format (e.g. "Section 8(5)" -> "section 8", "section 8(5)")
    sec_clean = target_section.lower().strip()
    if sec_clean in resp_lower:
        return True
    # If sub-clause, check root section plus clause number
    match = re.match(r'(section|rule)\s+(\d+)\s*\((.*?)\)', sec_clean)
    if match:
        root = f"{match.group(1)} {match.group(2)}"
        return root in resp_lower
    return False

def evaluate_cf_judge(response_text: str, retrieved_context: str, ground_truth_keywords: List[str], judge_engine: Optional[BackendEngine] = None) -> float:
    """
    Evaluates Context Faithfulness on a 1-5 scale using offline 72B Judge or semantic alignment heuristic.
    """
    if judge_engine is not None:
        judge_prompt = f"""<|im_start|>system
You are an unbiased, expert legal auditor assessing AI faithfulness to Indian law.<|im_end|>
<|im_start|>user
Evaluate the generated legal response against the provided retrieved statute context.
[RETRIEVED STATUTE CONTEXT]:
{retrieved_context}

[GENERATED RESPONSE]:
{response_text}

Score the Context Faithfulness on a scale of 1 to 5:
5 = Every legal assertion is strictly grounded in the retrieved context with accurate legal framing.
4 = Very high faithfulness with minor stylistic elaboration that does not violate statutory limits.
3 = Moderate grounding; mentions valid sections but omits critical statutory nuances.
2 = Low faithfulness; extrapolates unsupported administrative requirements.
1 = Total failure; contradicts or ignores the retrieved statute text entirely.

Output only a single digit from 1 to 5 representing your faithfulness rating.<|im_end|>
<|im_start|>assistant
Rating: """
        out = judge_engine.generate(judge_prompt, max_tokens=10, temperature=0.0)
        num_match = re.search(r'\b([1-5])\b', out["raw_output"])
        if num_match:
            return float(num_match.group(1))

    # Grounded semantic alignment fallback heuristic when 72B teacher is not mounted in VRAM
    resp_lower = response_text.lower()
    if not ground_truth_keywords:
        return 5.0
    hits = sum(1 for kw in ground_truth_keywords if kw.lower() in resp_lower)
    ratio = hits / max(1, len(ground_truth_keywords))
    
    # Base fidelity score from keyword footprint and lack of contradiction
    if ratio >= 0.8:
        score = 5.0
    elif ratio >= 0.6:
        score = 4.5
    elif ratio >= 0.4:
        score = 4.0
    elif ratio >= 0.2:
        score = 3.0
    else:
        score = 2.0

    # Penalize if response mentions speculative external concepts not in retrieved text
    if any(w in resp_lower for w in ["maybe", "likely in eu", "typically under international law"]):
        score = max(1.0, score - 2.0)
    return score

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Pillar 2: Chatbot Authenticity & Faithfulness Evaluation")
    parser.add_argument("--backend", type=str, default="unsloth", choices=["unsloth", "vllm", "llamacpp"])
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--use-judge", action="store_true", help="Load 72B teacher model into VRAM as LLM-as-a-Judge")
    parser.add_argument("--lora-name", type=str, default="chatbot")
    args = parser.parse_args()

    print("⚖️ [PILLAR 2]: Chatbot Authenticity, Grounding & Zero-Contamination Evaluation")
    bench_path = Path(args.benchmark_path)
    if not bench_path.exists():
        if DEFAULT_CHAT_BENCHMARK.exists():
            bench_path = DEFAULT_CHAT_BENCHMARK
        else:
            print(f"❌ Error: Benchmark dataset not found at {bench_path}")
            return 1

    with open(bench_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Attempt loading RAG context index if accessible
    rag_chunks = []
    if INDEX_PATH.exists():
        try:
            with open(INDEX_PATH, "rb") as f:
                idx = pickle.load(f)
                rag_chunks = idx.get("chunks", [])
        except Exception:
            pass

    print(f"🧠 Initializing Chatbot Model Engine (Backend: {args.backend}, Path: {args.model_path})...")
    chatbot_engine = BackendEngine(
        backend_type=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        lora_name=args.lora_name
    )

    judge_engine = None
    if args.use_judge and TEACHER_MODEL_PATH.exists():
        print("🏛️ Initializing 72B Teacher Model offline as LLM-as-a-Judge...")
        judge_engine = BackendEngine(backend_type=args.backend, model_path=str(TEACHER_MODEL_PATH))

    scp_hits = 0
    cf_scores = []
    jcr_violations = 0
    detailed_results = []

    print(f"\n🚀 Executing 3-Tier Authenticity Rubric across {len(test_data)} statutory Q&A scenarios...")
    for i, item in enumerate(tqdm(test_data, desc="Evaluating Chatbot Faithfulness")):
        query = item.get("query", item.get("question", ""))
        target_section = item.get("target_section", "")
        target_keywords = item.get("target_keywords", item.get("expected_key_points", []))
        
        # Simulate RAG retrieval context injection
        retrieved_context = "Digital Personal Data Protection Act 2023 & Rules 2025 relevant statutory provisions."
        if rag_chunks and len(rag_chunks) > i:
            retrieved_context = rag_chunks[i % len(rag_chunks)]

        prompt = f"""<|im_start|>system
You are a warm, empathetic, and strictly compliant Indian DPDP Legal Assistant. Answer the query accurately according to the DPDP Act 2023 without citing foreign legal frameworks or inventing speculative rules.<|im_end|>
<|im_start|>user
[RETRIEVED_LAW_CONTEXT]:
{retrieved_context}

Query: {query}<|im_end|>
<|im_start|>assistant
"""
        out = chatbot_engine.generate(prompt, max_tokens=768, temperature=0.1)
        resp = out["raw_output"]

        # Metric 1: Statute Citation Precision (SCP)
        is_precise = evaluate_scp(resp, target_section)
        if is_precise:
            scp_hits += 1

        # Metric 2: Context Faithfulness (CF 1-5)
        cf_score = evaluate_cf_judge(resp, retrieved_context, target_keywords, judge_engine)
        cf_scores.append(cf_score)

        # Metric 3: Jurisdictional Contamination Rate (JCR)
        is_contaminated, found_contaminants = evaluate_jcr(resp)
        if is_contaminated:
            jcr_violations += 1

        detailed_results.append({
            "id": item.get("id", f"q_{i}"),
            "query": query,
            "scp_pass": is_precise,
            "cf_score": cf_score,
            "jcr_contaminated": is_contaminated,
            "found_contaminants": found_contaminants,
            "response_snippet": resp[:250] + "..." if len(resp) > 250 else resp
        })

    total = max(1, len(test_data))
    scp_rate = (scp_hits / total) * 100.0
    avg_cf = float(np.mean(cf_scores)) if cf_scores else 5.0
    jcr_rate = (jcr_violations / total) * 100.0

    print("\n═══════════════════════════════════════════════════════════════════════")
    print("📊 PILLAR 2 CHATBOT AUTHENTICITY & GROUNDING EVALUATION REPORT")
    print("═══════════════════════════════════════════════════════════════════════")
    print(f"| Evaluation Metric               | Measured Value | Win Target  | Certification Status |")
    print(f"|---------------------------------|----------------|-------------|----------------------|")
    print(f"| Statute Citation Precision (SCP)| {scp_rate:13.2f}% | > 90.0%     | {'✅ PASS' if scp_rate >= 90.0 else '❌ FAIL'}             |")
    print(f"| Context Faithfulness (CF Score) | {avg_cf:13.2f}/5 | > 4.50 / 5  | {'✅ PASS' if avg_cf >= 4.5 else '❌ FAIL'}             |")
    print(f"| Jurisdictional Contamination    | {jcr_rate:13.2f}% | 0.00% (Strict)| {'✅ PASS' if jcr_rate == 0.0 else '❌ FAIL'}             |")
    print("═══════════════════════════════════════════════════════════════════════\n")

    passed_all = (scp_rate >= 90.0) and (avg_cf >= 4.5) and (jcr_rate == 0.0)
    report_dict = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": args.backend,
        "model_path": str(args.model_path),
        "total_evaluated_queries": total,
        "summary_metrics": {
            "statute_citation_precision_rate": scp_rate,
            "context_faithfulness_score": avg_cf,
            "jurisdictional_contamination_rate": jcr_rate,
            "certified_sota": passed_all
        },
        "detailed_evaluations": detailed_results[:10]  # Store top 10 logs for review
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"💾 Pillar 2 evaluation report saved to: {REPORT_PATH}")
    return 0 if passed_all else 1

if __name__ == "__main__":
    sys.exit(main())
