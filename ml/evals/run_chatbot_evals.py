#!/usr/bin/env python3
"""
run_chatbot_evals.py – Conversational Chatbot Authenticity, Fluidity & Live RAG Evaluation

Measures Chatbot SLM performance across:
1. Live SOTA Hybrid RAG Retrieval (BM25 + BGE Dense + Cross-Encoder Reranking)
2. Statutory Accuracy Rate & Key Points Coverage
3. Vocabulary Diversity & Fluidity (MTLD)
4. Schema Bleed Rate (Ensuring Auditor JSON does not leak into Chatbot UX)
5. Statute Citation Precision (SCP) & Jurisdictional Contamination (JCR)
6. Context Faithfulness (CF) using Heuristics or 72B Teacher Judge

SOTA Upgrades Implemented:
1. Live Hybrid RAG Integration: Queries `dpdp_hybrid_index.pkl` dynamically if context is needed.
2. 3-Stage VRAM Airlock: Sequentially runs (1) RAG Retriever -> (2) 7B Chatbot -> (3) 72B Judge.
3. Full Context Window: Synchronized with 32k context envelope.
4. Prompt & Persona Alignment: Grounded in empathetic legal assistant constraints.
5. Diagnostic Exit Codes: Always returns 0 for clean aggregation in `verify.py`.
"""

import os
import sys
import gc
import re
import json
import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone
from tqdm import tqdm

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

try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False


def flush_gpu():
    """Forces garbage collection and clears CUDA allocator caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ═══════════════════════════════════════════════════════════════════════════
# SOTA HYBRID RAG RETRIEVER MODULE
# ═══════════════════════════════════════════════════════════════════════════
class DynamicRAGRetriever:
    """Encapsulates the SOTA BM25 + Dense BGE + Cross-Encoder Reranking pipeline."""

    def __init__(self, index_path: Path, device: str = "cuda"):
        if not index_path.exists():
            raise FileNotFoundError(f"Hybrid index not found at {index_path}. Run build_vector_db.py first.")

        with open(index_path, "rb") as f:
            index_data = pickle.load(f)

        self.chunks = index_data["chunks"]
        self.metadatas = index_data["metadatas"]
        self.bm25 = index_data["bm25_index"]
        
        # Ensure L2 normalization for exact cosine similarity
        raw_dense = index_data["dense_embeddings"]
        self.dense_embeddings = raw_dense / np.linalg.norm(raw_dense, axis=1, keepdims=True)

        self.device = device if torch.cuda.is_available() else "cpu"
        bge_path = Paths.resolve_model_path("BAAI/bge-small-en-v1.5", "bge-small-en-v1.5")
        reranker_path = Paths.resolve_model_path("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3")

        print(f"🧠 [RAG Retriever] Loading BGE Embedder ({bge_path})...")
        self.embed_model = SentenceTransformer(str(bge_path), device=self.device)

        print(f"⚖️ [RAG Retriever] Loading Cross-Encoder Reranker ({reranker_path})...")
        self.reranker_model = CrossEncoder(str(reranker_path), max_length=512, device=self.device)

        self.generic_stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "of", "and", "in", 
            "to", "for", "with", "on", "at", "by", "from", "as", "that", "this", 
            "it", "be", "or", "which", "will", "would", "could", "should", "their", "they"
        }

    def tokenize_query(self, query: str) -> List[str]:
        words = re.findall(r'\w+', query.lower())
        return [w for w in words if w not in self.generic_stopwords]

    def retrieve(self, query: str, top_k: int = 3, rrf_k: int = 60, rerank_depth: int = 10) -> str:
        """Executes Hybrid Search with Cross-Encoder Reranking and returns formatted context."""
        # 1. Sparse Lexical BM25
        q_tokens = self.tokenize_query(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:100]

        # 2. Dense Semantic Search
        dense_q = f"Represent this sentence for searching relevant passages: {query}"
        q_emb = self.embed_model.encode([dense_q], normalize_embeddings=True, show_progress_bar=False)[0]
        dense_scores = np.dot(self.dense_embeddings, q_emb)
        top_dense_idx = np.argsort(dense_scores)[::-1][:100]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, idx in enumerate(top_bm25_idx):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(top_dense_idx):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        top_rrf = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:rerank_depth]

        # 4. Cross-Encoder Batch Reranking
        cross_pairs = [[query, self.chunks[idx]] for idx in top_rrf]
        cross_scores = self.reranker_model.predict(cross_pairs, batch_size=len(cross_pairs), show_progress_bar=False)

        ranked_pairs = sorted(zip(top_rrf, cross_scores), key=lambda x: x[1], reverse=True)
        final_indices = [p[0] for p in ranked_pairs[:top_k]]

        retrieved_texts = [self.chunks[i].strip() for i in final_indices]
        return "\n\n".join(retrieved_texts)

    def unload(self):
        """Purges RAG embedding models from GPU memory."""
        del self.embed_model
        del self.reranker_model
        flush_gpu()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Pillar 2 & 4: Conversational Chatbot Authenticity & RAG Evals")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "unsloth", "llamacpp"])
    parser.add_argument("--model-path", type=str, default=str(Paths.resolve_model_path(None, "chatbot-model-final")))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=str(Paths.CHATBOT_QA_BENCHMARK))
    parser.add_argument("--index-path", type=str, default=str(Paths.HYBRID_INDEX))
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
    print(f"🚀 [PILLAR 2 & 4]: CHATBOT AUTHENTICITY, FLUIDITY & LIVE RAG ({args.backend.upper()})")
    print("═══════════════════════════════════════════════════════════════════════")

    # -------------------------------------------------------------------------
    # STAGE 1: Live SOTA Hybrid RAG Retrieval (Pre-fetching Context)
    # -------------------------------------------------------------------------
    retrieved_contexts: List[str] = []
    index_file = Path(args.index_path)

    if HAS_RAG_DEPS and index_file.exists():
        print(f"\n📦 [Stage 1/3] Executing Live SOTA Hybrid RAG Retrieval across {len(test_data)} queries...")
        try:
            rag_engine = DynamicRAGRetriever(index_path=index_file)
            for item in tqdm(test_data, desc="RAG Pre-fetching"):
                query = item.get("query", item.get("question", ""))
                # If static context already exists in the test item, respect it; otherwise retrieve live
                static_ctx = item.get("context", "")
                if static_ctx.strip():
                    retrieved_contexts.append(static_ctx)
                else:
                    live_ctx = rag_engine.retrieve(query, top_k=3)
                    retrieved_contexts.append(live_ctx)
        except Exception as e:
            print(f"⚠️ Live RAG retrieval failed: {e}. Falling back to baseline contexts.")
            retrieved_contexts = [item.get("context", "") for item in test_data]
        finally:
            if 'rag_engine' in locals():
                rag_engine.unload()
                del rag_engine
                print("🧹 [VRAM Airlock] RAG embedder & reranker purged from GPU memory.")
    else:
        print("\nℹ️ [Stage 1/3] Using bundled statutory contexts (RAG index not active).")
        retrieved_contexts = [item.get("context", "") for item in test_data]

    # -------------------------------------------------------------------------
    # STAGE 2: Chatbot SLM Response Generation
    # -------------------------------------------------------------------------
    print(f"\n🧠 [Stage 2/3] Initializing Chatbot SLM Engine (Backend: {args.backend})...")
    chatbot_engine = BackendEngine(
        backend_type=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        vllm_url=args.vllm_url,
        lora_name=args.lora_name,
        max_seq_length=32768
    )

    generated_responses = []
    try:
        print(f"Generating completions for {len(test_data)} conversational scenarios...")
        for i, item in enumerate(tqdm(test_data, desc="Chatbot Generation")):
            query = item.get("query", item.get("question", ""))
            ctx = retrieved_contexts[i]

            sys_msg = (
                "You are an empathetic and expert Indian DPDP Legal Assistant. "
                "Answer the user's query accurately according to the Digital Personal Data Protection Act 2023. "
                "If statutory context is provided, you must ground your answer strictly within that context. "
                "If the context does not contain the answer, or if the query falls outside the scope of the DPDP Act, "
                "you must politely decline to answer or state that the Act is silent. "
                "Crucially, you must explicitly cite applicable statutory section numbers. Do not cite foreign statutes (GDPR)."
            )

            if ctx.strip():
                user_msg = f"[STATUTORY CONTEXT]:\n{ctx}\n\nQuery: {query}"
            else:
                user_msg = f"Query: {query}"

            prompt = format_chatml_prompt(sys_msg, user_msg)
            out = chatbot_engine.generate(prompt, max_tokens=2048, temperature=0.0)
            generated_responses.append(out.get("raw_output", ""))

    finally:
        # Strict VRAM Airlock Stage 2
        chatbot_engine.unload()
        del chatbot_engine
        flush_gpu()
        print("🧹 [VRAM Airlock] Chatbot model purged from GPU memory.")

    # -------------------------------------------------------------------------
    # STAGE 3: Optional 72B Teacher Judging & Metrics Scoring
    # -------------------------------------------------------------------------
    judge_engine = None
    if args.use_judge and Path(args.judge_path).exists():
        print(f"\n🏛️ [Stage 3/3] Initializing 72B Teacher Judge ({args.judge_path})...")
        try:
            judge_engine = BackendEngine(backend_type=args.backend, model_path=args.judge_path, max_seq_length=8192)
        except Exception as e:
            print(f"⚠️ Failed to load 72B Judge: {e}. Falling back to heuristic CF scoring.")

    scp_hits = 0
    cf_scores = []
    jcr_violations = 0
    total_accuracies = []
    total_mtlds = []
    bleed_violations = 0
    detailed_results = []

    print("\n📊 Computing Authenticity Metrics & Statistical Distributions...")
    try:
        for i, item in enumerate(tqdm(test_data, desc="Evaluating Metrics")):
            resp = generated_responses[i]
            query = item.get("query", item.get("question", ""))
            target_section = item.get("target_section", "None")
            target_keywords = item.get("target_keywords", item.get("expected_key_points", []))
            ctx = retrieved_contexts[i]

            # 1. Statute Citation Precision (SCP)
            is_precise = evaluate_scp(resp, target_section)
            if is_precise: scp_hits += 1

            # 2. Context Faithfulness (CF 1-5)
            if ctx.strip():
                cf_score = evaluate_cf_judge(resp, ctx, target_keywords, judge_engine)
            else:
                cf_score = 5.0
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
    finally:
        if judge_engine is not None:
            judge_engine.unload()
            del judge_engine
            flush_gpu()
            print("🧹 [VRAM Airlock] Judge model purged from GPU memory.")

    # -------------------------------------------------------------------------
    # AGGREGATION & REPORT COMPILATION
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

    passed_scp = scp_low >= 90.0
    passed_cf = avg_cf >= 4.5
    passed_jcr = jcr_rate == 0.0 
    passed_acc = acc_low >= 95.0
    passed_bleed = bleed_rate == 0.0
    passed_mtld = avg_mtld >= 40.0
    passed_all = passed_scp and passed_cf and passed_jcr and passed_acc and passed_bleed and passed_mtld

    print("\n═══════════════════════════════════════════════════════════════════════════════")
    print("📊 PILLAR 2 & 4 CHATBOT AUTHENTICITY & GROUNDING EVALUATION REPORT")
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
            "avg_ttr_fluidity_score": round(avg_mtld, 4), 
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
        
    print(f"💾 Chatbot evaluation report saved to: {report_path}")

    # Return 0 so verify.py handles diagnostic scorecards cleanly
    return 0


if __name__ == "__main__":
    sys.exit(main())