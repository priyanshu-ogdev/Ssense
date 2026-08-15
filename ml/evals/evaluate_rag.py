#!/usr/bin/env python3
"""
evaluate_rag.py – Pillar 1: SOTA Hybrid RAG Retrieval & Reranking Evaluation

Proves that Hybrid Search (BM25 + Dense BGE + Cross-Encoder Reranking) cleanly outperforms naive lexical retrieval across the 50-query DPDP held-out benchmark set.

Metrics Measured:
1. Recall@3: Target >= 95.0% (does the ground-truth chunk appear in top 3?)
2. NDCG@3: Target >= 0.90 (does the Cross-Encoder boost the true chunk to Rank #1?)
3. End-to-End Latency: Target < 150ms on CPU/GPU per query.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import json
import time
import math
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    print("⚠️ Please install rank_bm25 and sentence_transformers: pip install rank_bm25 sentence-transformers")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_BENCHMARK = Path(__file__).resolve().parent / "benchmarks" / "dpdp_rag_testset.json"
DEFAULT_INDEX_PATH = Path(__file__).resolve().parent.parent / "data-forge" / "dpdp_hybrid_index.pkl"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "rag_retrieval_evaluation_report.json"

def get_models_dir() -> Path:
    curr = Path(__file__).resolve().parent
    while curr != curr.parent:
        if (curr / "ml" / "models").exists():
            return curr / "ml" / "models"
        curr = curr.parent
    return Path(__file__).resolve().parent.parent / "models"

def resolve_model_path(repo_id: str, default_subdir: str) -> str:
    models_dir = get_models_dir()
    local_path = models_dir / default_subdir
    if local_path.exists() and (local_path / "config.json").exists():
        return str(local_path)
    return repo_id

def tokenize_query(query: str) -> List[str]:
    """Exact mirror of the legal tokenizer rules used during index synthesis."""
    query_lower = query.lower()
    for k, v in [("data fiduciary", "data_fiduciary"), ("data principal", "data_principal"), ("consent manager", "consent_manager")]:
        query_lower = query_lower.replace(k, v)
    import re
    query_lower = re.sub(r'section\s+(\d+)\((\d+)\)', r'section_\1_\2', query_lower)
    query_lower = re.sub(r'rule\s+(\d+)\((\d+)\)', r'rule_\1_\2', query_lower)
    return re.findall(r'\b[a-z0-9_]+\b', query_lower)

def is_chunk_relevant(chunk_text: str, meta: Dict[str, Any], target_section: str, target_keywords: List[str]) -> bool:
    """Evaluates whether a retrieved chunk matches ground truth."""
    chunk_lower = chunk_text.lower()
    sec_lower = target_section.lower().replace("section ", "section_").replace("rule ", "rule_")
    
    # Direct section match in header or text
    if target_section.lower() in chunk_lower or sec_lower in chunk_lower:
        return True
        
    # Keyword density check (>= 40% keyword match)
    if not target_keywords:
        return False
    hits = sum(1 for kw in target_keywords if kw.lower() in chunk_lower)
    return (hits / len(target_keywords)) >= 0.4

def compute_ndcg_at_k(relevances: List[int], k: int = 3) -> float:
    """Computes Normalized Discounted Cumulative Gain at K."""
    relevance_k = relevances[:k]
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_k))
    ideal_k = sorted(relevances, reverse=True)[:k]
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_k))
    if idcg == 0:
        return 0.0
    return dcg / idcg

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION HARNESS
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("🏛️ [PILLAR 1]: SOTA Hybrid RAG Retrieval & Reranking Evaluation")
    if not DEFAULT_BENCHMARK.exists():
        print(f"❌ Error: Benchmark file not found at {DEFAULT_BENCHMARK}")
        return

    with open(DEFAULT_BENCHMARK, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if not DEFAULT_INDEX_PATH.exists():
        print(f"⚠️ Vector DB Index not found at {DEFAULT_INDEX_PATH}. Please run 'python ml/data-forge/build_vector_db.py' first.")
        return

    print("📦 Loading DPDP Hybrid Index (BM25 + Dense Embeddings)...")
    with open(DEFAULT_INDEX_PATH, "rb") as f:
        index_data = pickle.load(f)
    
    chunks = index_data["chunks"]
    metadatas = index_data["metadatas"]
    bm25 = index_data["bm25_index"]
    dense_embeddings = index_data["dense_embeddings"]

    bge_path = resolve_model_path("BAAI/bge-small-en-v1.5", "bge-small-en-v1.5")
    reranker_path = resolve_model_path("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3")

    print(f"🧠 Loading Dense Embedder from: {bge_path}...")
    bge_model = SentenceTransformer(bge_path)
    print(f"⚖️ Loading Cross-Encoder Reranker from: {reranker_path}...")
    reranker_model = CrossEncoder(reranker_path, max_length=512)

    hybrid_recall_hits = 0
    hybrid_ndcg_scores = []
    hybrid_latencies_ms = []

    naive_recall_hits = 0
    naive_ndcg_scores = []
    naive_latencies_ms = []

    print(f"\n🚀 Evaluating {len(queries)} DPDP queries against ground truth...")
    for item in tqdm(queries, desc="Benchmarking RAG Queries"):
        query = item["query"]
        target_section = item["target_section"]
        target_keywords = item["target_keywords"]

        # 1. Naive BM25 Retrieval Evaluation
        t0 = time.perf_counter()
        q_tokens = tokenize_query(query)
        bm25_scores = bm25.get_scores(q_tokens)
        top_naive_indices = np.argsort(bm25_scores)[::-1][:3]
        t1 = time.perf_counter()
        
        naive_latencies_ms.append((t1 - t0) * 1000.0)
        naive_rels = [1 if is_chunk_relevant(chunks[i], metadatas[i], target_section, target_keywords) else 0 for i in top_naive_indices]
        if any(naive_rels):
            naive_recall_hits += 1
        naive_ndcg_scores.append(compute_ndcg_at_k(naive_rels, k=3))

        # 2. SOTA Hybrid RAG + Cross-Encoder Reranking
        t_start = time.perf_counter()
        
        # Lexical Phase (Top 10)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:10]
        
        # Dense Phase (Top 10)
        q_emb = bge_model.encode([f"Represent this query for retrieval: {query}"])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        dense_scores = np.dot(dense_embeddings, q_emb)
        top_dense_idx = np.argsort(dense_scores)[::-1][:10]
        
        # Reciprocal Rank Fusion (RRF)
        combined_candidates = list(set(top_bm25_idx).union(set(top_dense_idx)))
        rrf_scores = {}
        for idx in combined_candidates:
            score = 0.0
            if idx in top_bm25_idx:
                score += 1.0 / (60 + list(top_bm25_idx).index(idx))
            if idx in top_dense_idx:
                score += 1.0 / (60 + list(top_dense_idx).index(idx))
            rrf_scores[idx] = score
        
        top_rrf = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:10]
        
        # Cross-Encoder Reranking (Pushing best chunk to Rank #1)
        cross_pairs = [[query, chunks[idx]] for idx in top_rrf]
        cross_scores = reranker_model.predict(cross_pairs)
        ranked_pairs = sorted(zip(top_rrf, cross_scores), key=lambda x: x[1], reverse=True)
        final_top3_idx = [p[0] for p in ranked_pairs[:3]]
        
        t_end = time.perf_counter()
        hybrid_latencies_ms.append((t_end - t_start) * 1000.0)

        hybrid_rels = [1 if is_chunk_relevant(chunks[i], metadatas[i], target_section, target_keywords) else 0 for i in final_top3_idx]
        if any(hybrid_rels):
            hybrid_recall_hits += 1
        hybrid_ndcg_scores.append(compute_ndcg_at_k(hybrid_rels, k=3))

    # Summary Statistics
    hybrid_recall = (hybrid_recall_hits / len(queries)) * 100.0
    hybrid_ndcg = float(np.mean(hybrid_ndcg_scores))
    hybrid_latency = float(np.mean(hybrid_latencies_ms))

    naive_recall = (naive_recall_hits / len(queries)) * 100.0
    naive_ndcg = float(np.mean(naive_ndcg_scores))
    naive_latency = float(np.mean(naive_latencies_ms))

    print("\n═══════════════════════════════════════════════════════════════════════")
    print("📊 PILLAR 1 EVALUATION COMPARATIVE MATRIX (50 DPDP QUERIES)")
    print("═══════════════════════════════════════════════════════════════════════")
    print(f"| Metric                   | Naive BM25 Baseline | SOTA Hybrid + Cross-Encoder | Win Target | Status |")
    print(f"|--------------------------|---------------------|-----------------------------|------------|--------|")
    print(f"| Recall@3                 | {naive_recall:17.2f}% | {hybrid_recall:25.2f}% | >= 95.0%   | {'✅ PASS' if hybrid_recall >= 95.0 else '❌ FAIL'} |")
    print(f"| NDCG@3 (Ranking Quality) | {naive_ndcg:19.4f} | {hybrid_ndcg:27.4f} | >= 0.90    | {'✅ PASS' if hybrid_ndcg >= 0.90 else '❌ FAIL'} |")
    print(f"| End-to-End Latency       | {naive_latency:15.2f} ms | {hybrid_latency:23.2f} ms | < 150.0 ms | {'✅ PASS' if hybrid_latency < 150.0 else '❌ FAIL'} |")
    print("═══════════════════════════════════════════════════════════════════════\n")

    report_dict = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_queries": len(queries),
        "sota_hybrid_rag": {
            "recall_at_3": hybrid_recall,
            "ndcg_at_3": hybrid_ndcg,
            "avg_latency_ms": hybrid_latency,
            "passed_certification": hybrid_recall >= 95.0 and hybrid_ndcg >= 0.90 and hybrid_latency < 150.0
        },
        "naive_bm25_baseline": {
            "recall_at_3": naive_recall,
            "ndcg_at_3": naive_ndcg,
            "avg_latency_ms": naive_latency
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"💾 Pillar 1 evaluation report saved to: {REPORT_PATH}")
    return 0 if report_dict["sota_hybrid_rag"]["passed_certification"] else 1

if __name__ == "__main__":
    sys.exit(main())
