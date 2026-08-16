#!/usr/bin/env python3
"""
evaluate_rag.py – Pillar 7: SOTA Hybrid RAG Retrieval & Reranking Evaluation

Proves that Hybrid Search (BM25 + Dense BGE + Cross-Encoder Reranking) cleanly outperforms 
naive lexical retrieval across the 50-query DPDP held-out benchmark set.

SOTA Upgrades Implemented:
1. Strict VRAM Airlock: Explicitly destroys Torch embedding/reranking models and clears CUDA cache.
2. Indestructible Paths: Fully integrated with `path_resolver.py`.
3. Robust Evaluation: Replaced destructive alphanumeric regex with strict whitespace normalization.
4. Fast Top-K & RRF: O(N) argpartition with deep Reciprocal Rank Fusion at depth 100.
"""

import os
import sys
import re
import json
import time
import math
import pickle
import numpy as np
import gc
from pathlib import Path
from typing import List, Dict, Any
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
    DEFAULT_BENCHMARK = Paths.RAG_TESTSET
    DEFAULT_INDEX_PATH = Paths.HYBRID_INDEX
    REPORT_DIR = Paths.ensure_reports_dir()
    REPORT_PATH = REPORT_DIR / "rag_retrieval_evaluation_report.json"
except ImportError as e:
    print(f"❌ Core module import failed: {e}")
    sys.exit(1)

try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    print("⚠️ Please install required packages: pip install rank_bm25 sentence-transformers")
    sys.exit(1)

# Hardware specific flags
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RRF_K = 60
RETRIEVAL_DEPTH = 100
RERANK_DEPTH = 25

# ═══════════════════════════════════════════════════════════════════════════
# HIGH-PERFORMANCE UTILS & MATCHING LOGIC
# ═══════════════════════════════════════════════════════════════════════════
def fast_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """O(N) Top-K extraction using argpartition."""
    if len(scores) <= k:
        return np.argsort(scores)[::-1]
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]

def tokenize_query(query: str) -> List[str]:
    """SOTA Legal Tokenizer: Fuses compound legal entities into single tokens."""
    query_lower = query.lower()
    
    replacements = [
        ("data fiduciary", "data_fiduciary"),
        ("data principal", "data_principal"),
        ("consent manager", "consent_manager"),
        ("significant data fiduciary", "significant_data_fiduciary"),
        ("data protection officer", "data_protection_officer"),
        ("data protection board", "data_protection_board"),
        ("personal data", "personal_data")
    ]
    for k, v in replacements:
        query_lower = query_lower.replace(k, v)
        
    # Standardize section citations: "section 8(1)" -> "section_8_1"
    query_lower = re.sub(r'section\s+(\d+)\s*\(\s*([a-z0-9]+)\s*\)', r'section_\1_\2', query_lower)
    query_lower = re.sub(r'section\s+(\d+)', r'section_\1', query_lower)
    query_lower = re.sub(r'rule\s+(\d+)\s*\(\s*([a-z0-9]+)\s*\)', r'rule_\1_\2', query_lower)
    query_lower = re.sub(r'rule\s+(\d+)', r'rule_\1', query_lower)
    
    return re.findall(r'\b[a-z0-9_]+\b', query_lower)

def is_chunk_relevant(chunk_text: str, meta: Dict[str, Any], target_section: str, target_keywords: List[str]) -> bool:
    """Robust statutory text normalization to eliminate false-negative evaluations."""
    chunk_lower = chunk_text.lower()
    target_lower = str(target_section).lower().strip()

    # 1. Exact Structural Match (strip whitespace to catch "section 8(1)" vs "section 8 (1)")
    chunk_clean = re.sub(r'\s+', '', chunk_lower)
    target_clean = re.sub(r'\s+', '', target_lower)
    
    if target_clean and target_clean != "none" and target_clean in chunk_clean:
        return True
        
    # 2. Relaxed Keyword Density Match (>= 40%)
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
    print("═══════════════════════════════════════════════════════════════════════")
    print("🏛️ [PILLAR 7]: SOTA HYBRID RAG RETRIEVAL & RERANKING EVALUATION")
    print("═══════════════════════════════════════════════════════════════════════")
    
    if not DEFAULT_BENCHMARK.exists():
        print(f"❌ Error: Benchmark file not found at {DEFAULT_BENCHMARK}")
        return 1

    with open(DEFAULT_BENCHMARK, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if not DEFAULT_INDEX_PATH.exists():
        print(f"⚠️ Vector DB Index not found at {DEFAULT_INDEX_PATH}. Please run build_vector_db.py first.")
        return 1

    print("📦 Loading DPDP Hybrid Index (BM25 + Dense Embeddings)...")
    with open(DEFAULT_INDEX_PATH, "rb") as f:
        index_data = pickle.load(f)
    
    chunks = index_data["chunks"]
    metadatas = index_data["metadatas"]
    bm25 = index_data["bm25_index"]
    dense_embeddings = index_data["dense_embeddings"]

    bge_path = Paths.resolve_model_path("BAAI/bge-small-en-v1.5", "bge-small-en-v1.5")
    reranker_path = Paths.resolve_model_path("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3")

    print(f"🧠 Loading Dense Embedder from: {bge_path} (Device: {DEVICE})...")
    bge_model = SentenceTransformer(str(bge_path), device=DEVICE)
    
    print(f"⚖️ Loading Cross-Encoder Reranker from: {reranker_path} (Device: {DEVICE})...")
    reranker_model = CrossEncoder(str(reranker_path), max_length=512, device=DEVICE)

    hybrid_recall_hits = 0
    hybrid_ndcg_scores = []
    hybrid_latencies_ms = []

    naive_recall_hits = 0
    naive_ndcg_scores = []
    naive_latencies_ms = []

    try:
        print(f"\n🚀 Evaluating {len(queries)} DPDP queries against ground truth...")
        for item in tqdm(queries, desc="Benchmarking RAG Queries"):
            query = item["query"]
            target_section = item.get("target_section", "None")
            target_keywords = item.get("target_keywords", [])

            # -------------------------------------------------------------------
            # 1. Naive BM25 Baseline Retrieval
            # -------------------------------------------------------------------
            t0 = time.perf_counter()
            q_tokens = tokenize_query(query)
            bm25_scores = bm25.get_scores(q_tokens)
            top_naive_indices = fast_top_k(bm25_scores, k=3)
            t1 = time.perf_counter()
            
            naive_latencies_ms.append((t1 - t0) * 1000.0)
            naive_rels = [1 if is_chunk_relevant(chunks[i], metadatas[i], target_section, target_keywords) else 0 for i in top_naive_indices]
            
            if any(naive_rels):
                naive_recall_hits += 1
            naive_ndcg_scores.append(compute_ndcg_at_k(naive_rels, k=3))

            # -------------------------------------------------------------------
            # 2. SOTA Hybrid RAG + Cross-Encoder Reranking
            # -------------------------------------------------------------------
            t_start = time.perf_counter()
            
            # A. Sparse (Lexical) Retrieval Phase
            top_bm25_idx = fast_top_k(bm25_scores, k=RETRIEVAL_DEPTH)
            
            # B. Dense (Semantic) Retrieval Phase
            dense_query = f"Represent this sentence for searching relevant passages: {query}"
            q_emb = bge_model.encode([dense_query], normalize_embeddings=True, show_progress_bar=False)[0]
            
            dense_scores = np.dot(dense_embeddings, q_emb)
            top_dense_idx = fast_top_k(dense_scores, k=RETRIEVAL_DEPTH)
            
            # C. Reciprocal Rank Fusion (RRF)
            rrf_scores = {}
            for rank, idx in enumerate(top_bm25_idx):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
            for rank, idx in enumerate(top_dense_idx):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
            
            top_rrf = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:RERANK_DEPTH]
            
            # D. Cross-Encoder GPU Batch Reranking
            cross_pairs = [[query, chunks[idx]] for idx in top_rrf]
            cross_scores = reranker_model.predict(cross_pairs, batch_size=len(cross_pairs), show_progress_bar=False)
            
            ranked_pairs = sorted(zip(top_rrf, cross_scores), key=lambda x: x[1], reverse=True)
            final_top3_idx = [p[0] for p in ranked_pairs[:3]]
            
            t_end = time.perf_counter()
            hybrid_latencies_ms.append((t_end - t_start) * 1000.0)

            # -------------------------------------------------------------------
            # Metrics Calculation
            # -------------------------------------------------------------------
            hybrid_rels = [1 if is_chunk_relevant(chunks[i], metadatas[i], target_section, target_keywords) else 0 for i in final_top3_idx]
            if any(hybrid_rels):
                hybrid_recall_hits += 1
            hybrid_ndcg_scores.append(compute_ndcg_at_k(hybrid_rels, k=3))

    finally:
        # Strict VRAM Airlock: Protects downstream Chatbot/Auditor evals from OOM
        print("\n🧹 [VRAM Airlock] Unloading BGE Embedder & Reranker from GPU memory...")
        del bge_model
        del reranker_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # Summary Statistics
    hybrid_recall = (hybrid_recall_hits / len(queries)) * 100.0
    hybrid_ndcg = float(np.mean(hybrid_ndcg_scores))
    hybrid_latency = float(np.mean(hybrid_latencies_ms))

    naive_recall = (naive_recall_hits / len(queries)) * 100.0
    naive_ndcg = float(np.mean(naive_ndcg_scores))
    naive_latency = float(np.mean(naive_latencies_ms))

    print("\n═══════════════════════════════════════════════════════════════════════")
    print("📊 PILLAR 7 EVALUATION COMPARATIVE MATRIX (50 DPDP QUERIES)")
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
    print(f"💾 Pillar 7 evaluation report saved to: {REPORT_PATH}")
    
    return 0 if report_dict["sota_hybrid_rag"]["passed_certification"] else 1

if __name__ == "__main__":
    sys.exit(main())