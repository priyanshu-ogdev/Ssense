#!/usr/bin/env python3
"""
rag_engine.py – SOTA Zero-Hop Hybrid RAG Engine (Attention-Optimized)
Features:
- XML-Structured Context Formatting for optimal LLM attention head parsing.
- BF16 Tensor Core acceleration for BGE Embedding & Reranking.
- O(N) Top-K argpartitioning & L2-normalized Cosine Parity.
- Dynamic CPU Thread scaling for non-blocking FastAPI integration.
"""

import os
import re
import json
import asyncio
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from rank_bm25 import BM25Okapi
from safetensors.numpy import load_file
import torch

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_ML = True
except ImportError:
    HAS_ML = False

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "ml" / "models"


# ═══════════════════════════════════════════════════════════════
# 1. HIGH-PERFORMANCE MATH & LRU CACHE
# ═══════════════════════════════════════════════════════════════
def fast_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """O(N) extraction. Bypasses full array sorting for <1ms execution."""
    if len(scores) <= k:
        return np.argsort(scores)[::-1]
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]

class LRUEmbeddingCache:
    """Bounded LRU Cache to instantly serve repeated semantic queries."""
    def __init__(self, maxsize: int = 2048):
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.maxsize = maxsize

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, vector: np.ndarray):
        self.cache[key] = vector
        self.cache.move_to_end(key)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)


# ═══════════════════════════════════════════════════════════════
# 2. ZERO-HOP HYBRID RAG ENGINE
# ═══════════════════════════════════════════════════════════════
class AsyncHybridRAG:
    def __init__(self, use_reranker: bool = True):
        self.index_json_path = MODELS_DIR / "rag-index" / "dpdp_index.json"
        self.safetensors_path = MODELS_DIR / "rag-index" / "dpdp_embeddings.safetensors"
        self.embed_model_path = MODELS_DIR / "bge-small-en-v1.5"
        self.reranker_model_path = MODELS_DIR / "bge-reranker-v2-m3"
        
        # SOTA FIX: Dynamically scale threads to hardware limits to prevent bottlenecking
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # SOTA FIX: Use BF16 if on Ampere/Ada/Hopper/Blackwell DGX, else FP16/FP32
        self.compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        
        self.use_reranker = use_reranker
        self.is_ready = False
        self.cache = LRUEmbeddingCache(maxsize=2048)
        
        # Exact Stopwords from evaluate_rag.py for 1:1 mathematical parity
        self.stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "of", "and", "in", 
            "to", "for", "with", "on", "at", "by", "from", "as", "that", "this", 
            "it", "be", "or", "which", "will", "would", "could", "should", "their", "they"
        }

    async def initialize(self):
        if not HAS_ML:
            print("⚠️ [RAGEngine] sentence-transformers missing. Ensure DGX environment.")
            return

        print(f"[RAGEngine] Booting Zero-Hop Hybrid Search on {self.device.upper()} ({self.compute_dtype})...")
        await asyncio.to_thread(self._sync_initialize)
        self.is_ready = True
        print("✅ [RAGEngine] Safetensors mapped & Models loaded into Tensor Cores.")

    def _sync_initialize(self):
        # 1. Map JSON Index
        with open(self.index_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.metadatas = data["metadatas"]

        # 2. Map Lexical Index
        tokenized_corpus = [self._tokenize(c) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Map Dense Safetensors (Zero-Copy)
        tensors = load_file(str(self.safetensors_path), backend="mmap")
        raw_dense = tensors["dense_embeddings"]
        self.dense_embeddings = raw_dense / np.linalg.norm(raw_dense, axis=1, keepdims=True)

        # 4. Mount Neural Models to GPU with BF16/FP16 Tensor Core Acceleration
        model_kwargs = {"torch_dtype": self.compute_dtype}
        self.embed_model = SentenceTransformer(str(self.embed_model_path), device=self.device, model_kwargs=model_kwargs)
        
        if self.use_reranker and self.reranker_model_path.exists():
            self.reranker_model = CrossEncoder(str(self.reranker_model_path), max_length=512, device=self.device, model_kwargs=model_kwargs)
        else:
            self.reranker_model = None

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'\w+', str(text).lower())
        return [w for w in words if w not in self.stopwords]

    def _sync_retrieve(self, query: str, top_k: int = 7, retrieval_depth: int = 50, rerank_depth: int = 25, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """Executes Hybrid Math."""
        if not self.is_ready: return []

        # A. Lexical (BM25)
        q_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        top_bm25_idx = fast_top_k(bm25_scores, k=retrieval_depth)

        # B. Dense (BGE)
        cache_key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        q_emb = self.cache.get(cache_key)
        if q_emb is None:
            dense_q = f"Represent this sentence for searching relevant passages: {query}"
            # Normalize embedding outputs to guarantee pure Cosine Similarity via Dot Product
            q_emb = self.embed_model.encode([dense_q], normalize_embeddings=True, show_progress_bar=False)[0]
            self.cache.put(cache_key, q_emb)

        dense_scores = np.dot(self.dense_embeddings, q_emb)
        top_dense_idx = fast_top_k(dense_scores, k=retrieval_depth)

        # C. RRF Merge
        rrf_scores = {}
        for rank, idx in enumerate(top_bm25_idx):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(top_dense_idx):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        top_rrf = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:rerank_depth]

        # D. Cross-Encoder Rerank (GPU Batched)
        if self.reranker_model:
            cross_pairs = [[query, self.chunks[idx]] for idx in top_rrf]
            cross_scores = self.reranker_model.predict(cross_pairs, batch_size=len(cross_pairs), show_progress_bar=False)
            ranked_pairs = sorted(zip(top_rrf, cross_scores), key=lambda x: x[1], reverse=True)[:top_k]
        else:
            ranked_pairs = [(idx, rrf_scores[idx]) for idx in top_rrf[:top_k]]

        # E. Pack Results
        return [
            {
                "chunk": self.chunks[idx],
                "metadata": self.metadatas[idx],
                "score": float(score)
            }
            for idx, score in ranked_pairs
        ]

    async def retrieve_context(self, query: str, top_k: int = 5, confidence_threshold: float = -5.0) -> Tuple[str, List[Dict[str, Any]]]:
        """
        SOTA XML-Structured Context Generation.
        Returns formatted XML prompt and raw citation hits.
        """
        if not self.is_ready:
            return ("<error>System initializing...</error>", [])

        loop = asyncio.get_running_loop()
        hits = await loop.run_in_executor(self.thread_pool, self._sync_retrieve, query, top_k)
        
        if not hits:
            return ("<context>\n  <info>No statutory context found.</info>\n</context>", [])

        # SOTA FIX: Filter out statistically irrelevant chunks to prevent hallucination
        if self.use_reranker:
            hits = [h for h in hits if h["score"] > confidence_threshold]
            if not hits:
                return ("<context>\n  <info>No confident statutory matches found.</info>\n</context>", [])

        # ── SOTA XML PROMPT FORMATTING ──
        xml_blocks = ["<context>"]
        
        for idx, hit in enumerate(hits):
            meta = hit["metadata"]
            section = meta.get("number", "General Provision")
            applies_to = meta.get("applies_to", "All Fiduciaries")
            
            # XML structurally isolates the text so the LLM perfectly associates the Law with the Section Code
            block = (
                f'  <document id="{idx+1}">\n'
                f'    <metadata>\n'
                f'      <section>{section}</section>\n'
                f'      <applies_to>{applies_to}</applies_to>\n'
                f'    </metadata>\n'
                f'    <text>{hit["chunk"]}</text>\n'
                f'  </document>'
            )
            xml_blocks.append(block)
            
        xml_blocks.append("</context>")
        formatted_context = "\n".join(xml_blocks)
        
        return formatted_context, hits


# Global Singleton (Defaults to Reranker Enabled for SOTA Accuracy)
rag_engine = AsyncHybridRAG(use_reranker=True)