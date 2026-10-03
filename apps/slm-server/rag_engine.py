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
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import re
import json
import asyncio
import hashlib
import threading
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
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT_DIR / "ml" / "models")))


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
    @staticmethod
    def _select_device() -> str:
        """Headroom-aware GPU/CPU placement for the RAG embedder + reranker.

        Deployment targets for this server are: a CPU-only server (no
        decision to make — CUDA isn't even available), and Jetson-class
        unified-memory boards ranging from a tightly-capped ~32GB Orin up to
        a much larger unified-memory box like an AGX Spark-class device
        (up to ~128GB shared between CPU and GPU). Those two ends of the
        Jetson range have very different amounts of memory headroom left
        over after vLLM claims its budget, so a single static default
        (always CPU, or always GPU) is wrong for one end or the other:
          - Always-CPU leaves real latency on the table on a high-memory
            unified box, where the GPU's Tensor Cores are sitting mostly
            idle relative to vLLM's own budget and a few hundred MB for the
            embedder/reranker genuinely would not be missed.
          - Always-GPU risks the boot-order VRAM problem on a tightly-capped
            board: RAGEngine.initialize() runs BEFORE the vLLM engine boots
            (see main.py's lifespan), and engine.py's own utilization math
            assumes it gets to claim its whole configured budget — if these
            models grab CUDA memory first on a board with little headroom,
            vLLM's already-computed fraction can exceed what's actually
            still free.

        So: estimate how much of the unified/VRAM pool vLLM is going to
        claim (mirroring engine.py's own logic, conservatively — see below),
        and only place RAG on the GPU if genuine headroom remains after
        that. This makes an AGX Spark-class box (huge unified pool, vLLM
        self-capped at TARGET_TOTAL_MEMORY_GB) auto-select GPU, and a
        tightly-capped ~32GB Jetson auto-select CPU — the same code path
        handling both ends of the same product line correctly, with no
        manual per-board tuning required.

        SSENSE_RAG_DEVICE=cpu|cuda always overrides this entirely, for
        anyone who wants to force one or the other regardless of the
        estimate (e.g. after benchmarking your specific traffic shape).
        """
        requested = os.getenv("SSENSE_RAG_DEVICE", "").strip().lower()
        if requested in ("cpu", "cuda"):
            if requested == "cuda" and not torch.cuda.is_available():
                print("⚠️ [RAGEngine] SSENSE_RAG_DEVICE=cuda requested but no CUDA device is available; using CPU.")
                return "cpu"
            print(f"ℹ️ [RAGEngine] SSENSE_RAG_DEVICE explicitly set to '{requested}' — honoring override.")
            return requested

        if not torch.cuda.is_available():
            return "cpu"  # CPU-only server profile — nothing else to decide.

        try:
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            target_mem_gb = float(os.getenv("SSENSE_TARGET_MEMORY_GB", "32.0"))
            # Conservative upper-bound estimate of what vLLM will claim — a
            # safe overestimate for BOTH engine.py code paths (the discrete/
            # gpu profile claims min(target, total)*~0.9-ish of the card;
            # the jetson profile claims min(target, total*0.6), which is
            # always <= this estimate) so this headroom check errs toward
            # "assume less room than there really is" rather than the
            # reverse.
            assumed_engine_claim_gb = min(target_mem_gb, total_mem_gb * 0.9)
            headroom_gb = total_mem_gb - assumed_engine_claim_gb
            min_headroom_gb = float(os.getenv("SSENSE_RAG_MIN_HEADROOM_GB", "8.0"))

            if headroom_gb >= min_headroom_gb:
                print(f"ℹ️ [RAGEngine] {total_mem_gb:.0f}GB unified/VRAM pool, vLLM budgeted to claim "
                      f"~{assumed_engine_claim_gb:.0f}GB → ~{headroom_gb:.0f}GB headroom "
                      f"(≥{min_headroom_gb:.0f}GB threshold) — placing RAG models on GPU.")
                return "cuda"
            print(f"ℹ️ [RAGEngine] {total_mem_gb:.0f}GB unified/VRAM pool, vLLM budgeted to claim "
                  f"~{assumed_engine_claim_gb:.0f}GB → only ~{headroom_gb:.0f}GB headroom "
                  f"(<{min_headroom_gb:.0f}GB threshold) — placing RAG models on CPU to protect "
                  f"vLLM's memory budget. Override with SSENSE_RAG_DEVICE=cuda if you've verified "
                  f"this is safe for your deployment.")
            return "cpu"
        except Exception as e:
            print(f"⚠️ [RAGEngine] Could not probe GPU memory for device auto-selection ({e}); defaulting to CPU (safe).")
            return "cpu"

    def __init__(self, use_reranker: bool = True):
        self.index_json_path = MODELS_DIR / "rag-index" / "dpdp_index.json"
        if not self.index_json_path.exists():
            alt_json = MODELS_DIR / "dpdp_index.json"
            if alt_json.exists():
                self.index_json_path = alt_json

        self.safetensors_path = MODELS_DIR / "rag-index" / "dpdp_embeddings.safetensors"
        if not self.safetensors_path.exists():
            alt_safe = MODELS_DIR / "dpdp_embeddings.safetensors"
            if alt_safe.exists():
                self.safetensors_path = alt_safe

        self.embed_model_path = MODELS_DIR / "bge-small-en-v1.5"
        self.reranker_model_path = MODELS_DIR / "bge-reranker-v2-m3"
        
        # Dynamically scale threads to hardware limits
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # GPU lock: self.embed_model and self.reranker_model are shared instances
        # on a single CUDA stream. PyTorch forward() is not thread-safe across OS
        # threads without explicit synchronisation. A single lock serialises GPU
        # calls while leaving CPU preprocessing (BM25, tokenise, RRF) parallel.
        self._gpu_lock = threading.Lock()
        
        self.device = self._select_device()

        # SOTA FIX: Use BF16/FP16 on GPU (Tensor Cores), FP32 on CPU (prevent PyTorch Half CPU kernels missing)
        if self.device == "cuda":
            self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.compute_dtype = torch.float32
        
        self.use_reranker = use_reranker
        self.is_ready = False
        self.cache = LRUEmbeddingCache(maxsize=2048)
        
        # Extended Stopwords for 1:1 mathematical parity and noise rejection
        self.stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "of", "and", "in", 
            "to", "for", "with", "on", "at", "by", "from", "as", "that", "this", 
            "it", "be", "or", "which", "will", "would", "could", "should", "their", "they",
            "what", "who", "whom", "whose", "when", "where", "why", "how",
            "there", "here", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "can", "just", "now",
            "i", "me", "my", "myself", "we", "us", "our", "ours", "you", "your", "yours",
            "he", "him", "his", "she", "her", "hers", "them", "tell", "give", "show", "please", "help"
        }

    async def initialize(self):
        if not HAS_ML:
            print("⚠️ [RAGEngine] sentence-transformers missing. Ensure DGX environment.")
            return

        print(f"[RAGEngine] Booting Zero-Hop Hybrid Search on {self.device.upper()} ({self.compute_dtype})...")
        await asyncio.to_thread(self._sync_initialize)
        if self.is_ready:
            print(f"✅ [RAGEngine] Safetensors mapped & Models loaded into {self.device.upper()}.")
        else:
            print("⚠️ [RAGEngine] Initialized in fallback mode (indices/models missing).")

    def _sync_initialize(self):
        if not self.index_json_path.exists() or not self.safetensors_path.exists():
            print(f"⚠️ [RAGEngine] Index files missing at {self.index_json_path} or {self.safetensors_path}. RAG disabled until indexed.")
            return

        # 1. Map JSON Index
        with open(self.index_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data.get("chunks", [])
        self.metadatas = data.get("metadatas", [])

        # 2. Map Lexical Index
        tokenized_corpus = [self._tokenize(c) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        # 3. Map Dense Safetensors (Zero-Copy)
        tensors = load_file(str(self.safetensors_path), backend="mmap")
        raw_dense = tensors["dense_embeddings"]
        self.dense_embeddings = raw_dense / np.linalg.norm(raw_dense, axis=1, keepdims=True)

        # 4. Mount Neural Models to GPU with BF16/FP16 Tensor Core Acceleration
        model_kwargs = {"torch_dtype": self.compute_dtype}
        embed_target = str(self.embed_model_path) if self.embed_model_path.exists() else "BAAI/bge-small-en-v1.5"
        try:
            self.embed_model = SentenceTransformer(embed_target, device=self.device, model_kwargs=model_kwargs)
        except TypeError:
            self.embed_model = SentenceTransformer(embed_target, device=self.device)

        if self.use_reranker:
            reranker_target = str(self.reranker_model_path) if self.reranker_model_path.exists() else "BAAI/bge-reranker-v2-m3"
            try:
                self.reranker_model = CrossEncoder(reranker_target, max_length=512, device=self.device, model_kwargs=model_kwargs)
            except TypeError:
                try:
                    self.reranker_model = CrossEncoder(reranker_target, max_length=512, device=self.device)
                except Exception as e:
                    print(f"⚠️  [RAGEngine] Failed to load reranker {reranker_target}: {e}")
                    self.reranker_model = None
            except Exception as e:
                print(f"⚠️  [RAGEngine] Failed to load reranker {reranker_target}: {e}")
                self.reranker_model = None
        else:
            self.reranker_model = None

        self.is_ready = True

    # ── State-query intent detection ──────────────────────────────────────────
    # Fixes the bug from the Rust daemon where is_state_query was hardcoded
    # False at its only call-site, permanently excluding state-specific DPDP
    # provisions (e.g. "Does Rajasthan have additional data rules?") from every
    # single retrieval.  This lightweight keyword matcher mirrors the intent
    # detector now in local_engine.rs so both inference paths behave the same.
    _STATE_KEYWORDS = re.compile(
        r"\b(state|rajasthan|maharashtra|karnataka|delhi|telangana|andhra|gujarat|"
        r"bihar|punjab|haryana|tamil\s*nadu|west\s*bengal|uttar\s*pradesh|"
        r"provincial|regional|local\s+law|local\s+regulation|state-level|"
        r"jurisdiction|territory|union\s+territory)\b",
        re.I,
    )

    def _detect_state_query(self, query: str) -> bool:
        return bool(self._STATE_KEYWORDS.search(query))

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'\w+', str(text).lower())
        return [w for w in words if w not in self.stopwords]

    def _sync_retrieve(
        self,
        query: str,
        top_k: int = 7,
        retrieval_depth: int = 50,
        rerank_depth: int = 25,
        rrf_k: int = 60,
        is_state_query: bool = False,
    ) -> List[Dict[str, Any]]:
        """Hybrid BM25 + dense retrieval with RRF fusion and cross-encoder reranking."""
        if not self.is_ready:
            return []

        # A. Lexical (BM25)
        q_tokens = self._tokenize(query)
        if self.bm25 and q_tokens:
            bm25_scores  = self.bm25.get_scores(q_tokens)
            top_bm25_idx = fast_top_k(bm25_scores, k=retrieval_depth)
        else:
            bm25_scores  = np.zeros(len(self.chunks))
            top_bm25_idx = np.array([], dtype=int)

        # B. Dense (BGE cosine) — GPU lock guards the shared embed_model instance
        cache_key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        q_emb = self.cache.get(cache_key)
        if q_emb is None:
            dense_q = f"Represent this sentence for searching relevant passages: {query}"
            with self._gpu_lock:
                q_emb = self.embed_model.encode(
                    [dense_q], normalize_embeddings=True, show_progress_bar=False
                )[0]
            self.cache.put(cache_key, q_emb)

        dense_scores  = np.dot(self.dense_embeddings, q_emb)
        top_dense_idx = fast_top_k(dense_scores, k=retrieval_depth)

        # Relevance guard: require either strong semantic grounding (dense >= 0.60)
        # or moderate semantic grounding (dense >= 0.52) alongside lexical confirmation (bm25 >= 4.5).
        # Out-of-domain, adversarial, or ungrounded queries return empty to trigger RAFT refusal.
        max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) > 0 else 0.0
        max_dense = float(np.max(dense_scores)) if len(dense_scores) > 0 else 0.0
        if max_dense < 0.52 or (max_bm25 < 4.5 and max_dense < 0.60):
            return []

        # C. RRF merge with pre-filter state-isolation.
        #
        # CRITICAL FIX (Atigravity review — Defect 1):
        # The previous implementation accumulated ALL chunks into rrf_scores first,
        # then applied a post-hoc set filter. This was wrong in two ways:
        #
        #   (a) For non-state queries (is_state_query=False): no filtering happened
        #       at all — state-only chunks leaked into every commercial-site audit,
        #       violating Pillar 18 (jurisdictional_contamination_rate <= 0.0%).
        #
        #   (b) The post-hoc filter checked `applies_to not in ("all fiduciaries",
        #       "all", "")` — but build_vector_db.py stores "all" for general
        #       provisions and "state" for state-only provisions. The string
        #       "all fiduciaries" never appears in any metadata record, so the
        #       filter silently matched nothing.
        #
        # Fix: mirror build_vector_db.py exactly — exclude chunks whose
        # applies_to == "state" at accumulation time when is_state_query is False.
        # State queries receive all chunks (both general and state-specific).
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(top_bm25_idx):
            if bm25_scores[idx] <= 0.0:
                continue  # Never award RRF credit to non-matching documents
            if not is_state_query and self.metadatas[idx].get("applies_to") == "state":
                continue  # banned from private/commercial query candidate pool
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, idx in enumerate(top_dense_idx):
            if not is_state_query and self.metadatas[idx].get("applies_to") == "state":
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        if not rrf_scores:
            return []

        top_rrf = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:rerank_depth]

        # E. Cross-encoder rerank — GPU lock guards the shared reranker instance
        if self.reranker_model:
            pairs = [[query, self.chunks[idx]] for idx in top_rrf]
            with self._gpu_lock:
                cross_scores = self.reranker_model.predict(
                    pairs, batch_size=len(pairs), show_progress_bar=False
                )
            ranked = sorted(zip(top_rrf, cross_scores), key=lambda x: x[1], reverse=True)[:top_k]
        else:
            ranked = [(idx, rrf_scores[idx]) for idx in top_rrf[:top_k]]

        return [
            {"chunk": self.chunks[idx], "metadata": self.metadatas[idx], "score": float(sc)}
            for idx, sc in ranked
        ]

    async def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        confidence_threshold: float = -5.0,
        rerank_depth: int = 25,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Hybrid retrieval with automatic state-query detection.

        Returns (context_str, hits) where context_str uses the plain-text
        [STATUTORY CONTEXT]: block that exactly matches the format used in
        train_chatbot.py and run_chatbot_evals.py — the format the fine-tuned
        Qwen2.5-7B model was aligned to attend to.

        Returns ("", []) on no confident hits. An empty context string triggers
        the RAFT refusal phrase ("the provided context does not contain...") that
        was trained into the model, instead of a pseudo-XML stub that confuses
        attention heads and risks schema-bleed (Pillar 8).
        """
        if not self.is_ready or not query or not re.search(r'\w', query):
            return "", []

        is_state = self._detect_state_query(query)
        loop     = asyncio.get_running_loop()
        hits     = await loop.run_in_executor(
            self.thread_pool, self._sync_retrieve, query, top_k,
            50, rerank_depth, 60, is_state,
        )

        if not hits:
            return "", []

        # CRITICAL FIX (Atigravity review — Defect 3b):
        # Only apply the confidence threshold when self.reranker_model is an
        # actual CrossEncoder (logit scores, range ~-10 to +10).
        # When the reranker fails to mount, _sync_retrieve falls back to RRF
        # scores (~0.016-0.033). Those are always > -5.0, making the threshold
        # check semantically meaningless — it never prunes anything, silently
        # masking the model-load failure.
        if self.reranker_model is not None:
            hits = [h for h in hits if h["score"] > confidence_threshold]
            if not hits:
                return "", []

        # Deduplicate identical or near-identical text passages across hits
        seen_chunks = set()
        deduped_hits = []
        for h in hits:
            norm_c = " ".join(h["chunk"].split())
            if norm_c not in seen_chunks:
                seen_chunks.add(norm_c)
                deduped_hits.append(h)
        hits = deduped_hits
        if not hits:
            return "", []

        # Plain-text [STATUTORY CONTEXT]: block matching model training distribution
        formatted_chunks = [h["chunk"].strip() for h in hits]
        context_str = "[STATUTORY CONTEXT]:\n" + "\n\n".join(formatted_chunks)
        return context_str, hits


# Global Singleton (Auto-detects reranker model presence; falls back to fast RRF)
_default_use_reranker = (
    os.getenv("SSENSE_USE_RERANKER", "true" if (MODELS_DIR / "bge-reranker-v2-m3").exists() else "false")
    .strip()
    .lower() in ("1", "true")
)
rag_engine = AsyncHybridRAG(use_reranker=_default_use_reranker)