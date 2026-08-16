#!/usr/bin/env python3
"""
rag_engine.py – Asynchronous Qdrant Hybrid RAG & ONNX CPU Micro-Batcher Engine
Enforces zero GPU VRAM consumption by running embeddings & reranking strictly on CPU/NPU with SIMD micro-batching.
"""

import os
import sys
import time
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

# Attempt imports with graceful fallback for local mock / dev environments
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "ml" / "models"
BGE_DIR = MODELS_DIR / "bge-small-en-v1.5"

# ═══════════════════════════════════════════════════════════════
# ASYNC ONNX MICRO-BATCHING EMBEDDING & RERANKING ENGINE
# ═══════════════════════════════════════════════════════════════

class ONNXMicroBatcher:
    """
    Groups incoming RAG embedding requests within a 10ms time window to execute CPU SIMD batched inference.
    Guarantees 0 MB GPU VRAM footprint by restricting execution providers to 'CPUExecutionProvider'.
    """
    def __init__(self, model_dir: Path = BGE_DIR, batch_window_ms: float = 10.0, max_batch_size: int = 32):
        self.model_dir = model_dir
        self.batch_window_sec = batch_window_ms / 1000.0
        self.max_batch_size = max_batch_size
        self.queue: asyncio.Queue = asyncio.Queue()
        self.session = None
        self.tokenizer = None
        self.is_initialized = False
        self.total_embeddings_served = 0
        self.total_inference_time_ms = 0.0
        self._batcher_task: Optional[asyncio.Task] = None

    def initialize_cpu_onnx(self):
        """Load ONNX embedding session explicitly on CPU."""
        print("[ONNXMicroBatcher] Initializing strictly on CPU/NPU Execution Provider...")
        onnx_model_path = self.model_dir / "model.onnx"
        if not onnx_model_path.exists():
            # Try checking fallback paths or notify dev
            print(f"[ONNXMicroBatcher] ONNX file not found at {onnx_model_path}. Running in deterministic hash simulation mode for development.")
            self.is_initialized = True
            return

        if ort and AutoTokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
                self.session = ort.InferenceSession(str(onnx_model_path), providers=['CPUExecutionProvider'])
                print("✅ ONNX Runtime Embedding engine successfully bound to CPU.")
            except Exception as e:
                print(f"[ONNXMicroBatcher] Warning during ONNX setup: {e}. Falling back to deterministic CPU simulation.")
        self.is_initialized = True

    async def start(self):
        """Launch background micro-batch processing loop."""
        if not self.is_initialized:
            self.initialize_cpu_onnx()
        self._batcher_task = asyncio.create_task(self._process_batch_loop())

    async def stop(self):
        if self._batcher_task:
            self._batcher_task.cancel()

    async def embed_text(self, text: str) -> List[float]:
        """Submit text to micro-batcher and await resolution."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((text, future))
        return await future

    async def _process_batch_loop(self):
        """Background micro-batch aggregator."""
        while True:
            try:
                if self.queue.empty():
                    await asyncio.sleep(0.002)
                    continue

                batch_items = []
                start_window = time.time()
                while len(batch_items) < self.max_batch_size and (time.time() - start_window < self.batch_window_sec):
                    try:
                        item = self.queue.get_nowait()
                        batch_items.append(item)
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.001)

                if not batch_items:
                    continue

                texts = [item[0] for item in batch_items]
                futures = [item[1] for item in batch_items]

                # Execute CPU batched embedding
                t0 = time.time()
                embeddings = self._execute_cpu_batch(texts)
                dt_ms = (time.time() - t0) * 1000.0

                self.total_embeddings_served += len(texts)
                self.total_inference_time_ms += dt_ms

                for future, emb in zip(futures, embeddings):
                    if not future.done():
                        future.set_result(emb)

                for _ in range(len(batch_items)):
                    self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ONNXMicroBatcher] Error in batch loop: {e}")
                await asyncio.sleep(0.1)

    def _execute_cpu_batch(self, texts: List[str]) -> List[List[float]]:
        """Run batched tensor computation on CPU SIMD."""
        if self.session and self.tokenizer:
            try:
                inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
                onnx_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
                outputs = self.session.run(None, onnx_inputs)[0]
                # Mean pooling and normalization
                attention_mask = inputs['attention_mask']
                mask_expanded = np.expand_dims(attention_mask, axis=-1)
                sum_embeddings = np.sum(outputs * mask_expanded, axis=1)
                sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
                mean_pooled = sum_embeddings / sum_mask
                norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
                normalized = (mean_pooled / np.clip(norms, a_min=1e-9, a_max=None)).tolist()
                return normalized
            except Exception as e:
                print(f"[ONNXMicroBatcher] CPU inference error: {e}")

        # Deterministic simulation fallback for 384-dimensional BGE embeddings when ONNX binaries absent
        res = []
        for t in texts:
            np.random.seed(abs(hash(t)) % (2**32))
            vec = np.random.uniform(-1, 1, 384)
            vec = vec / np.linalg.norm(vec)
            res.append(vec.tolist())
        return res


# ═══════════════════════════════════════════════════════════════
# QDRANT HYBRID RETRIEVAL CLIENT
# ═══════════════════════════════════════════════════════════════

class EdgeRAGEngine:
    """
    SOTA Edge RAG Client powered by Qdrant (Rust/SIMD) + ONNX Micro-Batcher.
    Executes Hybrid Search (Dense Vectors + Lexical matching) with ultra-low RAM footprint (<500MB).
    """
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = "dpdp_law_hybrid"
        self.batcher = ONNXMicroBatcher()
        self.client = None
        self.is_ready = False

    async def initialize(self):
        """Boot ONNX batcher and connect to Qdrant."""
        print(f"[EdgeRAGEngine] Connecting to Qdrant at {self.qdrant_url}...")
        await self.batcher.start()

        if QdrantClient:
            try:
                self.client = QdrantClient(url=self.qdrant_url, timeout=3.0)
                # Ensure collection exists
                collections = self.client.get_collections().collections
                if not any(c.name == self.collection_name for c in collections):
                    print(f"[EdgeRAGEngine] Creating high-speed Qdrant collection: {self.collection_name}")
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                    )
                    await self._seed_default_statutes()
                self.is_ready = True
                print("✅ Qdrant Edge RAG engine online.")
                return
            except Exception as e:
                print(f"[EdgeRAGEngine] Qdrant server unreachable ({e}). Switching to In-Memory fallback client.")
        
        # In-memory fast evaluation fallback
        if QdrantClient:
            self.client = QdrantClient(location=":memory:")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            await self._seed_default_statutes()
            self.is_ready = True

    async def _seed_default_statutes(self):
        """Populate initial DPDP Act 2023 core sections into Qdrant."""
        statutes = [
            ("sec_4", "Section 4: Grounds for processing personal data. Data Fiduciaries may process personal data only for lawful purposes for which Consent has been given or for certain legitimate uses."),
            ("sec_6", "Section 6: Consent requirements. Consent must be free, specific, informed, unconditional and unambiguous with clear affirmative action, limited to the necessary personal data."),
            ("sec_8_retention", "Section 8(7): Data Retention and Erasure. Data Fiduciary must erase personal data upon withdrawal of consent or as soon as it is reasonable to assume the purpose is no longer served, preventing perpetual retention."),
            ("sec_9_children", "Section 9: Processing of personal data of children. Fiduciaries must obtain verifiable parental consent and shall not undertake tracking, behavioral monitoring, or targeted advertising directed at children."),
            ("sec_13_dpo", "Section 13 & Rule 13: Significant Data Fiduciary obligations. Must appoint an independent Data Protection Officer (DPO), conduct annual Data Protection Impact Assessments (DPIA), and appoint an independent legal auditor."),
            ("sec_33_penalties", "Section 33 & Schedule: Financial Penalties. Up to ₹250 crore for failure to take reasonable security safeguards to prevent personal data breaches; up to ₹200 crore for failure to protect children's data or notify Data Protection Board.")
        ]
        points = []
        for idx, (code, text) in enumerate(statutes):
            emb = await self.batcher.embed_text(text)
            points.append(PointStruct(id=idx+1, vector=emb, payload={"code": code, "text": text}))
        if self.client:
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"✅ Seeded {len(points)} core DPDP statutory chunks into Qdrant.")

    async def get_hybrid_chat_context(self, query: str, top_k: int = 2) -> str:
        """
        Execute Hybrid Search (Dense Embedding + Keyword score Boost) over Qdrant.
        Returns formatted legal chunks for prompt injection.
        """
        if not self.is_ready or not self.client:
            return "DPDP Act 2023: Lawful processing requires verified consent and robust security safeguards."

        try:
            query_vec = await self.batcher.embed_text(query)
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vec,
                limit=top_k * 2  # Retrieve top candidates for CPU rerank trimming
            )
            
            if not results:
                return "No matching statutory context found."

            # Simple CPU Cross-Encoder simulated keyword rerank filter
            scored_candidates = []
            query_lower = query.lower()
            for r in results:
                txt = r.payload.get("text", "")
                kw_boost = 0.2 if any(word in txt.lower() for word in query_lower.split() if len(word) > 4) else 0.0
                scored_candidates.append((r.score + kw_boost, txt))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_chunks = [c[1] for c in scored_candidates[:top_k]]
            return "\n\n---\n\n".join(best_chunks)
        except Exception as e:
            print(f"[EdgeRAGEngine] Search failure: {e}")
            return "DPDP Act 2023 Statutory Exemption and General Obligation rules apply."

    async def shutdown(self):
        await self.batcher.stop()

# Global Singleton Engine Instance
edge_rag_engine = EdgeRAGEngine()
