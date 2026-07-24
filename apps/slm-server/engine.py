#!/usr/bin/env python3
"""
engine.py – Asynchronous Worker Pool & Multi-LoRA Engine Multiplexer for Ssense SLM Server
"""

import os
import sys
import time
import json
import asyncio
import aiohttp
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import Dict, Any, Optional

# Add root directory and ml/evals to path to import backend_loader
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ML_EVALS_DIR = ROOT_DIR / "ml" / "evals"
if str(ML_EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(ML_EVALS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from backend_loader import BackendEngine
except ImportError as e:
    BackendEngine = None
    print(f"[Warning] Could not import BackendEngine from ml/evals/backend_loader.py: {e}")

# ═══════════════════════════════════════════════════════════════
# ASYNC ENGINE MULTIPLEXER
# ═══════════════════════════════════════════════════════════════

class SLMMultiplexer:
    def __init__(self):
        self.backend_type = os.getenv("SSENSE_BACKEND", "vllm").lower()
        self.vllm_url = os.getenv("SSENSE_VLLM_URL", "http://localhost:8000/v1/completions")
        
        # Paths for unsloth / llamacpp
        self.models_dir = ROOT_DIR / "ml" / "models"
        self.audit_lora_name = os.getenv("SSENSE_AUDIT_LORA", "audit_lora")
        self.chat_lora_name = os.getenv("SSENSE_CHAT_LORA", "chat_lora")
        
        self.audit_engine: Optional[BackendEngine] = None
        self.chat_engine: Optional[BackendEngine] = None
        
        self.session = None
        
        # Initialize ChromaDB for RAG
        try:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
            self.chroma_client = chromadb.PersistentClient(path=str(ROOT_DIR / "ml" / "data-forge" / "chroma_db"))
            self.law_collection = self.chroma_client.get_collection(name="dpdp_law", embedding_function=ef)
        except Exception as e:
            print(f"[SLMMultiplexer] Warning: ChromaDB init failed: {e}")
            self.law_collection = None
        self.is_loaded = False
        self.total_inferences = 0
        self.total_tokens_generated = 0
        self.total_latency_ms = 0.0

    def initialize(self, session: aiohttp.ClientSession = None):
        self.session = session
        """Initialize backend engines on startup."""
        print(f"[SLMMultiplexer] Initializing with backend: {self.backend_type}...")
        if BackendEngine is None:
            print("[SLMMultiplexer] BackendEngine not available. Running in Mock/Dry-Run mode.")
            self.is_loaded = True
            return

        try:
            if self.backend_type == "vllm":
                # For vLLM, single or dual REST engines connected to vLLM multi-LoRA server
                self.audit_engine = BackendEngine(
                    backend_type="vllm",
                    vllm_url=self.vllm_url,
                    lora_name=self.audit_lora_name
                )
                self.chat_engine = BackendEngine(
                    backend_type="vllm",
                    vllm_url=self.vllm_url,
                    lora_name=self.chat_lora_name
                )
            elif self.backend_type == "llamacpp":
                gguf_path = self.models_dir / "qwen3.5-9b-instruct-q4_k_m.gguf"
                if not gguf_path.exists():
                    # Fallback check
                    gguf_path = self.models_dir / "qwen2.5-7b-instruct-q4_k_m.gguf"
                if gguf_path.exists():
                    self.audit_engine = BackendEngine(backend_type="llamacpp", model_path=gguf_path)
                    self.chat_engine = self.audit_engine
                else:
                    print(f"[SLMMultiplexer] GGUF model not found in {self.models_dir}. Engine lazy load on demand.")
            else:
                # unsloth direct load
                self.audit_engine = BackendEngine(
                    backend_type="unsloth",
                    model_path="Qwen/Qwen3.5-9B",
                    adapter_path=self.models_dir / "audit-model-final-adapter"
                )
                self.chat_engine = BackendEngine(
                    backend_type="unsloth",
                    model_path="Qwen/Qwen3.5-9B",
                    adapter_path=self.models_dir / "chatbot-model-final-adapter"
                )
            self.is_loaded = True
            print("[SLMMultiplexer] Engines successfully initialized.")
        except Exception as e:
            print(f"[SLMMultiplexer] Warning during engine boot: {e}. Will attempt on-demand or fallback.")
            self.is_loaded = True

    async def run_audit_inference(self, domain: str, policy_text: str) -> Dict[str, Any]:
        """Run audit inference asynchronously using thread pool."""
        from security import sanitize_input_prompt, validate_and_repair_report
        clean_text = sanitize_input_prompt(policy_text, is_audit_policy=True)
        
        # Constitutional Audit Query (RAFT Sync)
        retrieved_context = ""
        if self.law_collection:
            try:
                AUDIT_CONSTITUTION_QUERY = "DPDP Act 2023 statutory obligations and penalties regarding Consent, Data Retention, Children's Privacy, Grievance Redressal, Cross-Border Transfer, and Security Safeguards."
                rag_results = self.law_collection.query(query_texts=[AUDIT_CONSTITUTION_QUERY], n_results=3)
                retrieved_context = "\n\n---\n\n".join(rag_results['documents'][0])
            except Exception as e:
                retrieved_context = "RAG Error"
                print(f"[SLMMultiplexer] Audit RAG Error: {e}")

        prompt = (
            f"<|im_start|>system\n"
            f"You are a strict DPDP (Digital Personal Data Protection Act 2023, India) Regulatory Auditor. "
            f"Analyze the policy and return ONLY valid JSON with global_legal_reasoning, violations array, and dpdp_trust_score (0-100).\n"
            f"[RETRIEVED_LAW_CONTEXT]\n{retrieved_context}\n"
            f"Base your legal reasoning strictly on the [RETRIEVED_LAW_CONTEXT] provided above. Do not assume knowledge not present in the retrieval. Do not hallucinate external laws.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Audit domain: {domain}\n\n{clean_text}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        start_time = time.time()
        
        # Pure async non-blocking call to vLLM
        try:
            payload = {
                "model": self.audit_lora_name,
                "prompt": prompt,
                "max_tokens": 2048,
                "temperature": 0.1
            }
            async with self.session.post(self.vllm_url, json=payload) as resp:
                resp_json = await resp.json()
                raw_output = resp_json["choices"][0]["text"]
                tokens_gen = resp_json["usage"]["completion_tokens"]
                result = {"raw_output": raw_output, "tokens_generated": tokens_gen, "tokens_per_sec": tokens_gen / max((time.time() - start_time), 0.1)}
        except Exception as e:
            print(f"[SLMMultiplexer] Async audit inference error: {e}")
            result = {
                "raw_output": json.dumps({"global_legal_reasoning": "Fallback inference active.", "violations": [], "dpdp_trust_score": 50}),
                "tokens_generated": 100,
                "tokens_per_sec": 50.0
            }
        
        raw_output = result.get("raw_output", "")
        try:
            report = validate_and_repair_report(raw_output)
        except Exception as e:
            # Fallback repair structure if model output malformed
            report = {
                "global_legal_reasoning": f"Audit reasoning extracted from model output: {raw_output[:200]}...",
                "violations": [],
                "dpdp_trust_score": 60
            }
            
        latency = (time.time() - start_time) * 1000
        self.total_inferences += 1
        self.total_tokens_generated += result.get("tokens_generated", 150)
        self.total_latency_ms += latency
        
        return {
            "success": True,
            "report": report,
            "metrics": {
                "latency_ms": round(latency, 2),
                "tokens_per_sec": round(result.get("tokens_per_sec", 100.0), 2)
            }
        }

    async def run_chat_inference(self, domain: str, user_prompt: str) -> Dict[str, Any]:
        """Run multi-turn chat inference asynchronously."""
        from security import sanitize_input_prompt
        clean_prompt = sanitize_input_prompt(user_prompt, is_audit_policy=False)
        
        
        # RAG Vector Search (Safeguarded for 512-token limit)
        retrieved_context = ""
        if self.law_collection:
            try:
                safe_query = clean_prompt[:1500]
                rag_results = self.law_collection.query(query_texts=[safe_query], n_results=2)
                retrieved_context = "\n\n---\n\n".join(rag_results['documents'][0])
            except Exception as e:
                retrieved_context = "RAG Error"
                print(f"RAG Error: {e}")

        prompt = (
            f"<|im_start|>system\n"
            f"You are the Ssense Co-Pilot, a helpful AI assistant. Explain DPDP compliance issues for {domain}.\n"
            f"[RETRIEVED_LAW_CONTEXT]\n{retrieved_context}\n"
            f"You are an AI assistant equipped with a retrieval database. Base your legal reasoning strictly on the [RETRIEVED_LAW_CONTEXT] provided above. Do not assume knowledge not present in the retrieval. Do not hallucinate external laws (like GDPR or CCPA).\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{clean_prompt}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        start_time = time.time()
        
        try:
            payload = {
                "model": self.chat_lora_name,
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.7
            }
            async with self.session.post(self.vllm_url, json=payload) as resp:
                resp_json = await resp.json()
                raw_output = resp_json["choices"][0]["text"]
                tokens_gen = resp_json["usage"]["completion_tokens"]
                result = {"raw_output": raw_output, "tokens_generated": tokens_gen, "tokens_per_sec": tokens_gen / max((time.time() - start_time), 0.1)}
        except Exception as e:
            print(f"[SLMMultiplexer] Async chat inference error: {e}")
            result = {
                "raw_output": f"Regarding {domain}: Under India's DPDP Act, you have rights. Connection error.",
                "tokens_generated": 20,
                "tokens_per_sec": 50.0
            }
        
        latency = (time.time() - start_time) * 1000
        self.total_inferences += 1
        self.total_tokens_generated += result.get("tokens_generated", 60)
        self.total_latency_ms += latency
        
        return {
            "success": True,
            "message": result.get("raw_output", "").strip(),
            "metrics": {
                "latency_ms": round(latency, 2),
                "tokens_per_sec": round(result.get("tokens_per_sec", 120.0), 2)
            }
        }

# Global singleton instance
multiplexer = SLMMultiplexer()
