#!/usr/bin/env python3
"""
engine.py – Asynchronous Worker Pool & Multi-LoRA Engine Multiplexer for Ssense SLM Server
Upgraded for 48GB VRAM Edge environments: SSE Streaming, O(1) Redis Audit lookup, and CPU ONNX RAG.
"""

import os
import sys
import time
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional, AsyncGenerator
from pathlib import Path

# Local edge engine imports
from redis_queue import redis_queue
from rag_engine import edge_rag_engine

# ═══════════════════════════════════════════════════════════════
# ASYNC ENGINE MULTIPLEXER (SSE STREAMING)
# ═══════════════════════════════════════════════════════════════

class SLMMultiplexer:
    def __init__(self):
        self.vllm_url = os.getenv("SSENSE_VLLM_URL", "http://localhost:8000/v1/completions")
        self.vllm_stream_url = self.vllm_url
        self.audit_lora_name = os.getenv("SSENSE_AUDIT_LORA", "audit_lora")
        self.chat_lora_name = os.getenv("SSENSE_CHAT_LORA", "chat_lora")
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_loaded = False
        
        # Telemetry Metrics
        self.total_inferences = 0
        self.total_tokens_generated = 0
        self.total_latency_ms = 0.0

    async def initialize(self, session: aiohttp.ClientSession):
        """Initialize backend engines on startup."""
        self.session = session
        print("[SLMMultiplexer] Booting RAG & Redis subsystem connections...")
        await redis_queue.initialize()
        await edge_rag_engine.initialize()
        self.is_loaded = True
        print("✅ SLMMultiplexer Edge engine successfully initialized.")

    async def _stream_vllm_generator(self, payload: dict) -> AsyncGenerator[str, None]:
        """Core streaming generator consuming vLLM chunked responses."""
        if not self.session:
            yield "data: " + json.dumps({"status": "error", "message": "Server session offline"}) + "\n\n"
            return

        try:
            async with self.session.post(self.vllm_stream_url, json=payload) as resp:
                if resp.status != 200:
                    yield "data: " + json.dumps({"status": "error", "message": f"vLLM engine returned HTTP {resp.status}"}) + "\n\n"
                    return
                
                # Yield stream tokens as they arrive
                async for chunk in resp.content.iter_any():
                    if chunk:
                        chunk_str = chunk.decode("utf-8")
                        yield chunk_str
        except Exception as e:
            print(f"[SLMMultiplexer] vLLM Streaming Error: {e}")
            yield "data: " + json.dumps({"status": "error", "message": str(e)}) + "\n\n"

    async def run_audit_inference_stream(self, domain: str, policy_text: str, is_enterprise: bool = False) -> AsyncGenerator[str, None]:
        """
        Execute high-speed privacy policy audit using O(1) Redis dict lookup (Zero-RAG).
        Yields Server-Sent Events (SSE).
        """
        from security import sanitize_input_prompt
        clean_text = sanitize_input_prompt(policy_text, is_audit_policy=True)
        
        # O(1) Deterministic Redis Statutory Fetch (<1ms latency!)
        retrieved_context = await redis_queue.get_audit_statute(clean_text)

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
        
        # Register in Redis Queue for deduplication and prioritization
        priority = 10 if not is_enterprise else 100
        req_id, is_coalesced, position = await redis_queue.register_or_subscribe_request(prompt, self.audit_lora_name, priority)
        
        # Yield initial queue status
        yield f"data: {json.dumps({'status': 'queued', 'position': position, 'requestId': req_id, 'is_coalesced': is_coalesced})}\n\n"
        
        payload = {
            "model": self.audit_lora_name,
            "prompt": prompt,
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": True
        }
        
        start_time = time.time()
        
        async for token_chunk in self._stream_vllm_generator(payload):
            yield token_chunk
            
        latency = (time.time() - start_time) * 1000
        self.total_inferences += 1
        self.total_latency_ms += latency
        
        await redis_queue.complete_request_cleanup(req_id, redis_queue.compute_coalesce_hash(prompt, self.audit_lora_name))

    async def run_chat_inference_stream(self, domain: str, user_prompt: str, is_enterprise: bool = False) -> AsyncGenerator[str, None]:
        """
        Execute conversational Co-Pilot chat using CPU-offloaded ONNX Qdrant RAG.
        Yields Server-Sent Events (SSE).
        """
        from security import sanitize_input_prompt
        clean_prompt = sanitize_input_prompt(user_prompt, is_audit_policy=False)
        
        # Hybrid Search via EdgeRAGEngine (ONNX micro-batcher on CPU)
        retrieved_context = await edge_rag_engine.get_hybrid_chat_context(clean_prompt, top_k=2)

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
        
        priority = 10 if not is_enterprise else 100
        req_id, is_coalesced, position = await redis_queue.register_or_subscribe_request(prompt, self.chat_lora_name, priority)
        
        yield f"data: {json.dumps({'status': 'queued', 'position': position, 'requestId': req_id, 'is_coalesced': is_coalesced})}\n\n"
        
        payload = {
            "model": self.chat_lora_name,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": True
        }
        
        start_time = time.time()
        
        async for token_chunk in self._stream_vllm_generator(payload):
            yield token_chunk
            
        latency = (time.time() - start_time) * 1000
        self.total_inferences += 1
        self.total_latency_ms += latency
        
        await redis_queue.complete_request_cleanup(req_id, redis_queue.compute_coalesce_hash(prompt, self.chat_lora_name))

# Global singleton instance
multiplexer = SLMMultiplexer()
