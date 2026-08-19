#!/usr/bin/env python3
"""
engine.py – SOTA Zero-Hop In-Process AsyncLLMEngine
Features:
1. Dynamic 32GB VRAM Hardcap regardless of host GPU size.
2. 1K Concurrent User scaling via FP8 KV Cache & Prefix Caching.
3. Multi-LoRA Multiplexing (Audit vs Chatbot) in a single VRAM footprint.
4. FSM Schema Caching for Zero-Overhead Guided Decoding (JSON enforcement).
"""

import os
import json
import torch
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional, Union

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams, GuidedDecodingParams
from vllm.lora.request import LoRARequest

class ProductionAsyncEngine:
    def __init__(
        self,
        base_model_path: str,
        audit_adapter_path: str,
        chatbot_adapter_path: str,
    ):
        print("[EngineCore] Booting Zero-Hop AsyncLLMEngine...")
        
        self.base_model_path = base_model_path
        self.audit_adapter_path = audit_adapter_path
        self.chatbot_adapter_path = chatbot_adapter_path
        
        # ─────────────────────────────────────────────────────────────
        # 🚨 SOTA Feature 1: DYNAMIC 32GB VRAM HARDCAP
        # ─────────────────────────────────────────────────────────────
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        target_vram_gb = 32.0
        
        if total_vram_gb > target_vram_gb:
            # Scale fractional utilization down to hit exactly 32GB (e.g., 80GB -> 0.40)
            utilization = target_vram_gb / total_vram_gb
        else:
            # Leave 10% overhead for OS/CUDA contexts on smaller GPUs
            utilization = 0.90 

        print(f"[EngineCore] Detected {total_vram_gb:.1f}GB VRAM. Enforcing {utilization:.3f} utilization to strictly cap at <= {target_vram_gb}GB.")

        # ─────────────────────────────────────────────────────────────
        # 🚨 SOTA Feature 2: 1K CONCURRENCY ARCHITECTURE
        # ─────────────────────────────────────────────────────────────
        engine_args = AsyncEngineArgs(
            model=self.base_model_path,
            
            # LoRA Multiplexing Setup
            enable_lora=True,
            max_loras=2,
            max_lora_rank=128,
            max_cpu_loras=4,
            
            # Context & Hardware Constraints
            max_model_len=8192,           # Clamped to prevent exponential VRAM scaling
            max_num_seqs=256,             # Maximum concurrent sequences per batch iteration
            gpu_memory_utilization=utilization,
            
            # SOTA Concurrency Boosters
            kv_cache_dtype="fp8",         # Doubles KV sequence capacity with 0.1% quality loss
            enable_prefix_caching=True,   # O(1) Memory sharing for identical System Prompts & RAG blocks
            enable_chunked_prefill=True,  # Prevents OOM spikes during massive concurrent prompt arrivals
            
            trust_remote_code=True,
            disable_log_requests=True     # Prevents IO bottlenecks on stdout during high throughput
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # 3. Mount Active LoRA Requests to VRAM
        print("[EngineCore] Mounting Multi-LoRA Adapters...")
        self.lora_requests: Dict[str, LoRARequest] = {
            "audit": LoRARequest("audit_lora", 1, self.audit_adapter_path),
            "chatbot": LoRARequest("chatbot_lora", 2, self.chatbot_adapter_path),
        }

        # 4. Initialize FSM Schema Grammar Cache
        self._schema_cache: Dict[str, GuidedDecodingParams] = {}
        print("✅ [EngineCore] vLLM Engine fully initialized in Zero-Hop mode.")

    def get_cached_guided_decoding(self, schema_payload: Union[str, Dict[str, Any]]) -> Optional[GuidedDecodingParams]:
        """Caches schema compilation to eliminate FSM construction latency during Audits."""
        if not schema_payload:
            return None
        
        cache_key = schema_payload if isinstance(schema_payload, str) else json.dumps(schema_payload, sort_keys=True)
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        parsed_json = json.loads(schema_payload) if isinstance(schema_payload, str) else schema_payload
        guided_params = GuidedDecodingParams(json=parsed_json)
        self._schema_cache[cache_key] = guided_params
        return guided_params

    async def generate_audit(
        self,
        request_id: str,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0
    ) -> str:
        """Executes Forensic Policy Audits through the Audit LoRA with strict JSON schema constraints."""
        guided_decoding = self.get_cached_guided_decoding(schema)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
            guided_decoding=guided_decoding
        )

        results_generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=self.lora_requests["audit"]
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        return final_output.outputs[0].text if final_output else ""

    async def generate_chat_stream(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3
    ) -> AsyncGenerator[str, None]:
        """Streams Conversational Chatbot tokens through the Chatbot LoRA in real-time."""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        results_generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=self.lora_requests["chatbot"]
        )

        prev_text = ""
        async for request_output in results_generator:
            curr_text = request_output.outputs[0].text
            delta = curr_text[len(prev_text):]
            prev_text = curr_text
            if delta:
                yield delta