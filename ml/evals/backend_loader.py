#!/usr/bin/env python3
"""
backend_loader.py – Universal Dual-Backend & Multi-LoRA Inference Abstraction for DPDP SLM Evaluation

Supports three universal backends:
1. `unsloth` / `transformers`: Direct PyTorch bfloat16 safetensors / LoRA adapters loaded in DGX VRAM.
2. `vllm`: High-throughput production REST API (`http://localhost:8000/v1/completions`) with Multi-LoRA routing.
3. `llamacpp`: Quantized local GGUF evaluation with optional GBNF grammar enforcement.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import time
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

class BackendEngine:
    def __init__(
        self,
        backend_type: str = "unsloth",
        model_path: Optional[Union[str, Path]] = None,
        adapter_path: Optional[Union[str, Path]] = None,
        vllm_url: str = "http://localhost:8000/v1/completions",
        lora_name: Optional[str] = None,
        max_seq_length: int = 24576
    ):
        self.backend_type = backend_type.lower().strip()
        self.model_path = str(model_path) if model_path else None
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.vllm_url = vllm_url
        self.lora_name = lora_name
        self.max_seq_length = max_seq_length
        self.llm = None
        self.tokenizer = None
        self.model = None

        if self.backend_type == "unsloth":
            self._init_unsloth()
        elif self.backend_type == "llamacpp":
            self._init_llamacpp()
        elif self.backend_type == "vllm":
            self._init_vllm()
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}. Must be 'unsloth', 'vllm', or 'llamacpp'.")

    def _init_unsloth(self):
        print(f"[BackendEngine] Initializing Unsloth/Transformers Backend with model: {self.model_path} | adapter: {self.adapter_path}...")
        import torch
        from unsloth import FastLanguageModel
        
        load_path = self.adapter_path if (self.adapter_path and os.path.exists(self.adapter_path)) else self.model_path
        if not load_path or not os.path.exists(load_path):
            raise FileNotFoundError(f"Cannot locate model/adapter path for unsloth backend: {load_path}")
            
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=load_path,
            max_seq_length=self.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            use_flash_attention_2=True
        )
        FastLanguageModel.for_inference(self.model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("[BackendEngine] Unsloth model successfully loaded into DGX VRAM.")

    def _init_llamacpp(self):
        print(f"[BackendEngine] Initializing llama_cpp GGUF Backend with model: {self.model_path}...")
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama_cpp not installed. Please run `pip install llama-cpp-python`.")
            
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"GGUF model not found at path: {self.model_path}")
            
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=min(self.max_seq_length, 32768),
            n_gpu_layers=-1,
            verbose=False
        )
        print("[BackendEngine] llama_cpp GGUF model successfully loaded.")

    def _init_vllm(self):
        print(f"[BackendEngine] Initializing vLLM REST Client connected to: {self.vllm_url} | LoRA: {self.lora_name}...")
        # Verify connectivity
        health_url = self.vllm_url.replace("/v1/completions", "/health")
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    print(f"⚠️ Warning: vLLM health check returned status {resp.status}")
        except Exception as e:
            print(f"⚠️ Warning: Could not reach vLLM health endpoint ({health_url}): {e}. Ensure vLLM server is running.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        grammar: Optional[Any] = None
    ) -> Dict[str, Any]:
        if stop is None:
            stop = ["<|im_end|>"]

        start_time = time.time()
        first_token_time = None
        output_text = ""
        token_count = 0

        if self.backend_type == "unsloth":
            import torch
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
            first_token_time = time.time() # Proxy for local inference
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 0.01,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            end_time = time.time()
            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            output_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            for s in stop:
                if s in output_text:
                    output_text = output_text.split(s)[0]
            token_count = len(gen_tokens)

        elif self.backend_type == "llamacpp":
            kwargs = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "stream": True
            }
            if grammar is not None:
                kwargs["grammar"] = grammar
                
            stream = self.llm(prompt, **kwargs)
            for chunk in stream:
                if first_token_time is None and chunk['choices'][0].get('text', ''):
                    first_token_time = time.time()
                text = chunk['choices'][0].get('text', '')
                output_text += text
                token_count += 1
            end_time = time.time()

        elif self.backend_type == "vllm":
            payload = {
                "model": self.lora_name if self.lora_name else "default",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "stream": True
            }
            req = urllib.request.Request(
                self.vllm_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for line in resp:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                text = chunk.get("choices", [{}])[0].get("text", "")
                                if text:
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    output_text += text
                                    token_count += 1
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                return {
                    "raw_output": f"ERROR_VLLM_CONNECT: {e}",
                    "latency_ms": 0,
                    "ttft_ms": 0,
                    "tokens_generated": 0,
                    "tokens_per_sec": 0
                }
            end_time = time.time()

        total_latency_ms = (end_time - start_time) * 1000
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_latency_ms
        generation_time_ms = total_latency_ms - ttft_ms
        tokens_per_sec = (token_count / (generation_time_ms / 1000)) if generation_time_ms > 0.001 else 0

        return {
            "raw_output": output_text.strip(),
            "latency_ms": total_latency_ms,
            "ttft_ms": ttft_ms,
            "tokens_generated": token_count,
            "tokens_per_sec": tokens_per_sec
        }
