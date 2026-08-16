#!/usr/bin/env python3
"""
backend_loader.py – Universal Dual-Backend & Multi-LoRA Inference Abstraction for DPDP SLM Evaluation

Supports three universal backends:
1. `unsloth` / `transformers`: Direct PyTorch bfloat16 safetensors / LoRA adapters loaded in DGX VRAM.
2. `vllm`: High-throughput production REST API (`http://localhost:8000/v1/completions`) with Multi-LoRA routing.
3. `llamacpp`: Quantized local GGUF evaluation with optional GBNF grammar enforcement.

Enhancements:
- DGX 128GB Production Ready: Implements polymorphic loading (standalone vs adapter), torchao compat, and explicit garbage collection.
- GGUF auto-discovery.
- ChatML template enforcement via native tokenizers.
- LLM-as-Judge client.
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
import gc
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

DEFAULT_BASE_MODEL = "../models/Qwen2.5-7B-Instruct"

def patch_torchao_dispatch_compat():
    try:
        import torchao.quantization as _tao_q
        if hasattr(_tao_q, "LinearActivationQuantizedTensor"):
            return
        try:
            from torchao.quantization.linear_activation_quantized_tensor import (
                LinearActivationQuantizedTensor as _RealLAQT,
            )
            _tao_q.LinearActivationQuantizedTensor = _RealLAQT
        except ImportError:
            class _StubLinearActivationQuantizedTensor:
                pass
            _tao_q.LinearActivationQuantizedTensor = _StubLinearActivationQuantizedTensor
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# CHATML TEMPLATE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════
def format_chatml_prompt(
    system_message: str,
    user_message: str,
    assistant_prefix: str = ""
) -> str:
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n{assistant_prefix}"
    return prompt

def format_chatml_multi_turn(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True
) -> str:
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        prompt += "<|im_start|>assistant\n"
    return prompt

# ═══════════════════════════════════════════════════════════════════════════
# GGUF AUTO-DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_gguf_path(model_path: str) -> str:
    p = Path(model_path)
    if p.is_file() and p.suffix == ".gguf":
        return str(p)
    if p.is_dir():
        ggufs = sorted(p.glob("*.gguf"))
        if len(ggufs) == 0:
            raise FileNotFoundError(f"No .gguf files found in directory: {p}")
        if len(ggufs) > 1:
            q4_candidates = [g for g in ggufs if "Q4_K_M" in g.name]
            if len(q4_candidates) == 1:
                return str(q4_candidates[0])
            raise ValueError(f"Multiple .gguf files found in {p}. Please specify exact file.")
        return str(ggufs[0])
    return str(p)


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
        self.model_path = model_path
        self.adapter_path = adapter_path if adapter_path else None
        self.vllm_url = vllm_url
        self.lora_name = lora_name
        self.max_seq_length = max_seq_length
        self.llm = None
        self.tokenizer = None
        self.model = None

        if self.backend_type == "unsloth":
            self._init_unsloth()
        elif self.backend_type == "llamacpp":
            if self.model_path:
                self.model_path = _resolve_gguf_path(self.model_path)
            self._init_llamacpp()
        elif self.backend_type == "vllm":
            self._init_vllm()
        elif self.backend_type == "mock":
            self._init_mock()
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}.")

    def unload(self):
        """Deterministically clear GPU memory to prevent VRAM fragmentation."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.llm is not None:
            del self.llm
            self.llm = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _init_mock(self):
        print(f"[BackendEngine] Initializing MOCK Backend (Local testing only)...")
        self.model = "mock_model"
        self.tokenizer = "mock_tokenizer"

    def _init_unsloth(self):
        print(f"[BackendEngine] Initializing Unsloth Backend with model: {self.model_path} | adapter: {self.adapter_path}...")
        import torch
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        
        patch_torchao_dispatch_compat()

        load_path = self.adapter_path if (self.adapter_path and os.path.exists(self.adapter_path)) else self.model_path
        if not load_path or not os.path.exists(load_path):
            raise FileNotFoundError(f"Cannot locate model/adapter path for unsloth backend: {load_path}")

        is_local_dir = os.path.isdir(load_path)
        is_adapter = is_local_dir and os.path.exists(os.path.join(load_path, "adapter_config.json"))
        is_merged = is_local_dir and os.path.exists(os.path.join(load_path, "config.json")) and not is_adapter

        # If it's not a local directory, assume it's a HuggingFace Hub repo ID
        if not is_local_dir:
            print(f"[BackendEngine] Assuming {load_path} is a HuggingFace Hub repo ID.")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=load_path,
                max_seq_length=self.max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=False
            )
        elif is_merged:
            print(f"[BackendEngine] Polymorphic detection: Loading MERGED model from {load_path}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=load_path,
                max_seq_length=self.max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=False
            )
        elif is_adapter:
            print(f"[BackendEngine] Polymorphic detection: Loading ADAPTER from {load_path} onto base {DEFAULT_BASE_MODEL}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=DEFAULT_BASE_MODEL,
                max_seq_length=self.max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=False
            )
            self.model.load_adapter(load_path)
        else:
            raise ValueError(f"Path {load_path} does not appear to be a valid merged model or adapter.")

        # Enforce native ChatML template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="chatml",
            mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
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
        print(f"[BackendEngine] Initializing vLLM Backend natively for: {self.model_path}...")
        try:
            import vllm
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is not installed in this environment. Use a dedicated vLLM env for serving.")
        
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=1,
            trust_remote_code=True
        )
        print("[BackendEngine] vLLM Engine successfully initialized natively.")

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

        if self.backend_type == "mock":
            import time
            time.sleep(0.01)
            mock_resp = '{"violations": [{"type": "Excessive Collection", "severity": "HIGH"}], "dpdp_trust_score": 30, "subtlety_score": 5}'
            return {"raw_output": mock_resp, "latency_ms": 10.0, "ttft_ms": 5.0, "tokens_generated": 10, "tokens_per_sec": 100.0}

        start_time = time.time()
        first_token_time = None
        output_text = ""
        token_count = 0

        if self.backend_type == "unsloth":
            import torch
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
            first_token_time = time.time()
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
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=temperature if temperature > 0 else 0.0,
                max_tokens=max_tokens,
                stop=stop
            )
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
            output_text = outputs[0].outputs[0].text
            token_count = len(outputs[0].outputs[0].token_ids)
            first_token_time = start_time
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


# ═══════════════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE CLIENT
# ═══════════════════════════════════════════════════════════════════════════
class JudgeClient:
    def __init__(self, api_url: str, model_name: str = "teacher"):
        self.api_url = api_url
        self.model_name = model_name

    @classmethod
    def try_connect(
        cls,
        api_url: str = "http://localhost:8001/v1/completions",
        model_name: str = "teacher",
        timeout: int = 5
    ) -> Optional["JudgeClient"]:
        health_url = api_url.replace("/v1/completions", "/health")
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    print(f"✅ [JudgeClient] Connected to LLM-as-Judge at: {api_url}")
                    return cls(api_url, model_name)
                else:
                    print(f"⚠️ [JudgeClient] LOUD SKIP: Judge health check returned status {resp.status}.")
                    return None
        except Exception as e:
            print(f"⚠️ [JudgeClient] LOUD SKIP: Cannot reach LLM-as-Judge at {api_url}: {e}")
            return None

    def generate(self, prompt: str, max_tokens: int = 10, temperature: float = 0.0) -> Dict[str, Any]:
        if self.backend_type == "mock":
            import time
            time.sleep(0.01)
            mock_resp = '{"violations": [{"type": "Excessive Collection", "severity": "HIGH"}], "dpdp_trust_score": 30, "subtlety_score": 5}'
            return {"raw_output": mock_resp, "latency_ms": 10.0, "tokens_per_second": 100.0, "token_count": 10}
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["<|im_end|>"],
            "stream": False
        }
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            text = data.get("choices", [{}])[0].get("text", "")
            return {"raw_output": text.strip()}
