#!/usr/bin/env python3
"""
backend_loader.py – Universal Dual-Backend & Multi-LoRA Inference Abstraction

Supports three universal backends:
1. `unsloth` / `transformers`: Direct PyTorch BF16 safetensors / LoRA adapters loaded in DGX VRAM.
2. `vllm`: High-throughput production engine with Multi-LoRA routing, PagedAttention, and Guided Decoding.
3. `llamacpp`: Quantized local GGUF evaluation with native FlashAttention support.

SOTA Enhancements:
- vLLM Guided Decoding: Natively injects `guided_json` schemas into the C++ CUDA kernel for strict API contracts.
- KV-Cache Scaling: Optimized `gpu_memory_utilization=0.95` to support full 32k token windows in production.
- Strict VRAM Airlock: Deep distributed state destruction for vLLM to prevent memory leaks.
- Dynamic LoRA Routing: vLLM natively mounts `adapter_path` via `LoRARequest` on top of base model.
- Indestructible Paths: Perfectly synchronized with `path_resolver.py`.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import gc
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Ensure terminal stdout/stderr uses UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Dynamic path resolution to hit path_resolver.py
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

try:
    from path_resolver import Paths
except ImportError:
    print("❌ Core module import failed: path_resolver")
    sys.exit(1)


def patch_torchao_dispatch_compat():
    """Patches TorchAO dispatch logic to prevent Unsloth PEFT instantiation crashes."""
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
    """Strictly formats prompts using the native Qwen ChatML delimiters."""
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n{assistant_prefix}"
    return prompt

def format_chatml_multi_turn(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True
) -> str:
    """Formats an arbitrary conversation history into a unified ChatML string."""
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        prompt += "<|im_start|>assistant\n"
    return prompt


# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSAL BACKEND ENGINE
# ═══════════════════════════════════════════════════════════════════════════
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
        
        # SOTA FIX: Dynamically resolve paths against absolute repository anchors
        self.model_path = str(Paths.resolve_model_path(model_path, "Qwen2.5-7B-Instruct")) if model_path else None
        self.adapter_path = str(Paths.resolve_model_path(adapter_path, "")) if adapter_path else None
        self.vllm_url = vllm_url
        self.lora_name = lora_name or "default_lora"
        self.max_seq_length = max_seq_length
        
        self.llm = None
        self.tokenizer = None
        self.model = None
        self.lora_request = None

        if self.backend_type == "unsloth":
            self._init_unsloth()
        elif self.backend_type == "llamacpp":
            if self.model_path:
                self.model_path = str(Paths.resolve_gguf_path(self.model_path))
            self._init_llamacpp()
        elif self.backend_type == "vllm":
            self._init_vllm()
        elif self.backend_type == "mock":
            self._init_mock()
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}. Supported: unsloth, vllm, llamacpp.")

    def unload(self):
        """
        Deterministically clears GPU memory to prevent VRAM fragmentation.
        Critically implements deep vLLM distributed state destruction.
        """
        if self.backend_type == "vllm" and self.llm is not None:
            print("🧹 [VRAM Airlock] Destroying vLLM distributed engine state...")
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel, destroy_env
                destroy_model_parallel()
                destroy_env()
            except ImportError:
                pass
            except Exception as e:
                print(f"⚠️ [VRAM Airlock] Non-fatal vLLM cleanup issue: {e}")

        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.llm is not None:
            del self.llm
            self.llm = None
            
        self.lora_request = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
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
        
        # Maximize Ampere/Hopper GPU efficiency
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        patch_torchao_dispatch_compat()

        load_path = self.adapter_path if (self.adapter_path and os.path.exists(self.adapter_path)) else self.model_path
        if not load_path or not os.path.exists(load_path):
            raise FileNotFoundError(f"Cannot locate model/adapter path for unsloth backend: {load_path}")

        is_local_dir = os.path.isdir(load_path)
        is_adapter = is_local_dir and os.path.exists(os.path.join(load_path, "adapter_config.json"))
        is_merged = is_local_dir and os.path.exists(os.path.join(load_path, "config.json")) and not is_adapter

        if not is_local_dir:
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
            base_model_abs = str(Paths.resolve_model_path(None, "Qwen2.5-7B-Instruct"))
            print(f"[BackendEngine] Polymorphic detection: Loading ADAPTER from {load_path} onto base {base_model_abs}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_abs,
                max_seq_length=self.max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=False
            )
            self.model.load_adapter(load_path)

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
            n_gpu_layers=-1, # Offload entirely to GPU
            flash_attn=True, # Dramatically reduces KV cache memory footprint
            verbose=False
        )
        print("[BackendEngine] llama_cpp GGUF model successfully loaded.")

    def _init_vllm(self):
        print(f"[BackendEngine] Initializing vLLM Production Backend natively for: {self.model_path}...")
        try:
            import vllm
            from vllm import LLM
            from vllm.lora.request import LoRARequest
        except ImportError:
            raise ImportError("vLLM is not installed in this environment. Use a dedicated vLLM env for serving.")
        
        # Base Model Initialization with dynamic LoRA allocation support
        enable_lora_flag = bool(self.adapter_path and os.path.exists(self.adapter_path))
        
        # SOTA FIX: Pushed GPU utilization to 0.95 to safely fit full 32k KV Cache payloads
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enable_lora=enable_lora_flag,
            max_lora_rank=128 if enable_lora_flag else 16,
            max_model_len=self.max_seq_length,
            gpu_memory_utilization=0.95
        )
        
        if enable_lora_flag:
            print(f"[BackendEngine] Multi-LoRA vLLM Routing Enabled. Mounting {self.lora_name} -> {self.adapter_path}")
            self.lora_request = LoRARequest(self.lora_name, 1, self.adapter_path)
            
        print("[BackendEngine] vLLM Engine successfully initialized.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        grammar: Optional[Any] = None
    ) -> Dict[str, Any]:
        
        if stop is None:
            stop = ["<|im_end|>", "<|endoftext|>"]

        if self.backend_type == "mock":
            time.sleep(0.01)
            mock_resp = '{"violations": [{"type": "Excessive Collection", "severity": "HIGH"}], "dpdp_trust_score": 30, "subtlety_score": 5}'
            return {"raw_output": mock_resp, "latency_ms": 10.0, "ttft_ms": 5.0, "tokens_generated": 10, "tokens_per_sec": 1000.0}

        start_time = time.perf_counter()
        first_token_time = None
        output_text = ""
        token_count = 0

        # ---------------------------------------------------------------------
        # UNSLOTH INFERENCE (No Native Guided Decoding Support)
        # ---------------------------------------------------------------------
        if self.backend_type == "unsloth":
            import torch
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
            first_token_time = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    max_length=None, # Prevents CUDA illegal memory access
                    temperature=temperature if temperature > 0 else 0.01,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            end_time = time.perf_counter()
            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            
            # Explicitly decode WITHOUT dropping special tokens to reliably split on stop words
            output_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)
            for s in stop:
                if s in output_text:
                    output_text = output_text.split(s)[0]
                    
            output_text = output_text.replace("<|im_end|>", "").strip()
            token_count = len(gen_tokens)

        # ---------------------------------------------------------------------
        # LLAMACPP INFERENCE (Ignores JSON-schema grammar, uses GBNF internally)
        # ---------------------------------------------------------------------
        elif self.backend_type == "llamacpp":
            kwargs = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "stream": True
            }
            # Note: llama_cpp uses GBNF grammar, not JSON. We ignore the passed JSON schema here
            # to prevent crashes, as Edge GGUF inference relies on fine-tuned alignment.
                
            stream = self.llm(prompt, **kwargs)
            for chunk in stream:
                if first_token_time is None and chunk['choices'][0].get('text', ''):
                    first_token_time = time.perf_counter()
                text = chunk['choices'][0].get('text', '')
                output_text += text
                token_count += 1
            end_time = time.perf_counter()

        # ---------------------------------------------------------------------
        # vLLM INFERENCE (Native Guided Decoding / Structured Outputs)
        # ---------------------------------------------------------------------
        elif self.backend_type == "vllm":
            from vllm import SamplingParams
            
            # SOTA FIX: Parse the stringified schema back into dict to inject into outlines guided_json
            parsed_guided_json = None
            if grammar is not None:
                try:
                    if isinstance(grammar, str):
                        parsed_guided_json = json.loads(grammar)
                    elif isinstance(grammar, dict):
                        parsed_guided_json = grammar
                except Exception as e:
                    print(f"⚠️ [vLLM] Failed to parse grammar payload for guided decoding: {e}")

            sampling_params = SamplingParams(
                temperature=temperature if temperature > 0 else 0.0,
                max_tokens=max_tokens,
                stop=stop,
                guided_json=parsed_guided_json
            )
            
            kwargs = {"use_tqdm": False}
            if self.lora_request is not None:
                kwargs["lora_request"] = self.lora_request
                
            outputs = self.llm.generate([prompt], sampling_params, **kwargs)
            
            output_text = outputs[0].outputs[0].text
            token_count = len(outputs[0].outputs[0].token_ids)
            first_token_time = start_time # vLLM batch abstraction
            end_time = time.perf_counter()

        total_latency_ms = (end_time - start_time) * 1000.0
        ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else total_latency_ms
        generation_time_ms = total_latency_ms - ttft_ms
        
        if generation_time_ms <= 0.001:
            tokens_per_sec = token_count / (total_latency_ms / 1000.0) if total_latency_ms > 0 else 0
        else:
            tokens_per_sec = token_count / (generation_time_ms / 1000.0)

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
        """Verifies connection to the LLM-as-a-Judge teacher model API."""
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

    def generate(self, prompt: str, max_tokens: int = 10, temperature: float = 0.0, retries: int = 3) -> Dict[str, Any]:
        """Dispatches prompts to the LLM judge API with exponential backoff."""
        if self.api_url == "mock":
            time.sleep(0.01)
            return {"raw_output": "5", "latency_ms": 10.0, "tokens_per_second": 100.0, "token_count": 1}
            
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["<|im_end|>"],
            "stream": False
        }
        
        for attempt in range(retries):
            try:
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
            except Exception as e:
                if attempt == retries - 1:
                    raise Exception(f"Judge API failed after {retries} attempts: {e}")
                time.sleep(2 ** attempt)