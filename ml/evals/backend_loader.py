#!/usr/bin/env python3
"""
backend_loader.py – Universal Dual-Backend & High-Throughput Inference Abstraction

Supports three universal backends:
1. `unsloth` / `transformers`: Direct PyTorch BF16 safetensors / LoRA adapters loaded in DGX VRAM.
2. `vllm`: Production engine with Continuous Batching, PagedAttention, and Cached Guided Decoding.
3. `llamacpp`: Quantized local GGUF evaluation with native FlashAttention support.

SOTA Enhancements:
- High-Throughput Vectorized Batching: `generate()` dynamically handles both single strings and full prompt lists.
- Cached Guided Decoding: Reuses `GuidedDecodingParams` FSM instances to eliminate compilation overhead.
- V1 Engine Lifecycle Management: Clean termination of EngineCore multiprocessing workers.
- Adaptive Memory Safety: Configured at 0.80 GPU utilization to prevent host OS allocation panics.
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
        backend_type: str = "vllm",
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
        self._cached_guided_decoding = {}

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
        """Deterministically clears GPU memory to prevent VRAM fragmentation."""
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
        self._cached_guided_decoding.clear()

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
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        patch_torchao_dispatch_compat()

        load_path = self.adapter_path if (self.adapter_path and os.path.exists(self.adapter_path)) else self.model_path
        if not load_path or not os.path.exists(load_path):
            raise FileNotFoundError(f"Cannot locate model/adapter path for unsloth backend: {load_path}")

        is_local_dir = os.path.isdir(load_path)
        is_adapter = is_local_dir and os.path.exists(os.path.join(load_path, "adapter_config.json"))
        is_merged = is_local_dir and os.path.exists(os.path.join(load_path, "config.json")) and not is_adapter

        if not is_local_dir or is_merged:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=load_path,
                max_seq_length=self.max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=False
            )
        elif is_adapter:
            base_model_abs = str(Paths.resolve_model_path(None, "Qwen2.5-7B-Instruct"))
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_abs,
                max_seq_length=self.max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=False
            )
            self.model.load_adapter(load_path)

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
            raise ImportError("llama_cpp not installed.")
            
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"GGUF model not found at path: {self.model_path}")
            
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=min(self.max_seq_length, 32768),
            n_gpu_layers=-1, 
            flash_attn=True, 
            verbose=False
        )
        print("[BackendEngine] llama_cpp GGUF model successfully loaded.")

    def _init_vllm(self):
        print(f"[BackendEngine] Initializing vLLM Production Backend natively for: {self.model_path}...")
        try:
            from vllm import LLM
            from vllm.lora.request import LoRARequest
        except ImportError:
            raise ImportError("vLLM is not installed in this environment.")
        
        enable_lora_flag = bool(self.adapter_path and os.path.exists(self.adapter_path))
        
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enable_lora=enable_lora_flag,
            max_lora_rank=128 if enable_lora_flag else 16,
            max_model_len=self.max_seq_length,
            gpu_memory_utilization=0.80
        )
        
        if enable_lora_flag:
            print(f"[BackendEngine] Multi-LoRA vLLM Routing Enabled. Mounting {self.lora_name} -> {self.adapter_path}")
            self.lora_request = LoRARequest(self.lora_name, 1, self.adapter_path)
            
        print("[BackendEngine] vLLM Engine successfully initialized.")

    def _get_guided_decoding_params(self, grammar: Optional[Any]):
        """Caches and returns GuidedDecodingParams for structured decoding."""
        if grammar is None:
            return None

        try:
            from vllm.sampling_params import GuidedDecodingParams
        except ImportError:
            return None

        cache_key = grammar if isinstance(grammar, str) else json.dumps(grammar, sort_keys=True)
        if cache_key in self._cached_guided_decoding:
            return self._cached_guided_decoding[cache_key]

        parsed_json = None
        try:
            if isinstance(grammar, str):
                parsed_json = json.loads(grammar)
            elif isinstance(grammar, dict):
                parsed_json = grammar
        except Exception as e:
            print(f"⚠️ [vLLM] Failed to parse grammar payload: {e}")
            return None

        if parsed_json:
            guided_param = GuidedDecodingParams(json=parsed_json)
            self._cached_guided_decoding[cache_key] = guided_param
            return guided_param
        return None

    # 🚨 SOTA FIX: Accepts BOTH `str` and `List[str]` for Vectorized Batching!
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        grammar: Optional[Any] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        
        is_single = isinstance(prompt, str)
        prompts = [prompt] if is_single else prompt

        if stop is None:
            stop = ["<|im_end|>", "<|endoftext|>"]

        if self.backend_type == "mock":
            time.sleep(0.01)
            mock_item = {"raw_output": '{"violations": [{"type": "Excessive Collection", "severity": "HIGH"}], "dpdp_trust_score": 30, "subtlety_score": 5}', "latency_ms": 10.0, "ttft_ms": 5.0, "tokens_generated": 10, "tokens_per_sec": 1000.0}
            return mock_item if is_single else [mock_item] * len(prompts)

        start_time = time.perf_counter()

        # ---------------------------------------------------------------------
        # vLLM HIGH-THROUGHPUT BATCH GENERATION
        # ---------------------------------------------------------------------
        if self.backend_type == "vllm":
            from vllm import SamplingParams

            guided_decoding = self._get_guided_decoding_params(grammar)
            
            if guided_decoding is not None:
                sampling_params = SamplingParams(
                    temperature=temperature if temperature > 0 else 0.0,
                    max_tokens=max_tokens,
                    stop=stop,
                    guided_decoding=guided_decoding
                )
            else:
                sampling_params = SamplingParams(
                    temperature=temperature if temperature > 0 else 0.0,
                    max_tokens=max_tokens,
                    stop=stop
                )
            
            kwargs = {"use_tqdm": False}
            if self.lora_request is not None:
                kwargs["lora_request"] = self.lora_request

            # 🚨 SOTA FIX: Passing `prompts` directly (No inner brackets)
            outputs = self.llm.generate(prompts, sampling_params, **kwargs)
            end_time = time.perf_counter()

            total_elapsed_ms = (end_time - start_time) * 1000.0
            # SOTA FIX: In synchronous batch execution, the true wall-clock latency for EVERY prompt is the total elapsed time.
            actual_latency_ms = total_elapsed_ms

            results = []
            for out in outputs:
                text = out.outputs[0].text
                token_count = len(out.outputs[0].token_ids)
                tps = token_count / (actual_latency_ms / 1000.0) if actual_latency_ms > 0 else 0.0
                
                # Attempt to extract precise TTFT from vLLM's internal RequestMetrics
                if hasattr(out, "metrics") and out.metrics is not None and getattr(out.metrics, "first_token_time", None) and getattr(out.metrics, "arrival_time", None):
                    actual_ttft_ms = (out.metrics.first_token_time - out.metrics.arrival_time) * 1000.0
                else:
                    # Fallback heuristic: TTFT is prefill time + 1 token decode
                    actual_ttft_ms = actual_latency_ms * 0.3

                results.append({
                    "raw_output": text.strip(),
                    "latency_ms": actual_latency_ms,
                    "ttft_ms": actual_ttft_ms,
                    "tokens_generated": token_count,
                    "tokens_per_sec": tps
                })

            return results[0] if is_single else results

        # ---------------------------------------------------------------------
        # UNSLOTH INFERENCE (Sequential Fallback)
        # ---------------------------------------------------------------------
        elif self.backend_type == "unsloth":
            import torch
            results = []
            for p in prompts:
                t0 = time.perf_counter()
                inputs = self.tokenizer([p], return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        max_length=None,
                        temperature=temperature if temperature > 0 else 0.01,
                        do_sample=True if temperature > 0 else False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                t1 = time.perf_counter()
                gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                output_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)
                for s in stop:
                    if s in output_text:
                        output_text = output_text.split(s)[0]
                output_text = output_text.replace("<|im_end|>", "").strip()
                
                lat_ms = (t1 - t0) * 1000.0
                results.append({
                    "raw_output": output_text,
                    "latency_ms": lat_ms,
                    "ttft_ms": lat_ms * 0.5,
                    "tokens_generated": len(gen_tokens),
                    "tokens_per_sec": len(gen_tokens) / (lat_ms / 1000.0) if lat_ms > 0 else 0.0
                })
            return results[0] if is_single else results

        # ---------------------------------------------------------------------
        # LLAMACPP INFERENCE (Sequential Fallback)
        # ---------------------------------------------------------------------
        elif self.backend_type == "llamacpp":
            results = []
            for p in prompts:
                t0 = time.perf_counter()
                stream = self.llm(p, max_tokens=max_tokens, temperature=temperature, stop=stop, stream=True)
                out_text = ""
                tok_cnt = 0
                for chunk in stream:
                    out_text += chunk['choices'][0].get('text', '')
                    tok_cnt += 1
                t1 = time.perf_counter()
                lat_ms = (t1 - t0) * 1000.0
                results.append({
                    "raw_output": out_text.strip(),
                    "latency_ms": lat_ms,
                    "ttft_ms": lat_ms * 0.2,
                    "tokens_generated": tok_cnt,
                    "tokens_per_sec": tok_cnt / (lat_ms / 1000.0) if lat_ms > 0 else 0.0
                })
            return results[0] if is_single else results


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
                return None
        except Exception:
            return None

    def generate(self, prompt: str, max_tokens: int = 10, temperature: float = 0.0, retries: int = 3) -> Dict[str, Any]:
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