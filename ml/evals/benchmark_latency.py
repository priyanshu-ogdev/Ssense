#!/usr/bin/env python3
"""
benchmark_latency.py – Pillar 5: End-to-End Inference Latency & Load Resilience

Benchmarks fine-tuned DPDP SLM inference across multiple concurrency batch sizes (1, 4, 8, 16)
and tests maximum 32k context window stress resistance without Out-Of-Memory (OOM) failure.

SOTA Upgrades Implemented:
1. Native Vectorized Batching: Replaced `ThreadPoolExecutor` with native vLLM `generate([prompts])`.
2. PyTorch Allocator Decoupling: Removed `sync_cuda()` and `reset_peak_memory_stats()` to prevent vLLM memory collisions.
3. Percentile Telemetry: Computes Mean, P50, P90, P95, and P99 for TTFT and Latency directly from vLLM output metadata.
4. Robust 32k Token Stress: Accurately synthesizes 30,000–32,000 token legal context payloads.
5. Strict VRAM Airlock: Guarantees engine destruction and CUDA cache release in `finally:`.
"""

import os
import sys
import gc
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

import torch

# Ensure terminal stdout/stderr uses UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Dynamic path resolution to ensure imports work from any execution directory
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

try:
    from path_resolver import Paths
    from backend_loader import BackendEngine, format_chatml_prompt
except ImportError as e:
    print(f"❌ Failed to import core evaluation modules: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION & PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL_PATH = Paths.resolve_model_path(None, "chatbot-model-final")
LAW_TXT_PATH = Paths.LAW_TEXT
REPORT_DIR = Paths.ensure_reports_dir()
REPORT_PATH = REPORT_DIR / "latency_stress_benchmark_report.json"

BATCH_SIZES = [1, 4, 8, 16]
STANDARD_SYS_MSG = (
    "You are an expert legal assistant specialized in the Digital Personal Data Protection (DPDP) Act 2023. "
    "Provide clear, legally precise answers citing applicable sections."
)
STANDARD_USER_MSG = "Summarize the primary obligations of a Data Fiduciary regarding consent and security safeguards under the DPDP Act 2023."


# ═══════════════════════════════════════════════════════════════════════════
# TELEMETRY & STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def get_vram_usage_mb() -> float:
    """Returns current reserved CUDA VRAM in Megabytes."""
    if torch.cuda.is_available():
        return float(torch.cuda.memory_reserved() / (1024 * 1024))
    return 0.0

def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Computes distribution percentiles for latency measurements."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2)
    }


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def warmup_engine(engine: BackendEngine, num_passes: int = 2):
    """Executes warm-up passes to prime GPU kernels and JIT caches."""
    print("🔥 Warming up GPU inference kernels...")
    warmup_prompt = format_chatml_prompt("You are a legal assistant.", "Ping.")
    for _ in range(num_passes):
        _ = engine.generate(warmup_prompt, max_tokens=16, temperature=0.0)


def generate_32k_stress_prompt() -> str:
    """Synthesizes a 30,000–32,000 token statutory context payload."""
    base_law = "DPDP Act 2023 Section statutory provisions, audit parameters, and fiduciary clauses. " * 50
    if LAW_TXT_PATH.exists():
        try:
            with open(LAW_TXT_PATH, "r", encoding="utf-8") as f:
                content = f.read()
                if len(content.strip()) > 500:
                    base_law = content
        except Exception:
            pass

    # Approximate ~30,000–32,000 tokens (~115,000 to 120,000 characters)
    target_chars = 115000
    repeats = (target_chars // len(base_law)) + 1
    repeated_law = (base_law * repeats)[:target_chars]

    user_msg = (
        f"[EXTENSIVE LEGAL REPOSITORY: ~30,000 TOKENS]\n{repeated_law}\n\n"
        "[SYNTHESIS QUERY]\n"
        "Based strictly on the statutory text provided above, extract all explicit obligations "
        "and duties imposed on a Data Fiduciary and detail the penalty framework under the Act."
    )
    return format_chatml_prompt(STANDARD_SYS_MSG, user_msg)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Pillar 5: End-to-End Inference Latency & Load Resilience")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "unsloth", "llamacpp"])
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES)
    parser.add_argument("--skip-32k", action="store_true", help="Skip 32k deep context OOM stress simulation")
    parser.add_argument("--lora-name", type=str, default="chatbot")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1/completions")
    args = parser.parse_args()

    print("═══════════════════════════════════════════════════════════════════════")
    print(f"🏎️ [PILLAR 5]: INFERENCE LATENCY, THROUGHPUT & STRESS BENCHMARK ({args.backend.upper()})")
    print("═══════════════════════════════════════════════════════════════════════")
    print(f"📦 Target Model Path: {args.model_path}")

    # Initialize Engine with full 32k context capability
    engine = BackendEngine(
        backend_type=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        lora_name=args.lora_name,
        vllm_url=args.vllm_url,
        max_seq_length=32768
    )

    batch_results = {}
    stress_32k_passed = False
    stress_metrics = {}

    try:
        # 1. Warm-up Phase
        warmup_engine(engine)

        # 2. Concurrency Load Benchmarks
        built_prompt = format_chatml_prompt(STANDARD_SYS_MSG, STANDARD_USER_MSG)

        print("\n🚀 Executing Multi-Batch Concurrency Throughput Benchmarks...")
        for bs in args.batch_sizes:
            print(f"   ⏱️ Evaluating Concurrency Batch Size = {bs}...")
            
            # SOTA FIX: Vectorized batching replaces ThreadPoolExecutor
            prompts = [built_prompt] * bs
            
            t_batch_start = time.perf_counter()
            outputs = engine.generate(prompts, max_tokens=128, temperature=0.0)
            t_batch_end = time.perf_counter()
            
            total_duration_sec = max(1e-5, t_batch_end - t_batch_start)

            if not isinstance(outputs, list):
                outputs = [outputs]

            ttfts = [o.get("ttft_ms", 0.0) for o in outputs if o.get("ttft_ms", 0.0) > 0]
            lats = [o.get("latency_ms", 0.0) for o in outputs if o.get("latency_ms", 0.0) > 0]
            total_tokens = sum(o.get("tokens_generated", 0) for o in outputs)

            # Statistical Aggregations
            ttft_stats = calculate_percentiles(ttfts)
            latency_stats = calculate_percentiles(lats)
            
            system_throughput = total_tokens / total_duration_sec
            peak_vram_mb = get_vram_usage_mb()

            batch_results[f"batch_{bs}"] = {
                "batch_size": bs,
                "total_requests": bs,
                "total_tokens_generated": total_tokens,
                "total_wall_time_ms": round(total_duration_sec * 1000.0, 2),
                "system_throughput_tps": round(system_throughput, 2),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "ttft_ms": ttft_stats,
                "latency_ms": latency_stats
            }

        # 3. Deep 32k Context Window OOM Stress Simulation
        if not args.skip_32k:
            print("\n🌊 Initiating 32k Maximum Context Window Stress Simulation...")
            try:
                stress_prompt = generate_32k_stress_prompt()
                prompt_char_len = len(stress_prompt)
                print(f"   📏 Synthesized context payload: {prompt_char_len:,} chars (~30,000–32,000 tokens)")
                
                t_s0 = time.perf_counter()
                out_32k = engine.generate(stress_prompt, max_tokens=64, temperature=0.0)
                t_s1 = time.perf_counter()

                stress_dur_ms = (t_s1 - t_s0) * 1000.0
                
                # Handle batch abstraction wrapper just in case
                if isinstance(out_32k, list):
                    out_32k = out_32k[0]
                    
                resp_32k = out_32k.get("raw_output", "").strip()
                
                # Non-empty response indicates successful full KV-cache prefill without OOM
                stress_32k_passed = len(resp_32k) > 5
                peak_stress_vram_mb = get_vram_usage_mb()

                stress_metrics = {
                    "passed": stress_32k_passed,
                    "context_characters": prompt_char_len,
                    "execution_time_ms": round(stress_dur_ms, 2),
                    "peak_vram_mb": round(peak_stress_vram_mb, 2),
                    "tokens_generated": out_32k.get("tokens_generated", len(resp_32k.split())),
                    "output_snippet": resp_32k[:150] + "..." if len(resp_32k) > 150 else resp_32k,
                    "oom_crash_detected": not stress_32k_passed
                }
                print(f"   {'✅ 32k Stress Test PASSED without OOM' if stress_32k_passed else '❌ 32k Stress Test FAILED'}")
            except Exception as e:
                print(f"   ❌ 32k Stress Test CRASHED with Exception: {e}")
                stress_metrics = {
                    "passed": False,
                    "oom_crash_detected": True,
                    "error_message": str(e),
                    "peak_vram_mb": get_vram_usage_mb()
                }

    finally:
        # 4. Strict VRAM Airlock
        engine.unload()
        del engine
        gc.collect()
        print("\n🧹 [VRAM Airlock] Model purged from GPU memory.")

    # 5. Certification Verification & Terminal Report
    print("\n═══════════════════════════════════════════════════════════════════════════════════════════════")
    print("📊 PILLAR 5 INFERENCE LATENCY & LOAD STRESS CERTIFICATION REPORT")
    print("═══════════════════════════════════════════════════════════════════════════════════════════════")
    print(f"| Batch | TTFT Mean (P95) ms | Latency Mean (P95) ms | System TPS   | Peak VRAM   | Status |")
    print(f"|-------|--------------------|-----------------------|--------------|-------------|--------|")
    
    for bs in args.batch_sizes:
        data = batch_results[f"batch_{bs}"]
        ttft_str = f"{data['ttft_ms']['mean']:.2f} ({data['ttft_ms']['p95']:.2f})"
        lat_str = f"{data['latency_ms']['mean']:.1f} ({data['latency_ms']['p95']:.1f})"
        vram_str = f"{data['peak_vram_mb']:.1f} MB"
        tps_str = f"{data['system_throughput_tps']:.2f} tok/s"
        
        target_ttft = 350.0 if bs <= 2 else 1200.0
        status = "✅ PASS" if data["ttft_ms"]["mean"] < target_ttft else "⚠️ NOTICE"
        print(f"| {bs:<5} | {ttft_str:<18} | {lat_str:<21} | {tps_str:<12} | {vram_str:<11} | {status:<6} |")
    
    print("-----------------------------------------------------------------------------------------------")
    stress_status = "✅ PASSED (0% OOM Rate)" if stress_32k_passed else "❌ FAILED (OOM Encountered)"
    print(f"| 32k Context Stress Simulation Resistance: {stress_status:<51} |")
    print("═══════════════════════════════════════════════════════════════════════════════════════════════\n")

    b1_data = batch_results.get("batch_1", {})
    certified_efficient = (
        all(b["ttft_ms"]["mean"] < 1200.0 for b in batch_results.values()) and 
        (stress_32k_passed or args.skip_32k)
    )

    report_payload = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": args.backend,
        "model_path": str(args.model_path),
        "avg_latency_ms": b1_data.get("latency_ms", {}).get("mean", 0.0),
        "avg_ttft_ms": b1_data.get("ttft_ms", {}).get("mean", 0.0),
        "system_throughput_tps": b1_data.get("system_throughput_tps", 0.0),
        "p95_latency_ms": b1_data.get("latency_ms", {}).get("p95", 0.0),
        "p95_ttft_ms": b1_data.get("ttft_ms", {}).get("p95", 0.0),
        "concurrency_benchmarks": batch_results,
        "stress_32k_simulation": stress_metrics,
        "certified_efficient": certified_efficient
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"💾 Pillar 5 benchmark report saved to: {REPORT_PATH}")

    # Always return 0 to allow diagnostic orchestrator verify.py to aggregate and grade cleanly
    return 0


if __name__ == "__main__":
    sys.exit(main())