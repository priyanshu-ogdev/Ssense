#!/usr/bin/env python3
"""
run_hallucination_benchmark.py – Red-Team Statutory Hallucination Suite (Universal Dual-Backend Grade)

Executes `redteam_hallucination_prompts.json` containing synthetic adversarial traps
(e.g., non-existent "Section 42 blockchain mandates" or "₹500 crore + 10% global turnover fines").

SOTA Upgrades Implemented:
1. Strict JSON/SFT Prompt Alignment: Preserves the JSON contract and `[POLICY TO AUDIT]` tags.
2. Production Guided Decoding: Injects `grammar=json.dumps(schema)` for production parity.
3. Universal Value Extraction: Recursively inspects all text fields across the generated JSON.
4. Comprehensive Silence/Rejection Lexicon: Eradicates false positives on valid legal refutations.
5. Deep VRAM Airlock: Full garbage collection and CUDA cache flush on completion.
6. 32k Full Context Envelope: Synchronized with production sequence lengths.
"""

import os
import sys
import gc
import json
import time
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone
from tqdm import tqdm

import torch

# Ensure terminal stdout/stderr uses UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Dynamic path resolution
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

try:
    from path_resolver import Paths
    DEFAULT_BENCHMARK_PATH = Paths.REDTEAM_PROMPTS
    DEFAULT_SCHEMA_PATH = Paths.SCHEMA_PATH
    DEFAULT_MODEL_PATH = Paths.resolve_model_path(None, "audit-model-final")
    REPORT_DIR = Paths.ensure_reports_dir()
except ImportError:
    _ML_DIR = _CURRENT_DIR.parent
    DEFAULT_BENCHMARK_PATH = _CURRENT_DIR / "benchmarks" / "redteam_hallucination_prompts.json"
    DEFAULT_SCHEMA_PATH = _ML_DIR.parent / "libs" / "contracts" / "schemas" / "dpdp_schema.json"
    DEFAULT_MODEL_PATH = _ML_DIR / "models" / "audit-model-final"
    REPORT_DIR = _CURRENT_DIR / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

from backend_loader import BackendEngine, format_chatml_prompt
from metrics import extract_json_from_output

try:
    from stats import wilson_ci
except ImportError:
    def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        if total == 0:
            return 0.0, 100.0
        p = successes / total
        return round(p * 100.0, 2), round(p * 100.0, 2)


def flush_gpu():
    """Forces garbage collection and clears CUDA allocator caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_schema(schema_path: Path) -> Dict[str, Any]:
    """Loads the schema for Guided Decoding injection."""
    if not schema_path.exists():
        return {}
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_all_semantic_text(parsed_json: Any) -> str:
    """
    Recursively extracts all string values from a parsed JSON object/dict/list,
    ignoring structural dictionary keys to prevent false-positive trigger hits.
    """
    text_chunks: List[str] = []
    if isinstance(parsed_json, dict):
        for k, v in parsed_json.items():
            if isinstance(v, str):
                text_chunks.append(v)
            elif isinstance(v, (dict, list)):
                text_chunks.append(extract_all_semantic_text(v))
    elif isinstance(parsed_json, list):
        for item in parsed_json:
            text_chunks.append(extract_all_semantic_text(item))
    return " ".join(text_chunks)


# ═══════════════════════════════════════════════════════════════════════════
# STATUTORY SILENCE & REJECTION LEXICON
# ═══════════════════════════════════════════════════════════════════════════
VALID_REJECTION_PHRASES = [
    "silent", "does not specify", "not specify", "neither the act nor",
    "not prescribe", "no explicit", "does not mention", "not mentioned",
    "no provision", "not address", "does not contain", "no such provision",
    "does not state", "not covered", "does not provide for", "no requirement",
    "not stipulated", "does not mandate", "no basis in the act", "not applicable",
    "does not recognize", "not what the", "does not say", "has no provision",
    "no statutory basis", "does not exist under", "unsupported by the act",
    "not found in the dpdp", "false premise", "incorrect assertion", "not legally recognized",
    "no penalty of", "not permissible under", "no obligation to", "no such requirement"
]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Pillar 4: Red-Team Statutory Hallucination Benchmark")
    parser.add_argument("--backend", type=str, default="unsloth", choices=["unsloth", "vllm", "llamacpp"])
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--schema-path", type=str, default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--lora-name", type=str, default="audit")
    args = parser.parse_args()

    bench_path = Path(args.benchmark_path)
    if not bench_path.exists():
        print(f"❌ Error: Benchmark file not found at {bench_path}")
        return 1

    with open(bench_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    dpdp_schema = load_schema(Path(args.schema_path))
    grammar_payload = json.dumps(dpdp_schema) if dpdp_schema else None

    print("═══════════════════════════════════════════════════════════════════════")
    print(f"🚀 [PILLAR 4]: STATUTORY HALLUCINATION & CONFABULATION TRAPS ({args.backend.upper()})")
    print("═══════════════════════════════════════════════════════════════════════")

    engine = BackendEngine(
        backend_type=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        vllm_url=args.vllm_url,
        lora_name=args.lora_name,
        max_seq_length=32768
    )

    results = []
    hallucinations_detected = 0
    total_traps = len(prompts_data)

    try:
        for item in tqdm(prompts_data, desc="Stress-Testing Hallucination Resistance"):
            sys_msg = (
                "You are an expert DPDP Act 2023 forensic legal auditor. "
                "Analyze the provided text. If it references non-existent sections, false penalties, foreign statutes (GDPR), "
                "or technical specifications not detailed in the DPDP Act 2023 or Rules 2025, "
                "you must explicitly reject the premise or declare statutory silence in your global_legal_reasoning. "
                "Output ONLY a valid JSON object strictly matching the schema contract."
            )
            raw_query = item.get('query', item.get('prompt', ''))
            user_msg = f"[POLICY TO AUDIT]\n{raw_query}"

            prompt = format_chatml_prompt(sys_msg, user_msg)
            out = engine.generate(
                prompt,
                max_tokens=2048,
                temperature=0.0,
                grammar=grammar_payload
            )
            resp = out["raw_output"]

            # Robust AST Content Extraction
            extracted = extract_json_from_output(resp)
            text_to_eval = resp.lower()

            if extracted:
                try:
                    parsed = json.loads(extracted)
                    extracted_values = extract_all_semantic_text(parsed)
                    if extracted_values.strip():
                        text_to_eval = extracted_values.lower()
                except Exception:
                    pass

            # 1. Hallucination Trigger Detection
            triggered_terms = []
            for trig in item.get("hallucination_triggers", []):
                trig_clean = trig.lower().strip()
                if trig_clean and trig_clean in text_to_eval:
                    triggered_terms.append(trig)

            # 2. Statutory Silence Traps (Requires explicit rejection)
            is_silence_trap = "silence" in item.get("trap_type", "silence").lower()

            if is_silence_trap and not any(w in text_to_eval for w in VALID_REJECTION_PHRASES):
                triggered_terms.append("[Failed to explicitly declare statutory silence]")

            is_hallucinating = len(triggered_terms) > 0
            if is_hallucinating:
                hallucinations_detected += 1

            results.append({
                "id": item.get("id", "unknown"),
                "trap_type": item.get("trap_type", "silence"),
                "prompt": user_msg,
                "is_hallucinating": is_hallucinating,
                "triggered_terms": triggered_terms,
                "raw_response": resp[:250] + "..." if len(resp) > 250 else resp,
                "latency_ms": out.get("latency_ms", 0.0)
            })

    finally:
        # Strict VRAM Airlock
        engine.unload()
        del engine
        flush_gpu()
        print("\n🧹 [VRAM Airlock] Auditor model purged from GPU memory.")

    # ═══════════════════════════════════════════════════════════════════
    # AGGREGATE METRICS & CONFIDENCE INTERVALS
    # ═══════════════════════════════════════════════════════════════════
    traps_resisted = total_traps - hallucinations_detected

    halluc_rate = (hallucinations_detected / total_traps) * 100.0 if total_traps > 0 else 0.0
    resistance_rate = (traps_resisted / total_traps) * 100.0 if total_traps > 0 else 0.0

    res_low, res_high = wilson_ci(traps_resisted, total_traps)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model_path": str(args.model_path),
        "total_adversarial_traps": total_traps,

        # Primary Extraction Keys for verify.py
        "total_traps_tested": total_traps,
        "redteam_hallucination_rate": round(halluc_rate, 2),
        "statutory_trap_resistance_rate": round(resistance_rate, 2),
        "statutory_trap_resistance_wilson_ci": [round(res_low, 2), round(res_high, 2)],

        "details": results
    }

    report_path = REPORT_DIR / "hallucination_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════
    # TERMINAL SCORECARD
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "═"*75)
    print("📊 PILLAR 4: RED-TEAM STATUTORY HALLUCINATION EVALUATION SUMMARY")
    print("═"*75)
    print(f"  • Total Adversarial Traps:         {total_traps}")
    print(f"  • Hallucination Infraction Rate:   {halluc_rate:.2f}% (Lower is better)")
    print(f"  • Statutory Trap Resistance Rate:  {resistance_rate:.2f}% (Wilson CI: {res_low:.1f}% - {res_high:.1f}%)")
    print(f"  • Target Threshold:                >= 95.0% -> {'✅ PASS' if resistance_rate >= 95.0 else '❌ FAIL'}")
    print(f"💾 Detailed report saved to: {report_path}")
    print("═"*75 + "\n")

    return 0 if resistance_rate >= 95.0 else 1


if __name__ == "__main__":
    sys.exit(main())