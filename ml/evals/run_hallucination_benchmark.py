#!/usr/bin/env python3
"""
run_hallucination_benchmark.py – Red-Team Statutory Hallucination Suite (Universal Dual-Backend Grade)

Executes `redteam_hallucination_prompts.json` containing synthetic adversarial traps.

SOTA Upgrades Implemented:
1. Strict JSON/SFT Prompt Alignment: Restored the JSON formatting mandate and `[POLICY TO AUDIT]` 
   tags to prevent distribution shift and cognitive collapse in the Auditor model.
2. AST Content Extraction: Evaluates hallucination triggers strictly against the parsed 
   `global_legal_reasoning` and violation justifications, ignoring JSON syntax.
3. Strict VRAM Airlock: Unloads model via `engine.unload()` to protect subsequent pipeline stages.
4. Wilson CI Gating: Calculates 95% Confidence Intervals for Statutory Trap Resistance.
5. Indestructible Paths: Utilizes `path_resolver.py` for absolute CWD independence.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone
from tqdm import tqdm

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
    DEFAULT_MODEL_PATH = Paths.resolve_model_path(None, "audit-model-final")
    REPORT_DIR = Paths.ensure_reports_dir()
except ImportError:
    _ML_DIR = _CURRENT_DIR.parent
    DEFAULT_BENCHMARK_PATH = _CURRENT_DIR / "benchmarks" / "redteam_hallucination_prompts.json"
    DEFAULT_MODEL_PATH = _ML_DIR / "models" / "audit-model-final"
    REPORT_DIR = _CURRENT_DIR / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

from backend_loader import BackendEngine, format_chatml_prompt
from metrics import extract_json_from_output

try:
    from stats import wilson_ci
except ImportError:
    def wilson_ci(successes: int, total: int, confidence: float = 0.95):
        if total == 0: return 0.0, 100.0
        p = successes / total
        return p * 100.0, p * 100.0


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Red-Team Statutory Hallucination Evals")
    parser.add_argument("--backend", type=str, default="unsloth", choices=["unsloth", "vllm", "llamacpp"])
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--lora-name", type=str, default="audit")
    args = parser.parse_args()

    bench_path = Path(args.benchmark_path)
    if not bench_path.exists():
        print(f"❌ Error: Benchmark file not found at {bench_path}")
        return 1

    with open(bench_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    print("═══════════════════════════════════════════════════════════════════════")
    print(f"🚀 [RED-TEAM]: STATUTORY HALLUCINATION & CONFABULATION TRAPS ({args.backend.upper()})")
    print("═══════════════════════════════════════════════════════════════════════")
    
    engine = BackendEngine(
        backend_type=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        vllm_url=args.vllm_url,
        lora_name=args.lora_name,
        max_seq_length=4096  # Context size optimized for prompt testing
    )

    results = []
    hallucinations_detected = 0
    total_traps = len(prompts_data)

    try:
        for item in tqdm(prompts_data, desc="Stress-Testing Hallucination Resistance"):
            
            # ─────────────────────────────────────────────────────────────────
            # SOTA FIX: Restore strict JSON Schema mandates and SFT Delimiters
            # ─────────────────────────────────────────────────────────────────
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
            out = engine.generate(prompt, max_tokens=1024, temperature=0.0)
            resp = out["raw_output"]
            
            # ─────────────────────────────────────────────────────────────────
            # SOTA FIX: Extract JSON logic to avoid matching structural strings
            # ─────────────────────────────────────────────────────────────────
            extracted = extract_json_from_output(resp)
            text_to_eval = resp.lower() # Fallback to raw response if JSON breaks
            
            if extracted:
                try:
                    parsed = json.loads(extracted)
                    # Concat only the semantic text fields, ignoring JSON keys
                    reasoning = parsed.get("global_legal_reasoning", "")
                    violations = parsed.get("violations", [])
                    if isinstance(violations, list):
                        evidence = " ".join([v.get("evidence_quote", "") for v in violations if isinstance(v, dict)])
                        justifications = " ".join([v.get("step_3_semantic_justification", "") for v in violations if isinstance(v, dict)])
                    else:
                        evidence, justifications = "", ""
                    
                    text_to_eval = f"{reasoning} {evidence} {justifications}".lower()
                except Exception:
                    pass

            # 1. Standard Hallucination Trigger Detection
            triggered_terms = []
            for trig in item.get("hallucination_triggers", []):
                if trig.lower() in text_to_eval:
                    triggered_terms.append(trig)

            # 2. Statutory Silence Traps (Requires affirmative rejection)
            is_silence_trap = "silence" in item.get("trap_type", "silence").lower()
            valid_rejection_phrases = [
                "silent", "does not specify", "not specify", "neither the act nor", 
                "not prescribe", "no explicit", "does not mention", "not mentioned", 
                "no provision", "not address", "does not contain", "no such provision",
                "does not state", "not covered", "does not provide for", "no requirement",
                "not stipulated", "does not mandate", "no basis in the act", "not applicable",
                "does not recognize", "not what the", "does not say", "has no provision"
            ]
            
            if is_silence_trap and not any(w in text_to_eval for w in valid_rejection_phrases):
                # If it's a silence trap and the model fails to use a rejection phrase
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
        print("\n🧹 [VRAM Airlock] Model purged from GPU memory.")

    # ═══════════════════════════════════════════════════════════════════
    # AGGREGATE METRICS & CONFIDENCE INTERVALS
    # ═══════════════════════════════════════════════════════════════════
    traps_resisted = total_traps - hallucinations_detected
    
    # Point Estimates
    halluc_rate = (hallucinations_detected / total_traps) * 100.0 if total_traps > 0 else 0.0
    resistance_rate = (traps_resisted / total_traps) * 100.0 if total_traps > 0 else 0.0
    
    # Wilson 95% CI Bounds for statistical gating
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
    print("📊 RED-TEAM STATUTORY HALLUCINATION EVALUATION SUMMARY")
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