#!/usr/bin/env python3
"""
run_grammar_evals.py – JSON Schema Compliance Evaluation (Universal Dual-Backend Grade)

Tests Pillar 1: Schema Compliance Rate & Pillar 5: Hardware Efficiency.
Evaluates whether the trained SLM outputs valid JSON strictly adhering to dpdp_schema.json.
Supports universal execution via `--backend unsloth | vllm | llamacpp`.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import json
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone
try:
    from jsonschema import validate, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    class ValidationError(Exception):
        pass
from tqdm import tqdm
import numpy as np


from metrics import extract_json_from_output

from backend_loader import BackendEngine, format_chatml_prompt
# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
_EVALS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVALS_DIR.parent.parent
DEFAULT_SCHEMA_PATH = _PROJECT_ROOT / "libs" / "contracts" / "schemas" / "dpdp_schema.json"
DEFAULT_GROUND_TRUTH_PATH = _EVALS_DIR / "holdout_policies" / "ground_truth.json"
DEFAULT_MODEL_PATH = Path("../models/audit-model-final") if Path("../models/audit-model-final").exists() else Path("../models/Qwen3.5-9B")
REPORT_DIR = _EVALS_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

VALID_VIOLATION_TYPES = {
    "PURPOSE_LIMITATION_VIOLATION",
    "CONSENT_NOT_FREE_OR_SPECIFIC",
    "NOTICE_INADEQUATE",
    "DATA_RETENTION_LIMIT_EXCEEDED",
    "CHILD_CONSENT_VIOLATION",
    "SECURITY_SAFEGUARDS_MISSING",
    "GRIEVANCE_REDRESSAL_INADEQUATE",
    "BREACH_NOTIFICATION_FAILURE",
    "SDF_OBLIGATIONS_MISSING",
    "CROSS_BORDER_TRANSFER_VIOLATION"
}

VALID_NETWORK_ACTIONS = {
    "BLOCK_THIRD_PARTY",
    "STRIP_TELEMETRY_HEADER",
    "SPOOF_HARDWARE_API",
    "INJECT_GPC_SIGNAL",
    "WARN_USER_ONLY"
}

REQUIRED_ROOT_FIELDS = ["global_legal_reasoning", "violations", "dpdp_trust_score", "subtlety_score"]
REQUIRED_VIOLATION_FIELDS = ["statute_reference", "violation_type", "evidence_quote", "network_action", "offending_entities"]

# ═══════════════════════════════════════════════════════════════════════════
# DATA & EXTRACTION HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def load_schema(schema_path: Path) -> Dict[str, Any]:
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_test_policies(gt_path: Path) -> List[Dict[str, str]]:
    policies = []
    if gt_path.exists():
        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        for item in ground_truth:
            if 'policy_text_snippet' in item:
                policies.append({
                    "case_id": item.get('case_id', item['filename']),
                    "filename": item['filename'],
                    "category": item.get('category', 'unknown'),
                    "content": item['policy_text_snippet']
                })
            else:
                policy_file = gt_path.parent / item['filename']
                if policy_file.exists():
                    with open(policy_file, 'r', encoding='utf-8') as f:
                        policies.append({
                            "case_id": item.get('case_id', item['filename']),
                            "filename": item['filename'],
                            "category": item.get('category', 'unknown'),
                            "content": f.read()
                        })
        return policies
    
    test_dir = gt_path.parent
    if test_dir.exists():
        for policy_file in sorted(test_dir.glob("*.txt")):
            with open(policy_file, 'r', encoding='utf-8') as f:
                policies.append({
                    "case_id": policy_file.stem,
                    "filename": policy_file.name,
                    "category": "unknown",
                    "content": f.read()
                })
    return policies

def validate_json_structure(output: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "is_valid_json": False,
        "matches_schema": False,
        "error": None,
        "parsed_output": None,
        "missing_fields": [],
        "enum_violations": [],
        "type_errors": []
    }
    extracted = extract_json_from_output(output)
    if not extracted:
        result["error"] = "Empty or unextractable JSON output"
        return result
        
    try:
        parsed = json.loads(extracted)
        result["is_valid_json"] = True
        result["parsed_output"] = parsed
    except json.JSONDecodeError as e:
        result["error"] = f"JSONDecodeError: {e}"
        return result

    if not isinstance(parsed, dict):
        result["error"] = "Root output is not a JSON object (dict)"
        return result

    missing = [f for f in REQUIRED_ROOT_FIELDS if f not in parsed]
    if missing:
        result["missing_fields"] = missing

    if "dpdp_trust_score" in parsed and not isinstance(parsed["dpdp_trust_score"], (int, float)):
        result["type_errors"].append("dpdp_trust_score must be numeric")
    if "subtlety_score" in parsed and not isinstance(parsed["subtlety_score"], (int, float)):
        result["type_errors"].append("subtlety_score must be numeric")

    violations = parsed.get("violations", [])
    if isinstance(violations, list):
        for idx, v in enumerate(violations):
            if not isinstance(v, dict):
                result["type_errors"].append(f"violation[{idx}] is not a dict")
                continue
            missing_v = [f for f in REQUIRED_VIOLATION_FIELDS if f not in v]
            if missing_v:
                result["missing_fields"].extend([f"violation[{idx}].{f}" for f in missing_v])
            v_type = v.get("violation_type")
            if v_type and v_type not in VALID_VIOLATION_TYPES:
                result["enum_violations"].append(f"violation[{idx}].violation_type='{v_type}'")
            v_action = v.get("network_action")
            if v_action and v_action not in VALID_NETWORK_ACTIONS:
                result["enum_violations"].append(f"violation[{idx}].network_action='{v_action}'")
    else:
        result["type_errors"].append("violations must be a list")

    if not result["missing_fields"] and not result["enum_violations"] and not result["type_errors"]:
        if HAS_JSONSCHEMA:
            try:
                validate(instance=parsed, schema=schema)
                result["matches_schema"] = True
            except ValidationError as e:
                result["error"] = f"JSONSchema ValidationError: {e.message if hasattr(e, 'message') else str(e)}"
                result["matches_schema"] = False
        else:
            result["matches_schema"] = True
    else:
        result["error"] = "Structural requirement / enum check failed"

    return result

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Pillar 1: JSON Schema & Grammar Compliance Evaluation")
    parser.add_argument("--backend", type=str, default="llamacpp", choices=["unsloth", "vllm", "llamacpp"])
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--schema-path", type=str, default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--ground-truth-path", type=str, default=str(DEFAULT_GROUND_TRUTH_PATH))
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--lora-name", type=str, default="audit")
    args = parser.parse_args()

    schema = load_schema(Path(args.schema_path))
    policies = load_test_policies(Path(args.ground_truth_path))
    if not policies:
        print("⚠️ No policies found to evaluate.")
        return

    print(f"🚀 Running Grammar & Schema Compliance Evals on {len(policies)} policies across backend: {args.backend}...")
    engine = BackendEngine(
        backend_type=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        vllm_url=args.vllm_url,
        lora_name=args.lora_name
    )

    results = []
    total_valid_json = 0
    total_schema_compliant = 0
    latencies = []
    ttfts = []
    throughputs = []

    for item in tqdm(policies, desc="Evaluating Compliance"):
        sys_msg = "You are a strict DPDP Regulatory Auditor. Output ONLY valid JSON matching the schema."
        user_msg = f"[SYNTHESIZED POLICY]\n{item['content']}"
        prompt = format_chatml_prompt(sys_msg, user_msg)
        inference_out = engine.generate(prompt, max_tokens=1024, temperature=0.0)
        validation = validate_json_structure(inference_out["raw_output"], schema)
        
        if validation["is_valid_json"]:
            total_valid_json += 1
        if validation["matches_schema"]:
            total_schema_compliant += 1
            
        latencies.append(inference_out["latency_ms"])
        if inference_out["ttft_ms"] > 0:
            ttfts.append(inference_out["ttft_ms"])
        if inference_out["tokens_per_sec"] > 0:
            throughputs.append(inference_out["tokens_per_sec"])

        results.append({
            "case_id": item["case_id"],
            "filename": item["filename"],
            "category": item.get("category", "unknown"),
            "is_valid_json": validation["is_valid_json"],
            "matches_schema": validation["matches_schema"],
            "missing_fields": validation["missing_fields"],
            "enum_violations": validation["enum_violations"],
            "type_errors": validation["type_errors"],
            "latency_ms": inference_out["latency_ms"],
            "ttft_ms": inference_out["ttft_ms"],
            "tokens_per_sec": inference_out["tokens_per_sec"]
        })

    n = len(policies)
    json_validity_rate = (total_valid_json / n) * 100
    schema_compliance_rate = (total_schema_compliant / n) * 100
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    avg_ttft = float(np.mean(ttfts)) if ttfts else 0.0
    avg_throughput = float(np.mean(throughputs)) if throughputs else 0.0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model_path": args.model_path,
        "total_policies_evaluated": n,
        "json_validity_rate": round(json_validity_rate, 2),
        "schema_compliance_rate": round(schema_compliance_rate, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_ttft_ms": round(avg_ttft, 2),
        "avg_tokens_per_sec": round(avg_throughput, 2),
        "details": results
    }

    report_path = REPORT_DIR / "grammar_compliance_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "═"*70)
    print("📊 PILLAR 1 & 5: GRAMMAR & SCHEMA COMPLIANCE SUMMARY")
    print("═"*70)
    print(f"   • Total Policies Evaluated: {n}")
    print(f"   • JSON Validity Rate:       {json_validity_rate:.2f}%")
    print(f"   • Schema Compliance Rate:   {schema_compliance_rate:.2f}% (Threshold: >= 98.0%)")
    print(f"   • Average Latency:          {avg_latency:.2f} ms")
    print(f"   • Average TTFT:             {avg_ttft:.2f} ms")
    print(f"   • Average Throughput:       {avg_throughput:.2f} tokens/sec")
    print(f"💾 Detailed report saved to: {report_path}")
    print("═"*70 + "\n")

if __name__ == "__main__":
    main()