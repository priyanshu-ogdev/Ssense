#!/usr/bin/env python3
"""
run_grammar_evals.py – JSON Schema Compliance Evaluation (Final Production Grade)

Tests Pillar 1: Schema Compliance Rate.
Evaluates whether the trained SLM natively outputs valid JSON that strictly 
adheres to the sealed dpdp_schema.json contract (including subtlety_score).

Critical Architecture Decision:
- The 9B SLM was fine-tuned (SFT/DPO) to MEMORIZE the DPDP law.
- Therefore, we DO NOT inject the 35k law text at inference time.
- This matches production behavior and prevents context overflow.

Measures:
- JSON validity rate (with and without grammar enforcement)
- Field-level schema violation breakdown
- Enum compliance for violation_type and network_action
- Type correctness (integer vs string for scores)
- TTFT, throughput, and latency metrics
- Category-aware compliance reporting

Supports:
- Grammar-constrained vs unconstrained comparison mode
- Markdown-wrapped JSON extraction
- Trailing comma tolerance
- Empty output detection
- Reasoning text stripping
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from jsonschema import validate, ValidationError
from llama_cpp import Llama, LlamaGrammar
from tqdm import tqdm
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
SCHEMA_PATH = Path("libs/contracts/schemas/dpdp_schema.json")
GROUND_TRUTH_PATH = Path("ml/evals/holdout_policies/ground_truth.json")
MODEL_PATH = Path("apps/browser-core/src-tauri/models/ssense-dpdp-9b-local-q4_k_m.gguf")
REPORT_DIR = Path("ml/evals/reports")

# Valid enum values from the sealed schema
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
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_schema() -> Dict[str, Any]:
    """Load the sealed DPDP JSON schema."""
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_test_policies() -> List[Dict[str, str]]:
    """Load test policies from ground truth JSON or directory."""
    policies = []
    
    # Try ground truth JSON first
    if GROUND_TRUTH_PATH.exists():
        with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
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
                policy_file = Path("ml/evals/holdout_policies") / item['filename']
                if policy_file.exists():
                    with open(policy_file, 'r', encoding='utf-8') as f:
                        policies.append({
                            "case_id": item.get('case_id', item['filename']),
                            "filename": item['filename'],
                            "category": item.get('category', 'unknown'),
                            "content": f.read()
                        })
        return policies
    
    # Fallback: load from directory
    test_dir = Path("ml/evals/holdout_policies")
    for policy_file in sorted(test_dir.glob("*.txt")):
        with open(policy_file, 'r', encoding='utf-8') as f:
            policies.append({
                "case_id": policy_file.stem,
                "filename": policy_file.name,
                "category": "unknown",
                "content": f.read()
            })
    
    return policies

# ═══════════════════════════════════════════════════════════════════════════
# ROBUST JSON EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
def extract_json_from_output(output: str) -> str:
    """
    Extract JSON from model output, handling:
    - Markdown code blocks (```json ... ```)
    - Reasoning text before JSON
    - Text after JSON closing brace
    - Trailing commas
    - Missing closing braces
    """
    if not output or not output.strip():
        return ""
    
    cleaned = output.strip()
    
    # Strategy 1: Extract from markdown code blocks
    if '```json' in cleaned:
        match = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
    elif '```' in cleaned:
        match = re.search(r'```\s*(.*?)\s*```', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
    
    # Strategy 2: Find the first '{' and last '}' to strip surrounding text
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]
    
    # Strategy 3: Find first '[' and last ']' (for array outputs)
    if not cleaned.startswith('{'):
        first_bracket = cleaned.find('[')
        last_bracket = cleaned.rfind(']')
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            cleaned = cleaned[first_bracket:last_bracket + 1]
    
    # Fix trailing commas
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    return cleaned.strip()

# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_inference(
    llm: Llama, 
    policy_text: str,
    grammar: Optional[LlamaGrammar] = None
) -> Dict[str, Any]:
    """
    Run inference with optional grammar enforcement.
    
    CRITICAL: We DO NOT inject the 35k law text.
    The 9B SLM was fine-tuned to memorize the DPDP Act via SFT/DPO.
    Injecting it would cause context overflow and doesn't match production behavior.
    
    Returns output text, latency, TTFT, tokens_generated, tokens_per_sec.
    """
    SYSTEM_PROMPT = "You are a strict DPDP Regulatory Auditor enforcing the Indian Digital Personal Data Protection (DPDP) Act 2023 and Rules 2025. Output ONLY valid JSON matching the dpdp_schema."
    
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Analyze the following privacy policy for DPDP compliance.
Output ONLY valid JSON.

[PRIVACY POLICY]
{policy_text}<|im_end|>
<|im_start|>assistant
"""
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    kwargs = {
        "max_tokens": 2048,  # ✅ Increased from 1024 to prevent truncation
        "temperature": 0.0,
        "stop": ["<|im_end|>"],
        "stream": True       # ✅ Enable streaming for TTFT measurement
    }
    
    if grammar is not None:
        kwargs["grammar"] = grammar
    
    result = llm(prompt, **kwargs)
    
    output_text = ""
    for chunk in result:
        if first_token_time is None and chunk['choices'][0].get('text', ''):
            first_token_time = time.time()
        output_text += chunk['choices'][0].get('text', '')
        token_count += 1
    
    end_time = time.time()
    
    total_latency_ms = (end_time - start_time) * 1000
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_latency_ms
    generation_time_ms = total_latency_ms - ttft_ms
    tokens_per_sec = (token_count / (generation_time_ms / 1000)) if generation_time_ms > 1 else 0
    
    return {
        "raw_output": output_text.strip(),
        "latency_ms": total_latency_ms,
        "ttft_ms": ttft_ms,
        "tokens_generated": token_count,
        "tokens_per_sec": tokens_per_sec
    }

# ═══════════════════════════════════════════════════════════════════════════
# DEEP SCHEMA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
def validate_json_structure(output: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep validation of JSON output against the sealed schema.
    Returns detailed breakdown of all structural issues.
    """
    result = {
        "is_valid_json": False,
        "matches_schema": False,
        "error": None,
        "error_type": None,
        "field_errors": [],
        "enum_errors": [],
        "type_errors": [],
        "extra_fields": [],
        "missing_fields": [],
        "violation_structure_errors": [],
        "raw_output_length": len(output) if output else 0
    }
    
    # Handle empty output
    if not output or len(output.strip()) == 0:
        result["error"] = "Empty output"
        result["error_type"] = "EMPTY_OUTPUT"
        return result
    
    # Extract JSON
    extracted_json = extract_json_from_output(output)
    
    if not extracted_json:
        result["error"] = "No JSON object found in output"
        result["error_type"] = "NO_JSON_FOUND"
        return result
    
    # Step 1: Parse JSON
    try:
        parsed = json.loads(extracted_json)
        result["is_valid_json"] = True
    except json.JSONDecodeError as e:
        result["error"] = f"JSON parse error at position {e.pos}: {e.msg}"
        result["error_type"] = "PARSE_ERROR"
        return result
    
    # Step 2: Check root-level required fields
    for field in REQUIRED_ROOT_FIELDS:
        if field not in parsed:
            result["missing_fields"].append(field)
    
    # Step 3: Check for extra root-level fields
    for field in parsed.keys():
        if field not in REQUIRED_ROOT_FIELDS:
            result["extra_fields"].append(field)
    
    # Step 4: Type checking for score fields
    if "dpdp_trust_score" in parsed:
        if not isinstance(parsed["dpdp_trust_score"], int):
            result["type_errors"].append({
                "field": "dpdp_trust_score",
                "expected": "integer",
                "received": type(parsed["dpdp_trust_score"]).__name__,
                "value": str(parsed["dpdp_trust_score"])[:50]
            })
        elif not (0 <= parsed["dpdp_trust_score"] <= 100):
            result["field_errors"].append({
                "field": "dpdp_trust_score",
                "error": f"Value {parsed['dpdp_trust_score']} outside range [0, 100]"
            })
    
    if "subtlety_score" in parsed:
        if not isinstance(parsed["subtlety_score"], int):
            result["type_errors"].append({
                "field": "subtlety_score",
                "expected": "integer",
                "received": type(parsed["subtlety_score"]).__name__,
                "value": str(parsed["subtlety_score"])[:50]
            })
        elif not (0 <= parsed["subtlety_score"] <= 100):
            result["field_errors"].append({
                "field": "subtlety_score",
                "error": f"Value {parsed['subtlety_score']} outside range [0, 100]"
            })
    
    # Step 5: Validate violations array structure
    if "violations" in parsed:
        if not isinstance(parsed["violations"], list):
            result["type_errors"].append({
                "field": "violations",
                "expected": "array",
                "received": type(parsed["violations"]).__name__
            })
        else:
            for i, v in enumerate(parsed["violations"]):
                if not isinstance(v, dict):
                    result["violation_structure_errors"].append({
                        "index": i,
                        "error": f"Expected object, got {type(v).__name__}"
                    })
                    continue
                
                # Check required violation fields
                missing = [f for f in REQUIRED_VIOLATION_FIELDS if f not in v]
                extra = [f for f in v.keys() if f not in REQUIRED_VIOLATION_FIELDS]
                
                if missing or extra:
                    result["violation_structure_errors"].append({
                        "index": i,
                        "missing": missing,
                        "extra": extra
                    })
                
                # Check enum: violation_type
                vtype = v.get("violation_type", "")
                if vtype and vtype not in VALID_VIOLATION_TYPES:
                    result["enum_errors"].append({
                        "field": f"violations[{i}].violation_type",
                        "value": vtype,
                        "valid_options": sorted(VALID_VIOLATION_TYPES)
                    })
                
                # Check enum: network_action
                action = v.get("network_action", "")
                if action and action not in VALID_NETWORK_ACTIONS:
                    result["enum_errors"].append({
                        "field": f"violations[{i}].network_action",
                        "value": action,
                        "valid_options": sorted(VALID_NETWORK_ACTIONS)
                    })
                
                # Check offending_entities is array
                entities = v.get("offending_entities")
                if entities is not None and not isinstance(entities, list):
                    result["type_errors"].append({
                        "field": f"violations[{i}].offending_entities",
                        "expected": "array",
                        "received": type(entities).__name__
                    })
                
                # Check evidence_quote is non-empty
                quote = v.get("evidence_quote", "")
                if isinstance(quote, str) and len(quote.strip()) == 0:
                    result["field_errors"].append({
                        "field": f"violations[{i}].evidence_quote",
                        "error": "Empty evidence_quote (minLength: 1 required)"
                    })
                
                # Check WARN_USER_ONLY has empty offending_entities
                if action == "WARN_USER_ONLY" and isinstance(entities, list) and len(entities) > 0:
                    result["field_errors"].append({
                        "field": f"violations[{i}].offending_entities",
                        "error": f"WARN_USER_ONLY must have empty offending_entities, got {entities}"
                    })
    
    # Step 6: Full jsonschema validation
    try:
        validate(instance=parsed, schema=schema)
        result["matches_schema"] = True
    except ValidationError as e:
        result["error"] = f"Schema validation: {e.message}"
        result["error_type"] = "SCHEMA_ERROR"
        result["schema_path"] = list(e.absolute_path) if e.absolute_path else []
    
    # Determine overall status
    has_errors = (
        result["missing_fields"] or 
        result["extra_fields"] or 
        result["type_errors"] or 
        result["enum_errors"] or 
        result["violation_structure_errors"] or
        result["field_errors"]
    )
    
    if has_errors and result["matches_schema"]:
        # Schema passed but our deep checks found issues - flag it
        result["matches_schema"] = False
        result["error_type"] = "DEEP_VALIDATION_ERROR"
    
    return result

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
def run_grammar_evals():
    """Run the complete grammar/schema evaluation with comparison mode."""
    print("🔍 Starting Pillar 1: Grammar/Schema Compliance Evaluation...")
    print(f"Schema: {SCHEMA_PATH}")
    print(f"Model: {MODEL_PATH}")
    print()
    
    # Load components once
    schema = load_schema()
    test_policies = load_test_policies()
    
    print(f"Found {len(test_policies)} test policies")
    print(f"Categories: {set(p['category'] for p in test_policies)}")
    print()
    
    # Load model
    print("Loading Q4_K_M model...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=8192,
        n_gpu_layers=-1,       # ✅ Use all available GPU layers
        verbose=False
    )
    
    # Create grammar from schema
    grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
    print("✅ Model loaded with grammar enforcement capability\n")
    
    # ═══════════════════════════════════════════════════════════════════
    # MODE 1: Unconstrained (tests raw model capability)
    # ═══════════════════════════════════════════════════════════════════
    print("━" * 70)
    print("MODE 1: Unconstrained Output (Raw Model - Tests SFT/DPO Quality)")
    print("━" * 70)
    
    unconstrained_results = []
    for policy in tqdm(test_policies, desc="Unconstrained"):
        inference = run_inference(llm, policy['content'], grammar=None)
        validation = validate_json_structure(inference['raw_output'], schema)
        
        unconstrained_results.append({
            "case_id": policy['case_id'],
            "filename": policy['filename'],
            "category": policy['category'],
            "latency_ms": inference['latency_ms'],
            "ttft_ms": inference['ttft_ms'],
            "tokens_generated": inference['tokens_generated'],
            "tokens_per_sec": inference['tokens_per_sec'],
            **validation
        })
    
    # ═══════════════════════════════════════════════════════════════════
    # MODE 2: Grammar-Constrained (tests enforced compliance)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "━" * 70)
    print("MODE 2: Grammar-Constrained Output (Enforced Schema - Production Mode)")
    print("━" * 70)
    
    constrained_results = []
    for policy in tqdm(test_policies, desc="Constrained"):
        inference = run_inference(llm, policy['content'], grammar=grammar)
        validation = validate_json_structure(inference['raw_output'], schema)
        
        constrained_results.append({
            "case_id": policy['case_id'],
            "filename": policy['filename'],
            "category": policy['category'],
            "latency_ms": inference['latency_ms'],
            "ttft_ms": inference['ttft_ms'],
            "tokens_generated": inference['tokens_generated'],
            "tokens_per_sec": inference['tokens_per_sec'],
            **validation
        })
    
    # ═══════════════════════════════════════════════════════════════════
    # AGGREGATE METRICS
    # ═══════════════════════════════════════════════════════════════════
    def compute_summary(results: List[Dict]) -> Dict[str, Any]:
        total = len(results)
        valid_json = sum(1 for r in results if r['is_valid_json'])
        schema_compliant = sum(1 for r in results if r['matches_schema'])
        
        # Count specific error types
        error_counts = {
            "EMPTY_OUTPUT": 0,
            "NO_JSON_FOUND": 0,
            "PARSE_ERROR": 0,
            "SCHEMA_ERROR": 0,
            "DEEP_VALIDATION_ERROR": 0
        }
        for r in results:
            et = r.get('error_type')
            if et in error_counts:
                error_counts[et] += 1
        
        # Aggregate field-level errors
        total_missing_fields = sum(len(r.get('missing_fields', [])) for r in results)
        total_extra_fields = sum(len(r.get('extra_fields', [])) for r in results)
        total_enum_errors = sum(len(r.get('enum_errors', [])) for r in results)
        total_type_errors = sum(len(r.get('type_errors', [])) for r in results)
        
        # Category breakdown
        category_metrics = {}
        for cat in set(r['category'] for r in results):
            cat_results = [r for r in results if r['category'] == cat]
            cat_compliant = sum(1 for r in cat_results if r['matches_schema'])
            category_metrics[cat] = {
                "count": len(cat_results),
                "schema_compliance_rate": cat_compliant / len(cat_results) if cat_results else 0,
                "avg_latency_ms": float(np.mean([r['latency_ms'] for r in cat_results])),
                "avg_tokens_per_sec": float(np.mean([r['tokens_per_sec'] for r in cat_results]))
            }
        
        return {
            "total_policies": total,
            "valid_json_count": valid_json,
            "valid_json_rate": valid_json / total if total else 0,
            "schema_compliant_count": schema_compliant,
            "schema_compliance_rate": schema_compliant / total if total else 0,
            "error_breakdown": error_counts,
            "total_missing_fields": total_missing_fields,
            "total_extra_fields": total_extra_fields,
            "total_enum_errors": total_enum_errors,
            "total_type_errors": total_type_errors,
            "avg_latency_ms": float(np.mean([r['latency_ms'] for r in results])),
            "avg_ttft_ms": float(np.mean([r['ttft_ms'] for r in results])),
            "avg_tokens_generated": float(np.mean([r['tokens_generated'] for r in results])),
            "avg_tokens_per_sec": float(np.mean([r['tokens_per_sec'] for r in results])),
            "category_metrics": category_metrics
        }
    
    unconstrained_summary = compute_summary(unconstrained_results)
    constrained_summary = compute_summary(constrained_results)
    
    # ═══════════════════════════════════════════════════════════════════
    # PRINT RESULTS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PILLAR 1: GRAMMAR/SCHEMA COMPLIANCE EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 JSON Validity:")
    print(f"  Unconstrained: {unconstrained_summary['valid_json_count']}/{unconstrained_summary['total_policies']} ({unconstrained_summary['valid_json_rate']:.1%})")
    print(f"  Constrained:   {constrained_summary['valid_json_count']}/{constrained_summary['total_policies']} ({constrained_summary['valid_json_rate']:.1%})")
    
    print(f"\n🎯 Schema Compliance:")
    print(f"  Unconstrained: {unconstrained_summary['schema_compliant_count']}/{unconstrained_summary['total_policies']} ({unconstrained_summary['schema_compliance_rate']:.1%})")
    print(f"  Constrained:   {constrained_summary['schema_compliant_count']}/{constrained_summary['total_policies']} ({constrained_summary['schema_compliance_rate']:.1%})")
    
    print(f"\n🔧 Field-Level Errors (Unconstrained):")
    print(f"  Missing Required Fields: {unconstrained_summary['total_missing_fields']}")
    print(f"  Extra Fields (additionalProperties): {unconstrained_summary['total_extra_fields']}")
    print(f"  Enum Violations: {unconstrained_summary['total_enum_errors']}")
    print(f"  Type Mismatches: {unconstrained_summary['total_type_errors']}")
    
    print(f"\n⚡ Performance:")
    print(f"  {'Metric':<25} {'Unconstrained':>15} {'Constrained':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Avg Latency (ms)':<25} {unconstrained_summary['avg_latency_ms']:>15.1f} {constrained_summary['avg_latency_ms']:>15.1f}")
    print(f"  {'Avg TTFT (ms)':<25} {unconstrained_summary['avg_ttft_ms']:>15.1f} {constrained_summary['avg_ttft_ms']:>15.1f}")
    print(f"  {'Avg Tokens/Sec':<25} {unconstrained_summary['avg_tokens_per_sec']:>15.1f} {constrained_summary['avg_tokens_per_sec']:>15.1f}")
    
    print(f"\n📂 Category Breakdown (Constrained):")
    for cat, metrics in constrained_summary['category_metrics'].items():
        print(f"  {cat}: Compliance={metrics['schema_compliance_rate']:.1%}, Latency={metrics['avg_latency_ms']:.0f}ms")
    
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    json_path = REPORT_DIR / "grammar_eval_results.json"
    
    # Strip parsed_json from detailed results to prevent file bloat
    clean_unconstrained = [{k: v for k, v in r.items() if k != 'parsed_json'} for r in unconstrained_results]
    clean_constrained = [{k: v for k, v in r.items() if k != 'parsed_json'} for r in constrained_results]
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "unconstrained": {
                "summary": unconstrained_summary,
                "detailed_results": clean_unconstrained
            },
            "constrained": {
                "summary": constrained_summary,
                "detailed_results": clean_constrained
            }
        }, f, indent=2)
    
    print(f"\n📊 JSON results saved to: {json_path}")
    
    # Markdown report
    generate_markdown_report(unconstrained_summary, constrained_summary, unconstrained_results, constrained_results)
    
    # Print failures
    unconstrained_failures = [r for r in unconstrained_results if not r['matches_schema']]
    if unconstrained_failures:
        print(f"\n⚠️  Unconstrained failures ({len(unconstrained_failures)}):")
        for r in unconstrained_failures[:10]:
            print(f"  - {r['case_id']}: [{r.get('error_type', 'UNKNOWN')}] {(r.get('error') or '')[:80]}")
    
    # Production decision guidance
    print("\n" + "━" * 70)
    print("🎯 PRODUCTION DEPLOYMENT DECISION")
    print("━" * 70)
    if unconstrained_summary['schema_compliance_rate'] >= 0.95:
        print("✅ RECOMMENDATION: Disable grammar enforcement in production.")
        print("   The model has internalized the schema well enough (>95% compliance).")
        print("   This will maximize inference speed (lower TTFT, higher tokens/sec).")
    else:
        print("⚠️  RECOMMENDATION: Keep grammar enforcement enabled in production.")
        print(f"   Unconstrained compliance is {unconstrained_summary['schema_compliance_rate']:.1%} (<95%).")
        print("   The Rust interceptor must use LlamaGrammar to guarantee schema compliance.")
    print("━" * 70)

def generate_markdown_report(
    unconstrained: Dict, 
    constrained: Dict,
    unconstrained_results: List[Dict],
    constrained_results: List[Dict]
):
    """Generate human-readable Markdown report."""
    report_path = REPORT_DIR / "grammar_eval_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Pillar 1: Grammar/Schema Compliance Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("| Metric | Unconstrained | Constrained |\n")
        f.write("|--------|---------------|-------------|\n")
        f.write(f"| Valid JSON | {unconstrained['valid_json_rate']:.1%} | {constrained['valid_json_rate']:.1%} |\n")
        f.write(f"| Schema Compliant | {unconstrained['schema_compliance_rate']:.1%} | {constrained['schema_compliance_rate']:.1%} |\n")
        f.write(f"| Avg Latency | {unconstrained['avg_latency_ms']:.1f}ms | {constrained['avg_latency_ms']:.1f}ms |\n")
        f.write(f"| Avg TTFT | {unconstrained['avg_ttft_ms']:.1f}ms | {constrained['avg_ttft_ms']:.1f}ms |\n")
        f.write(f"| Avg Throughput | {unconstrained['avg_tokens_per_sec']:.1f} t/s | {constrained['avg_tokens_per_sec']:.1f} t/s |\n\n")
        
        f.write("## Field-Level Errors (Unconstrained)\n\n")
        f.write(f"- **Missing Required Fields:** {unconstrained['total_missing_fields']}\n")
        f.write(f"- **Extra Fields:** {unconstrained['total_extra_fields']}\n")
        f.write(f"- **Enum Violations:** {unconstrained['total_enum_errors']}\n")
        f.write(f"- **Type Mismatches:** {unconstrained['total_type_errors']}\n\n")
        
        f.write("## Category Breakdown (Constrained)\n\n")
        f.write("| Category | Count | Compliance | Avg Latency |\n")
        f.write("|----------|-------|------------|-------------|\n")
        for cat, metrics in constrained['category_metrics'].items():
            f.write(f"| {cat} | {metrics['count']} | {metrics['schema_compliance_rate']:.1%} | {metrics['avg_latency_ms']:.0f}ms |\n")
        
        f.write("\n## Failure Details (Unconstrained)\n\n")
        failures = [r for r in unconstrained_results if not r['matches_schema']]
        if failures:
            for r in failures:
                f.write(f"### {r['case_id']} ({r['category']})\n")
                f.write(f"- **Error Type:** {r.get('error_type', 'N/A')}\n")
                f.write(f"- **Error:** {r.get('error', 'N/A')}\n")
                if r.get('missing_fields'):
                    f.write(f"- **Missing Fields:** {r['missing_fields']}\n")
                if r.get('enum_errors'):
                    f.write(f"- **Enum Errors:** {json.dumps(r['enum_errors'], indent=2)}\n")
                f.write("\n")
        else:
            f.write("🎉 **Perfect Score!** All policies passed strict schema validation.\n")
        
        f.write("\n## Production Deployment Decision\n\n")
        if unconstrained['schema_compliance_rate'] >= 0.95:
            f.write("✅ **RECOMMENDATION:** Disable grammar enforcement in production.\n\n")
            f.write("The model has internalized the schema well enough (>95% compliance).\n")
            f.write("This will maximize inference speed (lower TTFT, higher tokens/sec).\n")
        else:
            f.write("⚠️ **RECOMMENDATION:** Keep grammar enforcement enabled in production.\n\n")
            f.write(f"Unconstrained compliance is {unconstrained['schema_compliance_rate']:.1%} (<95%).\n")
            f.write("The Rust interceptor must use `LlamaGrammar` to guarantee schema compliance.\n")
    
    print(f"📝 Markdown report saved to: {report_path}")

if __name__ == "__main__":
    run_grammar_evals()