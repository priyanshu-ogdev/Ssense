#!/usr/bin/env python3
"""
run_grammar_evals.py – JSON Schema Compliance Evaluation (Production Grade)

Tests whether the trained SLM outputs valid JSON that strictly adheres
to the dpdp_schema.json contract.

Handles:
- Markdown-wrapped JSON extraction
- Trailing comma tolerance
- Empty output detection
- Timing and token metrics
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any
from jsonschema import validate, ValidationError
from llama_cpp import Llama
from tqdm import tqdm

# Load the schema
SCHEMA_PATH = Path("libs/contracts/schemas/dpdp_schema.json")
with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
    DPDP_SCHEMA = json.load(f)

# Load test policies
TEST_POLICIES_DIR = Path("ml/evals/holdout_policies")

def load_test_policies() -> List[Dict[str, str]]:
    """Load all test policies from the holdout directory."""
    policies = []
    for policy_file in sorted(TEST_POLICIES_DIR.glob("*.txt")):
        with open(policy_file, 'r', encoding='utf-8') as f:
            content = f.read()
            policies.append({
                "filename": policy_file.name,
                "content": content
            })
    return policies

def extract_json_from_output(output: str) -> str:
    """Extract JSON from model output, handling markdown wrapping and edge cases."""
    # Remove markdown code blocks if present
    if '```json' in output:
        match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if match:
            output = match.group(1)
    elif '```' in output:
        match = re.search(r'```\s*(.*?)\s*```', output, re.DOTALL)
        if match:
            output = match.group(1)
    
    # Remove trailing commas (common LLM mistake)
    output = re.sub(r',(\s*[}\]])', r'\1', output)
    
    return output.strip()

# Load the actual law text
LAW_FILE_PATH = Path("ml/data-forge/dpdp_act_and_rules_2025.txt")

def load_law_context():
    """Load the DPDP Act and Rules 2025 text."""
    if not LAW_FILE_PATH.exists():
        raise FileNotFoundError(f"Law file not found: {LAW_FILE_PATH}")
    
    with open(LAW_FILE_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def run_inference(llm: Llama, policy_text: str) -> tuple[str, float, int]:
    """Run inference on a single policy. Returns (output, latency_ms, tokens_generated)."""
    SYSTEM_PROMPT = "You are a strict DPDP Regulatory Auditor enforcing the Indian Digital Personal Data Protection (DPDP) Act 2023 and Rules 2025. Output ONLY valid JSON matching the dpdp_schema."
    
    law_context = load_law_context()
    
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
[CONTEXT: THE LAW]
{law_context}

[SYNTHESIZED POLICY]
{policy_text}<|im_end|>
<|im_start|>assistant
"""
    
    start_time = time.time()
    output = llm(
        prompt,
        max_tokens=1024,
        temperature=0.0,
        stop=["<|im_end|>"]
    )
    latency_ms = (time.time() - start_time) * 1000
    
    raw_output = output['choices'][0]['text'].strip()
    tokens_generated = output['usage'].get('completion_tokens', 0) if 'usage' in output else 0
    
    return raw_output, latency_ms, tokens_generated

def validate_json_structure(output: str) -> Dict[str, Any]:
    """Validate that the output is valid JSON and matches the schema."""
    result = {
        "is_valid_json": False,
        "matches_schema": False,
        "error": None,
        "error_type": None,
        "parsed_json": None,
        "raw_output_length": len(output)
    }
    
    # Handle empty output
    if not output or len(output.strip()) == 0:
        result["error"] = "Empty output"
        result["error_type"] = "EMPTY_OUTPUT"
        return result
    
    # Extract JSON from potential markdown wrapping
    extracted_json = extract_json_from_output(output)
    
    # Step 1: Check if it's valid JSON
    try:
        parsed = json.loads(extracted_json)
        result["is_valid_json"] = True
        result["parsed_json"] = parsed
    except json.JSONDecodeError as e:
        result["error"] = f"JSON parse error: {str(e)}"
        result["error_type"] = "PARSE_ERROR"
        return result
    
    # Step 2: Validate against schema
    try:
        validate(instance=parsed, schema=DPDP_SCHEMA)
        result["matches_schema"] = True
    except ValidationError as e:
        result["error"] = f"Schema validation error: {e.message}"
        result["error_type"] = "SCHEMA_ERROR"
        result["schema_path"] = list(e.absolute_path) if e.absolute_path else []
    
    return result

def run_grammar_evals():
    """Run the complete grammar evaluation suite."""
    print("🔍 Starting Grammar/Schema Compliance Evaluation...")
    print(f"Schema: {SCHEMA_PATH}")
    print(f"Test Policies: {TEST_POLICIES_DIR}")
    print()
    
    # Load model
    print("Loading Q4_K_M model...")
    llm = Llama(
        model_path="apps/browser-core/src-tauri/models/ssense-dpdp-9b-local-q4_k_m.gguf",
        n_ctx=8192,
        n_gpu_layers=0,
        verbose=False
    )
    print("✅ Model loaded\n")
    
    # Load test data
    test_policies = load_test_policies()
    print(f"Found {len(test_policies)} test policies\n")
    
    # Run evaluations
    results = []
    valid_json_count = 0
    schema_compliant_count = 0
    latencies = []
    token_counts = []
    error_types = {}
    
    for policy in tqdm(test_policies, desc="Evaluating"):
        output, latency_ms, tokens = run_inference(llm, policy['content'])
        latencies.append(latency_ms)
        token_counts.append(tokens)
        
        validation = validate_json_structure(output)
        
        result = {
            "filename": policy['filename'],
            "output_length": len(output),
            "latency_ms": latency_ms,
            "tokens_generated": tokens,
            **validation
        }
        results.append(result)
        
        if validation["is_valid_json"]:
            valid_json_count += 1
        if validation["matches_schema"]:
            schema_compliant_count += 1
        
        # Track error types
        if validation["error_type"]:
            error_types[validation["error_type"]] = error_types.get(validation["error_type"], 0) + 1
    
    # Calculate metrics
    total = len(test_policies)
    json_validity_rate = (valid_json_count / total) * 100
    schema_compliance_rate = (schema_compliant_count / total) * 100
    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(token_counts) / len(token_counts)
    
    # Print results
    print("\n" + "="*70)
    print("GRAMMAR EVALUATION RESULTS")
    print("="*70)
    print(f"Total Test Policies: {total}")
    print(f"\nJSON Validity:")
    print(f"  Valid JSON: {valid_json_count}/{total} ({json_validity_rate:.1f}%)")
    print(f"  Schema Compliant: {schema_compliant_count}/{total} ({schema_compliance_rate:.1f}%)")
    print(f"\nPerformance:")
    print(f"  Avg Latency: {avg_latency:.1f}ms")
    print(f"  Avg Tokens: {avg_tokens:.0f}")
    
    if error_types:
        print(f"\nError Breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count} ({count/total*100:.1f}%)")
    
    print("="*70)
    
    # Save detailed results
    output_path = Path("ml/evals/reports/grammar_eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_policies": total,
                "valid_json_count": valid_json_count,
                "schema_compliant_count": schema_compliant_count,
                "json_validity_rate": json_validity_rate,
                "schema_compliance_rate": schema_compliance_rate,
                "avg_latency_ms": avg_latency,
                "avg_tokens_generated": avg_tokens
            },
            "error_breakdown": error_types,
            "detailed_results": results
        }, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {output_path}")
    
    # Print failures
    failures = [r for r in results if not r["matches_schema"]]
    if failures:
        print(f"\n⚠️  Found {len(failures)} schema violations:")
        for failure in failures[:10]:  # Show first 10
            error_type = failure.get('error_type', 'UNKNOWN')
            print(f"  - {failure['filename']}: [{error_type}] {failure['error'][:80]}")

if __name__ == "__main__":
    run_grammar_evals()