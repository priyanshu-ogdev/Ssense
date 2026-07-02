#!/usr/bin/env python3
"""
run_accuracy_evals.py – Legal Reasoning Accuracy Evaluation (Production Grade v2)

Tests whether the trained SLM correctly identifies DPDP violations,
maps them to the correct statutory sections, and provides accurate reasoning.

Measures the 5 Pillars of Modern SLM Evaluation:
1. Schema Compliance Rate (Structural Integrity)
2. Violation F1 Score (Detection Accuracy)
3. Trust Score MAE (Scoring Accuracy)
4. Evidence Hallucination Rate (Legal Fidelity)
5. Inference Efficiency (Hardware Performance)

Supports:
- Statute alias matching (Section 8(7) == Section 8 == Rule 8(3))
- Severity-weighted F1 scoring (mapped from violation_type)
- Category-aware metrics (blatant, subtle, distractor, multi, edge_case)
- Network action validation
- Evidence hallucination detection
- TTFT and tokens/second measurement
"""

import os
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
from datetime import datetime
from llama_cpp import Llama, LlamaGrammar
from tqdm import tqdm
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
SCHEMA_PATH = Path("libs/contracts/schemas/dpdp_schema.json")
GROUND_TRUTH_PATH = Path("ml/evals/holdout_policies/ground_truth.json")
LAW_FILE_PATH = Path("ml/data-forge/dpdp_act_and_rules_2025.txt")
MODEL_PATH = Path("apps/browser-core/src-tauri/models/ssense-dpdp-9b-local-q4_k_m.gguf")

# Map violation_type to internal severity for weighted F1
VIOLATION_SEVERITY_MAP = {
    "CHILD_CONSENT_VIOLATION": "CRITICAL",
    "CROSS_BORDER_TRANSFER_VIOLATION": "HIGH",
    "CONSENT_NOT_FREE_OR_SPECIFIC": "HIGH",
    "DATA_RETENTION_LIMIT_EXCEEDED": "HIGH",
    "PURPOSE_LIMITATION_VIOLATION": "HIGH",
    "SECURITY_SAFEGUARDS_MISSING": "HIGH",
    "NOTICE_INADEQUATE": "MEDIUM",
    "GRIEVANCE_REDRESSAL_INADEQUATE": "MEDIUM",
    "SDF_OBLIGATIONS_MISSING": "MEDIUM",
    "BREACH_NOTIFICATION_FAILURE": "HIGH",
}

SEVERITY_WEIGHTS = {
    "CRITICAL": 1.0,
    "HIGH": 0.8,
    "MEDIUM": 0.6,
    "LOW": 0.4
}

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_schema() -> Dict[str, Any]:
    """Load the sealed DPDP JSON schema."""
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_law_context() -> str:
    """Load the DPDP Act and Rules 2025 text."""
    if not LAW_FILE_PATH.exists():
        raise FileNotFoundError(f"Law file not found: {LAW_FILE_PATH}")
    with open(LAW_FILE_PATH, 'r', encoding='utf-8') as f:
        return f.read()

def load_test_data() -> List[Dict[str, Any]]:
    """Load test policies with ground truth annotations."""
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    test_data = []
    for item in ground_truth:
        # Support both inline snippets and file-based policies
        if 'policy_text_snippet' in item:
            content = item['policy_text_snippet']
        else:
            policy_file = Path("ml/evals/holdout_policies") / item['filename']
            if not policy_file.exists():
                print(f"⚠️  Warning: Policy file not found: {policy_file}")
                continue
            with open(policy_file, 'r', encoding='utf-8') as f:
                content = f.read()
        
        test_data.append({
            "case_id": item.get('case_id', item['filename']),
            "filename": item['filename'],
            "category": item.get('category', 'unknown'),
            "description": item.get('description', ''),
            "content": content,
            "expected_output": item.get('expected_output', {}),
            "evaluation_targets": item.get('evaluation_targets', {})
        })
    
    return test_data

# ═══════════════════════════════════════════════════════════════════════════
# SECTION NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def normalize_section_reference(section: str) -> Set[str]:
    """Normalize a section reference to a set of equivalent references."""
    section = section.strip()
    variations = {section}
    
    # Handle "Section X read with Rule Y" compound references
    if "read with" in section.lower():
        parts = re.split(r'\s+read\s+with\s+', section, flags=re.IGNORECASE)
        for part in parts:
            variations.update(normalize_section_reference(part.strip()))
    
    # Handle "Section X(Y)" -> add "Section X"
    if '(' in section:
        base = section.split('(')[0].strip()
        variations.add(base)
    
    # Handle "Section X" -> add "X"
    if section.startswith("Section "):
        variations.add(section.replace("Section ", ""))
    
    # Handle "Rule X(Y)" -> add "Rule X"
    if section.startswith("Rule ") and '(' in section:
        base = section.split('(')[0].strip()
        variations.add(base)
    
    # Handle "Rule X" -> add "X"
    if section.startswith("Rule "):
        variations.add(section.replace("Rule ", ""))
    
    # Case normalization
    variations.add(section.lower())
    
    return variations

def sections_match(pred: str, gt: str) -> bool:
    """Check if two section references are equivalent."""
    pred_vars = normalize_section_reference(pred)
    gt_vars = normalize_section_reference(gt)
    return bool(pred_vars & gt_vars)

# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_inference(llm: Llama, policy_text: str, law_context: str, grammar: LlamaGrammar) -> Dict[str, Any]:
    """
    Run inference on a single policy with schema enforcement.
    Returns dict with output, latency_ms, ttft_ms, tokens_generated, tokens_per_sec.
    """
    SYSTEM_PROMPT = "You are a strict DPDP Regulatory Auditor enforcing the Indian Digital Personal Data Protection (DPDP) Act 2023 and Rules 2025. Output ONLY valid JSON matching the dpdp_schema."
    
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
[CONTEXT: THE LAW]
{law_context}

[SYNTHESIZED POLICY]
{policy_text}<|im_end|>
<|im_start|>assistant
"""
    
    # Measure TTFT via streaming
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    # Use streaming to measure TTFT
    result = llm(
        prompt,
        max_tokens=2048,
        temperature=0.0,
        stop=["<|im_end|>"],
        grammar=grammar,
        stream=True
    )
    
    output_text = ""
    for chunk in result:
        if first_token_time is None and chunk['choices'][0]['text']:
            first_token_time = time.time()
        output_text += chunk['choices'][0]['text']
        token_count += 1
    
    end_time = time.time()
    
    total_latency_ms = (end_time - start_time) * 1000
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_latency_ms
    generation_time_ms = total_latency_ms - ttft_ms
    tokens_per_sec = (token_count / (generation_time_ms / 1000)) if generation_time_ms > 0 else 0
    
    return {
        "raw_output": output_text.strip(),
        "latency_ms": total_latency_ms,
        "ttft_ms": ttft_ms,
        "tokens_generated": token_count,
        "tokens_per_sec": tokens_per_sec
    }

# ═══════════════════════════════════════════════════════════════════════════
# PILLAR 1: SCHEMA COMPLIANCE
# ═══════════════════════════════════════════════════════════════════════════
def validate_schema_compliance(parsed_output: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that output matches the sealed schema structure."""
    required_fields = schema.get('required', [])
    properties = schema.get('properties', {})
    
    missing_fields = [f for f in required_fields if f not in parsed_output]
    extra_fields = [f for f in parsed_output.keys() if f not in properties]
    
    # Check violation structure
    violation_errors = []
    if 'violations' in parsed_output and isinstance(parsed_output['violations'], list):
        violation_schema = properties.get('violations', {}).get('items', {})
        required_violation_fields = violation_schema.get('required', [])
        violation_properties = violation_schema.get('properties', {})
        
        for i, v in enumerate(parsed_output['violations']):
            missing = [f for f in required_violation_fields if f not in v]
            extra = [f for f in v.keys() if f not in violation_properties]
            if missing or extra:
                violation_errors.append({
                    "index": i,
                    "missing": missing,
                    "extra": extra
                })
    
    is_compliant = len(missing_fields) == 0 and len(extra_fields) == 0 and len(violation_errors) == 0
    
    return {
        "is_compliant": is_compliant,
        "missing_fields": missing_fields,
        "extra_fields": extra_fields,
        "violation_errors": violation_errors
    }

# ═══════════════════════════════════════════════════════════════════════════
# PILLAR 2: VIOLATION F1 SCORE
# ═══════════════════════════════════════════════════════════════════════════
def calculate_violation_f1(
    predicted: List[Dict[str, Any]], 
    expected_types: List[str],
    expected_aliases: List[List[str]]
) -> Dict[str, float]:
    """Calculate precision, recall, F1 with alias support."""
    
    # Build ground truth set
    gt_set = set()
    for vtype, aliases in zip(expected_types, expected_aliases):
        for alias in aliases:
            for variation in normalize_section_reference(alias):
                gt_set.add((variation.lower(), vtype))
    
    # Build predicted set
    pred_set = set()
    for v in predicted:
        section = v.get('statute_reference', '')
        vtype = v.get('violation_type', '')
        for variation in normalize_section_reference(section):
            pred_set.add((variation.lower(), vtype))
    
    # Calculate metrics
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }

def calculate_severity_weighted_f1(
    predicted: List[Dict[str, Any]], 
    expected_types: List[str],
    expected_aliases: List[List[str]]
) -> float:
    """Calculate F1 weighted by violation severity."""
    
    # Build ground truth with severity
    gt_dict = {}
    for vtype, aliases in zip(expected_types, expected_aliases):
        severity = VIOLATION_SEVERITY_MAP.get(vtype, "MEDIUM")
        weight = SEVERITY_WEIGHTS[severity]
        for alias in aliases:
            for variation in normalize_section_reference(alias):
                gt_dict[(variation.lower(), vtype)] = weight
    
    # Build predicted set
    pred_set = set()
    for v in predicted:
        section = v.get('statute_reference', '')
        vtype = v.get('violation_type', '')
        for variation in normalize_section_reference(section):
            pred_set.add((variation.lower(), vtype))
    
    # Calculate weighted metrics
    weighted_tp = sum(gt_dict[k] for k in pred_set if k in gt_dict)
    weighted_fp = sum(0.6 for k in pred_set if k not in gt_dict)
    weighted_fn = sum(gt_dict[k] for k in gt_dict if k not in pred_set)
    
    wp = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0.0
    wr = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0.0
    wf1 = 2 * (wp * wr) / (wp + wr) if (wp + wr) > 0 else 0.0
    
    return wf1

# ═══════════════════════════════════════════════════════════════════════════
# PILLAR 3: TRUST SCORE ACCURACY
# ═══════════════════════════════════════════════════════════════════════════
def calculate_trust_score_accuracy(
    predicted_score: int, 
    expected_range: List[int]
) -> float:
    """Calculate accuracy based on whether prediction falls within expected range."""
    min_score, max_score = expected_range
    
    if min_score <= predicted_score <= max_score:
        return 1.0
    
    # Calculate distance from nearest bound
    if predicted_score < min_score:
        distance = min_score - predicted_score
    else:
        distance = predicted_score - max_score
    
    # Linear decay: 0 accuracy at 30 points outside range
    return max(0.0, 1.0 - (distance / 30.0))

# ═══════════════════════════════════════════════════════════════════════════
# PILLAR 4: EVIDENCE HALLUCINATION CHECK
# ═══════════════════════════════════════════════════════════════════════════
def check_evidence_hallucination(
    predicted_violations: List[Dict[str, Any]],
    policy_text: str,
    check_required: bool
) -> Dict[str, Any]:
    """Verify that evidence_quote actually exists in the source policy text."""
    if not check_required:
        return {"hallucination_rate": 0.0, "checked": 0, "hallucinated": 0}
    
    checked = 0
    hallucinated = 0
    
    for v in predicted_violations:
        quote = v.get('evidence_quote', '')
        if not quote:
            continue
        
        checked += 1
        # Normalize both strings for comparison
        normalized_quote = ' '.join(quote.lower().split())
        normalized_policy = ' '.join(policy_text.lower().split())
        
        # Check if quote exists in policy (allowing for minor whitespace differences)
        if normalized_quote not in normalized_policy:
            # Try fuzzy matching (allow up to 20% character difference)
            if not fuzzy_match(quote, policy_text):
                hallucinated += 1
    
    hallucination_rate = hallucinated / checked if checked > 0 else 0.0
    
    return {
        "hallucination_rate": hallucination_rate,
        "checked": checked,
        "hallucinated": hallucinated
    }

def fuzzy_match(quote: str, text: str, threshold: float = 0.8) -> bool:
    """Simple fuzzy matching using character overlap."""
    quote_words = quote.lower().split()
    text_words = text.lower().split()
    
    if len(quote_words) == 0:
        return True
    
    # Check if quote words appear in sequence in text
    for i in range(len(text_words) - len(quote_words) + 1):
        match_count = sum(1 for j, qw in enumerate(quote_words) if qw in text_words[i+j])
        if match_count / len(quote_words) >= threshold:
            return True
    
    return False

# ═══════════════════════════════════════════════════════════════════════════
# PILLAR 5: NETWORK ACTION VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
def validate_network_actions(
    predicted_violations: List[Dict[str, Any]],
    expected_actions: List[str]
) -> Dict[str, Any]:
    """Validate that predicted network actions match expected actions."""
    pred_actions = set(v.get('network_action', '') for v in predicted_violations if v.get('network_action'))
    expected_set = set(expected_actions)
    
    correct = len(pred_actions & expected_set)
    total = max(len(pred_actions), len(expected_set))
    
    return {
        "accuracy": correct / total if total > 0 else 1.0,
        "predicted": list(pred_actions),
        "expected": list(expected_set),
        "missing": list(expected_set - pred_actions),
        "extra": list(pred_actions - expected_set)
    }

# ═══════════════════════════════════════════════════════════════════════════
# SUBTLETY SCORE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
def validate_subtlety_score(
    predicted_score: int,
    expected_score: int,
    category: str
) -> float:
    """Validate subtlety score accuracy."""
    # Only validate for subtle cases
    if "subtle" not in category:
        return 1.0
    
    error = abs(predicted_score - expected_score)
    return max(0.0, 1.0 - (error / 30.0))

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════
def run_accuracy_evals():
    """Run the complete 5-pillar accuracy evaluation suite."""
    print("🎯 Starting Legal Reasoning Accuracy Evaluation (5-Pillar Framework)...")
    print(f"Schema: {SCHEMA_PATH}")
    print(f"Ground Truth: {GROUND_TRUTH_PATH}")
    print(f"Model: {MODEL_PATH}")
    print()
    
    # Load components
    schema = load_schema()
    law_context = load_law_context()
    test_data = load_test_data()
    
    print(f"Found {len(test_data)} annotated test policies")
    print(f"Categories: {set(item['category'] for item in test_data)}")
    print()
    
    # Load model with grammar enforcement
    print("Loading Q4_K_M model with schema grammar...")
    grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=8192,
        n_gpu_layers=0,
        verbose=False
    )
    print("✅ Model loaded with grammar enforcement\n")
    
    # Evaluation accumulators
    all_results = []
    schema_compliance_rates = []
    f1_scores = []
    weighted_f1_scores = []
    trust_score_accuracies = []
    hallucination_rates = []
    network_action_accuracies = []
    subtlety_accuracies = []
    latencies = []
    ttfts = []
    tokens_per_secs = []
    
    # Run evaluations
    for item in tqdm(test_data, desc="Evaluating"):
        # Run inference
        inference_result = run_inference(llm, item['content'], law_context, grammar)
        
        latencies.append(inference_result['latency_ms'])
        ttfts.append(inference_result['ttft_ms'])
        tokens_per_secs.append(inference_result['tokens_per_sec'])
        
        # Parse output
        try:
            parsed_output = json.loads(inference_result['raw_output'])
        except json.JSONDecodeError as e:
            print(f"\n⚠️  {item['case_id']}: Invalid JSON output: {e}")
            all_results.append({
                "case_id": item['case_id'],
                "category": item['category'],
                "schema_compliant": False,
                "error": "JSON_PARSE_ERROR"
            })
            continue
        
        # Pillar 1: Schema Compliance
        schema_check = validate_schema_compliance(parsed_output, schema)
        schema_compliance_rates.append(1.0 if schema_check['is_compliant'] else 0.0)
        
        # Extract data
        predicted_violations = parsed_output.get('violations', [])
        expected_output = item['expected_output']
        eval_targets = item['evaluation_targets']
        
        # Pillar 2: Violation F1
        expected_types = eval_targets.get('expected_violation_types', [])
        expected_aliases = eval_targets.get('expected_statute_aliases', [])
        
        f1_metrics = calculate_violation_f1(predicted_violations, expected_types, expected_aliases)
        f1_scores.append(f1_metrics['f1'])
        
        weighted_f1 = calculate_severity_weighted_f1(predicted_violations, expected_types, expected_aliases)
        weighted_f1_scores.append(weighted_f1)
        
        # Pillar 3: Trust Score Accuracy
        predicted_trust_score = parsed_output.get('dpdp_trust_score', 50)
        expected_range = eval_targets.get('expected_trust_score_range', [50, 50])
        trust_acc = calculate_trust_score_accuracy(predicted_trust_score, expected_range)
        trust_score_accuracies.append(trust_acc)
        
        # Pillar 4: Evidence Hallucination
        hallucination_check = eval_targets.get('hallucination_check_required', False)
        hallucination_result = check_evidence_hallucination(
            predicted_violations, item['content'], hallucination_check
        )
        hallucination_rates.append(hallucination_result['hallucination_rate'])
        
        # Pillar 5: Network Action Validation
        expected_actions = eval_targets.get('expected_network_actions', [])
        network_action_result = validate_network_actions(predicted_violations, expected_actions)
        network_action_accuracies.append(network_action_result['accuracy'])
        
        # Subtlety Score Validation
        predicted_subtlety = parsed_output.get('subtlety_score', 0)
        expected_subtlety = expected_output.get('subtlety_score', 0)
        subtlety_acc = validate_subtlety_score(predicted_subtlety, expected_subtlety, item['category'])
        subtlety_accuracies.append(subtlety_acc)
        
        # Store result
        all_results.append({
            "case_id": item['case_id'],
            "category": item['category'],
            "schema_compliant": schema_check['is_compliant'],
            "f1": f1_metrics['f1'],
            "weighted_f1": weighted_f1,
            "trust_score_accuracy": trust_acc,
            "hallucination_rate": hallucination_result['hallucination_rate'],
            "network_action_accuracy": network_action_result['accuracy'],
            "subtlety_accuracy": subtlety_acc,
            "latency_ms": inference_result['latency_ms'],
            "ttft_ms": inference_result['ttft_ms'],
            "tokens_per_sec": inference_result['tokens_per_sec'],
            "predicted_violations": len(predicted_violations),
            "expected_violations": len(expected_types)
        })
    
    # ═══════════════════════════════════════════════════════════════════════
    # AGGREGATE METRICS
    # ═══════════════════════════════════════════════════════════════════════
    summary = {
        "total_policies": len(test_data),
        "pillar_1_schema_compliance": float(np.mean(schema_compliance_rates)),
        "pillar_2_avg_f1": float(np.mean(f1_scores)),
        "pillar_2_weighted_f1": float(np.mean(weighted_f1_scores)),
        "pillar_3_trust_score_accuracy": float(np.mean(trust_score_accuracies)),
        "pillar_4_hallucination_rate": float(np.mean(hallucination_rates)),
        "pillar_5_network_action_accuracy": float(np.mean(network_action_accuracies)),
        "subtlety_accuracy": float(np.mean(subtlety_accuracies)),
        "avg_latency_ms": float(np.mean(latencies)),
        "avg_ttft_ms": float(np.mean(ttfts)),
        "avg_tokens_per_sec": float(np.mean(tokens_per_secs))
    }
    
    # Category breakdown
    category_metrics = {}
    for category in set(item['category'] for item in test_data):
        cat_results = [r for r in all_results if r['category'] == category]
        category_metrics[category] = {
            "count": len(cat_results),
            "avg_f1": float(np.mean([r.get('f1', 0) for r in cat_results])),
            "schema_compliance": float(np.mean([1.0 if r.get('schema_compliant') else 0.0 for r in cat_results])),
            "hallucination_rate": float(np.mean([r.get('hallucination_rate', 0) for r in cat_results]))
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # OUTPUT RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("5-PILLAR EVALUATION RESULTS")
    print("="*70)
    print(f"\n📋 Pillar 1 - Schema Compliance: {summary['pillar_1_schema_compliance']:.1%}")
    print(f"🎯 Pillar 2 - Violation F1: {summary['pillar_2_avg_f1']:.3f} (Weighted: {summary['pillar_2_weighted_f1']:.3f})")
    print(f"📊 Pillar 3 - Trust Score Accuracy: {summary['pillar_3_trust_score_accuracy']:.1%}")
    print(f"🔍 Pillar 4 - Hallucination Rate: {summary['pillar_4_hallucination_rate']:.1%}")
    print(f"⚡ Pillar 5 - Network Action Accuracy: {summary['pillar_5_network_action_accuracy']:.1%}")
    print(f"🎭 Subtlety Score Accuracy: {summary['subtlety_accuracy']:.1%}")
    print(f"\n⚡ Performance:")
    print(f"  Avg Latency: {summary['avg_latency_ms']:.1f}ms")
    print(f"  Avg TTFT: {summary['avg_ttft_ms']:.1f}ms")
    print(f"  Avg Throughput: {summary['avg_tokens_per_sec']:.1f} tokens/sec")
    
    print(f"\n📂 Category Breakdown:")
    for cat, metrics in category_metrics.items():
        print(f"  {cat}: F1={metrics['avg_f1']:.3f}, Schema={metrics['schema_compliance']:.1%}, Halluc={metrics['hallucination_rate']:.1%}")
    
    print("="*70)
    
    # Save JSON results
    output_path = Path("ml/evals/reports/accuracy_eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "category_metrics": category_metrics,
            "detailed_results": all_results
        }, f, indent=2)
    
    print(f"\n📊 JSON results saved to: {output_path}")
    
    # Generate Markdown report
    generate_markdown_report(summary, category_metrics, all_results)
    
    # Print worst performers
    sorted_results = sorted([r for r in all_results if 'f1' in r], key=lambda x: x['f1'])
    if sorted_results:
        print("\n⚠️  Worst performing cases (lowest F1):")
        for r in sorted_results[:5]:
            print(f"  - {r['case_id']} ({r['category']}): F1={r['f1']:.3f}, Schema={r['schema_compliant']}")

def generate_markdown_report(summary: Dict, category_metrics: Dict, all_results: List[Dict]):
    """Generate a human-readable Markdown report."""
    report_path = Path("ml/evals/reports/accuracy_eval_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# DPDP SLM Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Policies Evaluated:** {summary['total_policies']}\n")
        f.write(f"- **Schema Compliance Rate:** {summary['pillar_1_schema_compliance']:.1%}\n")
        f.write(f"- **Violation F1 Score:** {summary['pillar_2_avg_f1']:.3f}\n")
        f.write(f"- **Weighted F1 Score:** {summary['pillar_2_weighted_f1']:.3f}\n")
        f.write(f"- **Trust Score Accuracy:** {summary['pillar_3_trust_score_accuracy']:.1%}\n")
        f.write(f"- **Evidence Hallucination Rate:** {summary['pillar_4_hallucination_rate']:.1%}\n")
        f.write(f"- **Network Action Accuracy:** {summary['pillar_5_network_action_accuracy']:.1%}\n")
        f.write(f"- **Subtlety Score Accuracy:** {summary['subtlety_accuracy']:.1%}\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write(f"- **Average Latency:** {summary['avg_latency_ms']:.1f}ms\n")
        f.write(f"- **Average TTFT:** {summary['avg_ttft_ms']:.1f}ms\n")
        f.write(f"- **Average Throughput:** {summary['avg_tokens_per_sec']:.1f} tokens/sec\n\n")
        
        f.write("## Category Breakdown\n\n")
        f.write("| Category | Count | Avg F1 | Schema Compliance | Hallucination Rate |\n")
        f.write("|----------|-------|--------|-------------------|--------------------|\n")
        for cat, metrics in category_metrics.items():
            f.write(f"| {cat} | {metrics['count']} | {metrics['avg_f1']:.3f} | {metrics['schema_compliance']:.1%} | {metrics['hallucination_rate']:.1%} |\n")
        
        f.write("\n## Detailed Results\n\n")
        for r in all_results:
            f.write(f"### {r['case_id']} ({r['category']})\n")
            f.write(f"- Schema Compliant: {'✅' if r.get('schema_compliant') else '❌'}\n")
            if 'f1' in r:
                f.write(f"- F1 Score: {r['f1']:.3f}\n")
                f.write(f"- Trust Score Accuracy: {r['trust_score_accuracy']:.1%}\n")
                f.write(f"- Hallucination Rate: {r['hallucination_rate']:.1%}\n")
                f.write(f"- Network Action Accuracy: {r['network_action_accuracy']:.1%}\n")
            f.write("\n")
    
    print(f"📝 Markdown report saved to: {report_path}")

if __name__ == "__main__":
    run_accuracy_evals()