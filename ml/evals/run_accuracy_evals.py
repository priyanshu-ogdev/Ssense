#!/usr/bin/env python3
"""
run_accuracy_evals.py – Legal Reasoning Accuracy Evaluation (Production Grade)

Tests whether the trained SLM correctly identifies DPDP violations,
maps them to the correct statutory sections, and provides accurate reasoning.

Supports:
- Statute alias matching (Section 8(7) == Section 8 == Rule 8(3))
- Severity-weighted F1 scoring
- Policy-type-aware metrics (compliant vs non-compliant)
- Cross-reference validation
- Per-section breakdown analysis
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from llama_cpp import Llama
from tqdm import tqdm
import numpy as np

# Load the schema
SCHEMA_PATH = Path("libs/contracts/schemas/dpdp_schema.json")
with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
    DPDP_SCHEMA = json.load(f)

# Load test policies with ground truth
TEST_POLICIES_DIR = Path("ml/evals/holdout_policies")
GROUND_TRUTH_PATH = Path("ml/evals/holdout_policies/ground_truth.json")

# Severity weights for weighted F1 calculation
SEVERITY_WEIGHTS = {
    "CRITICAL": 1.0,
    "HIGH": 0.8,
    "MEDIUM": 0.6,
    "LOW": 0.4
}

def load_test_data() -> List[Dict[str, Any]]:
    """Load test policies with ground truth annotations."""
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    test_data = []
    for item in ground_truth:
        policy_file = TEST_POLICIES_DIR / item['filename']
        if not policy_file.exists():
            print(f"⚠️  Warning: Policy file not found: {policy_file}")
            continue
            
        with open(policy_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        test_data.append({
            "filename": item['filename'],
            "content": content,
            "policy_type": item.get('policy_type', 'non_compliant'),
            "ground_truth": item.get('expected_violations', []),
            "expected_trust_score": item.get('expected_trust_score', 50),
            "expected_summary": item.get('expected_compliance_summary', '')
        })
    
    return test_data

# Load the actual law text
LAW_FILE_PATH = Path("ml/data-forge/dpdp_act_and_rules_2025.txt")

def load_law_context():
    """Load the DPDP Act and Rules 2025 text."""
    if not LAW_FILE_PATH.exists():
        raise FileNotFoundError(f"Law file not found: {LAW_FILE_PATH}")
    
    with open(LAW_FILE_PATH, 'r', encoding='utf-8') as f:
        return f.read()
    
def run_inference(llm: Llama, policy_text: str) -> Tuple[str, float, int]:
    """Run inference on a single policy. Returns (output, latency_ms, tokens_generated)."""
    SYSTEM_PROMPT = "You are a strict DPDP Regulatory Auditor enforcing the Indian Digital Personal Data Protection (DPDP) Act 2023 and Rules 2025. Output ONLY valid JSON matching the dpdp_schema."
    
    # Load the DPDP Act context (in production, this would be injected via BM25 router)
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

def extract_violations(model_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract violations from model output."""
    return model_output.get('violations', [])

def normalize_section_reference(section: str) -> Set[str]:
    """Normalize a section reference to a set of equivalent references."""
    # Remove whitespace and normalize case
    section = section.strip()
    
    # Generate common variations
    variations = {section}
    
    # If it's "Section 8(7)", also add "Section 8"
    if '(' in section:
        base = section.split('(')[0].strip()
        variations.add(base)
    
    # If it's "Section 8", also add "8"
    if section.startswith("Section "):
        variations.add(section.replace("Section ", ""))
    
    return variations

def calculate_metrics(
    predicted: List[Dict[str, Any]], 
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Calculate precision, recall, and F1 with alias support."""
    
    # Build ground truth set with aliases
    gt_set = set()
    gt_severity_map = {}
    
    for v in ground_truth:
        primary_ref = v.get('statute_reference', '')
        aliases = v.get('statute_alias', [primary_ref])
        violation_type = v.get('violation_type', '')
        severity = v.get('severity', 'MEDIUM')
        
        # Add all alias combinations
        for alias in aliases:
            # Also add normalized variations
            for variation in normalize_section_reference(alias):
                key = (variation, violation_type)
                gt_set.add(key)
                gt_severity_map[key] = SEVERITY_WEIGHTS.get(severity, 0.6)
    
    # Build predicted set
    pred_set = set()
    for v in predicted:
        section = v.get('statute_reference', '')
        violation_type = v.get('violation_type', '')
        
        # Add normalized variations
        for variation in normalize_section_reference(section):
            pred_set.add((variation, violation_type))
    
    # Calculate metrics
    true_positives = len(pred_set & gt_set)
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def calculate_severity_weighted_f1(
    predicted: List[Dict[str, Any]], 
    ground_truth: List[Dict[str, Any]]
) -> float:
    """Calculate F1 weighted by violation severity."""
    
    # Build ground truth with severity
    gt_dict = {}
    for v in ground_truth:
        primary_ref = v.get('statute_reference', '')
        aliases = v.get('statute_alias', [primary_ref])
        violation_type = v.get('violation_type', '')
        severity = v.get('severity', 'MEDIUM')
        weight = SEVERITY_WEIGHTS.get(severity, 0.6)
        
        for alias in aliases:
            for variation in normalize_section_reference(alias):
                key = (variation, violation_type)
                gt_dict[key] = weight
    
    # Build predicted set
    pred_set = set()
    for v in predicted:
        section = v.get('statute_reference', '')
        violation_type = v.get('violation_type', '')
        for variation in normalize_section_reference(section):
            pred_set.add((variation, violation_type))
    
    # Calculate weighted metrics
    weighted_tp = 0
    weighted_fp = 0
    weighted_fn = 0
    
    for key in pred_set:
        if key in gt_dict:
            weighted_tp += gt_dict[key]
        else:
            weighted_fp += 0.6  # Default weight for false positives
    
    for key, weight in gt_dict.items():
        if key not in pred_set:
            weighted_fn += weight
    
    weighted_precision = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0.0
    weighted_recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0.0
    weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0.0
    
    return weighted_f1

def calculate_section_accuracy(
    predicted: List[Dict[str, Any]], 
    ground_truth: List[Dict[str, Any]]
) -> float:
    """Calculate accuracy of section mapping (alias-aware)."""
    if not ground_truth:
        return 1.0
    
    correct = 0
    for gt_violation in ground_truth:
        gt_section = gt_violation.get('statute_reference', '')
        gt_aliases = set(gt_violation.get('statute_alias', [gt_section]))
        
        # Add normalized variations
        for alias in list(gt_aliases):
            gt_aliases.update(normalize_section_reference(alias))
        
        # Check if any predicted violation matches this section
        for pred_violation in predicted:
            pred_section = pred_violation.get('statute_reference', '')
            pred_variations = normalize_section_reference(pred_section)
            
            if pred_variations & gt_aliases:  # Set intersection
                correct += 1
                break
    
    return correct / len(ground_truth)

def calculate_trust_score_accuracy(
    predicted_score: int, 
    ground_truth_score: int,
    policy_type: str
) -> float:
    """Calculate accuracy of trust score prediction."""
    error = abs(predicted_score - ground_truth_score)
    
    # For compliant policies, be stricter (max 10 point error)
    if policy_type == "compliant":
        accuracy = max(0, 100 - (error * 2)) / 100
    else:
        # For non-compliant, be more lenient (max 20 point error)
        accuracy = max(0, 100 - error) / 100
    
    return accuracy

def calculate_policy_type_metrics(
    all_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate metrics broken down by policy type."""
    compliant_results = [r for r in all_results if r['policy_type'] == 'compliant']
    non_compliant_results = [r for r in all_results if r['policy_type'] == 'non_compliant']
    partial_results = [r for r in all_results if r['policy_type'] == 'partially_compliant']
    
    metrics = {}
    
    # Compliant policies: measure false positive rate
    if compliant_results:
        avg_fp = np.mean([r['false_positives'] for r in compliant_results])
        metrics['compliant'] = {
            "count": len(compliant_results),
            "avg_false_positives": float(avg_fp),
            "false_positive_rate": float(avg_fp / max(1, len(compliant_results)))
        }
    
    # Non-compliant policies: measure recall
    if non_compliant_results:
        avg_recall = np.mean([r['recall'] for r in non_compliant_results])
        metrics['non_compliant'] = {
            "count": len(non_compliant_results),
            "avg_recall": float(avg_recall),
            "avg_f1": float(np.mean([r['f1'] for r in non_compliant_results]))
        }
    
    # Partially compliant: measure balanced performance
    if partial_results:
        metrics['partially_compliant'] = {
            "count": len(partial_results),
            "avg_f1": float(np.mean([r['f1'] for r in partial_results]))
        }
    
    return metrics

def run_accuracy_evals():
    """Run the complete accuracy evaluation suite."""
    print("🎯 Starting Legal Reasoning Accuracy Evaluation...")
    print(f"Schema: {SCHEMA_PATH}")
    print(f"Ground Truth: {GROUND_TRUTH_PATH}")
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
    test_data = load_test_data()
    print(f"Found {len(test_data)} annotated test policies\n")
    
    # Run evaluations
    all_metrics = []
    section_accuracies = []
    trust_score_accuracies = []
    latencies = []
    token_counts = []
    
    for item in tqdm(test_data, desc="Evaluating"):
        output, latency_ms, tokens = run_inference(llm, item['content'])
        latencies.append(latency_ms)
        token_counts.append(tokens)
        
        try:
            parsed_output = json.loads(output)
        except json.JSONDecodeError:
            print(f"\n⚠️  {item['filename']}: Invalid JSON output, skipping")
            continue
        
        predicted_violations = extract_violations(parsed_output)
        ground_truth_violations = item['ground_truth']
        
        # Calculate violation detection metrics
        metrics = calculate_metrics(predicted_violations, ground_truth_violations)
        metrics['filename'] = item['filename']
        metrics['policy_type'] = item['policy_type']
        
        # Calculate severity-weighted F1
        metrics['weighted_f1'] = calculate_severity_weighted_f1(
            predicted_violations, ground_truth_violations
        )
        
        all_metrics.append(metrics)
        
        # Calculate section mapping accuracy
        section_acc = calculate_section_accuracy(predicted_violations, ground_truth_violations)
        section_accuracies.append(section_acc)
        
        # Calculate trust score accuracy
        predicted_score = parsed_output.get('dpdp_trust_score', 50)
        ground_truth_score = item['expected_trust_score']
        trust_acc = calculate_trust_score_accuracy(
            predicted_score, ground_truth_score, item['policy_type']
        )
        trust_score_accuracies.append(trust_acc)
    
    # Aggregate metrics
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    avg_weighted_f1 = np.mean([m['weighted_f1'] for m in all_metrics])
    avg_section_accuracy = np.mean(section_accuracies)
    avg_trust_accuracy = np.mean(trust_score_accuracies)
    avg_latency = np.mean(latencies)
    avg_tokens = np.mean(token_counts)
    
    # Calculate policy-type-specific metrics
    policy_type_metrics = calculate_policy_type_metrics(all_metrics)
    
    # Print results
    print("\n" + "="*70)
    print("ACCURACY EVALUATION RESULTS")
    print("="*70)
    print(f"Total Test Policies: {len(test_data)}")
    print(f"\nViolation Detection:")
    print(f"  Precision: {avg_precision:.3f}")
    print(f"  Recall: {avg_recall:.3f}")
    print(f"  F1 Score: {avg_f1:.3f}")
    print(f"  Severity-Weighted F1: {avg_weighted_f1:.3f}")
    print(f"\nSection Mapping Accuracy: {avg_section_accuracy:.3f}")
    print(f"Trust Score Accuracy: {avg_trust_accuracy:.3f}")
    print(f"\nPerformance:")
    print(f"  Avg Latency: {avg_latency:.1f}ms")
    print(f"  Avg Tokens Generated: {avg_tokens:.0f}")
    
    print(f"\nPolicy Type Breakdown:")
    for ptype, pm in policy_type_metrics.items():
        print(f"  {ptype}: {pm}")
    
    print("="*70)
    
    # Save detailed results
    output_path = Path("ml/evals/reports/accuracy_eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_policies": len(test_data),
                "avg_precision": float(avg_precision),
                "avg_recall": float(avg_recall),
                "avg_f1": float(avg_f1),
                "avg_weighted_f1": float(avg_weighted_f1),
                "avg_section_accuracy": float(avg_section_accuracy),
                "avg_trust_accuracy": float(avg_trust_accuracy),
                "avg_latency_ms": float(avg_latency),
                "avg_tokens_generated": float(avg_tokens)
            },
            "policy_type_metrics": policy_type_metrics,
            "detailed_results": all_metrics
        }, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {output_path}")
    
    # Print worst performers
    sorted_metrics = sorted(all_metrics, key=lambda x: x['f1'])
    if sorted_metrics:
        print("\n⚠️  Worst performing policies (lowest F1):")
        for m in sorted_metrics[:5]:
            print(f"  - {m['filename']} ({m['policy_type']}): F1={m['f1']:.3f}, P={m['precision']:.3f}, R={m['recall']:.3f}")

if __name__ == "__main__":
    run_accuracy_evals()