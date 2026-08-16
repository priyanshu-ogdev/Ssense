#!/usr/bin/env python3
"""
check_data_leakage.py – MinHash/Jaccard Data Leakage Detection for DPDP Eval Suite

Scans all evaluation benchmark datasets against all training data files to detect
accidental contamination (eval data that appears in training data).

If eval data leaked into training data, the certification is invalid:
you're measuring memorization, not generalization.

Uses MinHash for approximate Jaccard similarity at O(n) complexity per pair,
making it feasible even for large datasets.

Usage:
    python ml/scripts/check_data_leakage.py
    python ml/scripts/check_data_leakage.py --threshold 0.80 --verbose

Hard-fails (exit code 1) if any similarity > threshold is detected.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import json
import re
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════════════════
# PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════
_SCRIPT_DIR = Path(__file__).resolve().parent
_ML_DIR = _SCRIPT_DIR.parent
_ROOT_DIR = _ML_DIR.parent
_EVALS_DIR = _ML_DIR / "evals"
_TRAINING_DATA_DIR = _ML_DIR / "slm-training" / "data"


# ═══════════════════════════════════════════════════════════════════════════
# MINHASH IMPLEMENTATION (no external dependency)
# ═══════════════════════════════════════════════════════════════════════════
def _tokenize(text: str) -> List[str]:
    """Normalize and tokenize text into word n-grams for shingling."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    # Use 7-gram shingles for better matching to avoid legal boilerplate false positives
    shingles = []
    for i in range(len(words) - 6):
        shingles.append(' '.join(words[i:i+7]))
    return shingles if shingles else words


def _hash_shingle(shingle: str, seed: int) -> int:
    """Hash a shingle with a seed for MinHash."""
    h = hashlib.md5(f"{seed}:{shingle}".encode('utf-8')).hexdigest()
    return int(h[:16], 16)


class MinHash:
    """Lightweight MinHash signature for approximate Jaccard similarity."""

    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm
        self.hashvalues = [float('inf')] * num_perm

    def update(self, text: str):
        """Add text to the MinHash signature."""
        shingles = _tokenize(text)
        for shingle in shingles:
            for i in range(self.num_perm):
                h = _hash_shingle(shingle, i)
                if h < self.hashvalues[i]:
                    self.hashvalues[i] = h

    @staticmethod
    def jaccard(mh1: "MinHash", mh2: "MinHash") -> float:
        """Estimate Jaccard similarity between two MinHash signatures."""
        if len(mh1.hashvalues) != len(mh2.hashvalues):
            raise ValueError("MinHash signatures must have the same number of permutations")
        matches = sum(1 for a, b in zip(mh1.hashvalues, mh2.hashvalues) if a == b)
        return matches / len(mh1.hashvalues)


# ═══════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
def extract_eval_texts(eval_dir: Path) -> List[Dict[str, str]]:
    """Extract text content from all evaluation benchmark files."""
    texts = []

    # 1. Ground truth policies
    gt_path = eval_dir / "holdout_policies" / "ground_truth.json"
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                snippet = item.get("policy_text_snippet", "")
                if snippet:
                    texts.append({
                        "source": f"ground_truth:{item.get('case_id', 'unknown')}",
                        "text": snippet
                    })

    # 2. Chatbot QA
    qa_path = eval_dir / "benchmarks" / "dpdp_chatbot_qa.json"
    if qa_path.exists():
        with open(qa_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                q = item.get("query", item.get("question", ""))
                if q:
                    texts.append({
                        "source": f"chatbot_qa:{item.get('id', 'unknown')}",
                        "text": q
                    })

    # 3. RAG test set
    rag_path = eval_dir / "benchmarks" / "dpdp_rag_testset.json"
    if rag_path.exists():
        with open(rag_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                q = item.get("query", "")
                if q:
                    texts.append({
                        "source": f"rag_testset:{item.get('id', 'unknown')}",
                        "text": q
                    })

    # 4. Red-team hallucination prompts
    rt_path = eval_dir / "benchmarks" / "redteam_hallucination_prompts.json"
    if rt_path.exists():
        with open(rt_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                p = item.get("query", item.get("prompt", ""))
                if p:
                    texts.append({
                        "source": f"redteam:{item.get('id', 'unknown')}",
                        "text": p
                    })

    # 5. Security adversarial suite
    sec_path = eval_dir / "benchmarks" / "security_adversarial_suite.json"
    if sec_path.exists():
        with open(sec_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for key in ["prompt_injection_refusals", "prompt_injection", "sycophancy_tests", "sycophancy_traps", "json_schema_fuzzing", "json_fuzzing", "niah_context_injection"]:
                for item in data.get(key, []):
                    p = item.get("query", item.get("prompt", item.get("input_payload", item.get("chaotic_policy_text", item.get("context", "")))))
                    if p:
                        texts.append({
                            "source": f"security:{key}:{item.get('id', 'unknown')}",
                            "text": p
                        })

    return texts


def extract_training_texts(training_dir: Path) -> List[Dict[str, str]]:
    """Extract text content from all training data files."""
    texts = []
    if not training_dir.exists():
        print(f"⚠️ Training data directory not found: {training_dir}")
        return texts

    for f in sorted(training_dir.rglob("*.json")) + sorted(training_dir.rglob("*.jsonl")):
        try:
            content = f.read_text(encoding="utf-8").strip()
            if content.startswith("["):
                # JSON array
                items = json.loads(content)
                for idx, item in enumerate(items):
                    text = _extract_text_from_item(item)
                    if text:
                        texts.append({
                            "source": f"training:{f.name}:{idx}",
                            "text": text
                        })
            else:
                # JSONL
                for idx, line in enumerate(content.splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        text = _extract_text_from_item(item)
                        if text:
                            texts.append({
                                "source": f"training:{f.name}:{idx}",
                                "text": text
                            })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️ Could not parse {f}: {e}")
            continue

    return texts


def _extract_text_from_item(item: dict) -> str:
    """Extract meaningful text from a training data item (various formats)."""
    if not isinstance(item, dict):
        return ""
    # Try common fields
    for key in ["text", "input", "instruction", "prompt", "policy_text",
                 "policy_text_snippet", "content", "question", "query"]:
        if key in item and isinstance(item[key], str) and len(item[key]) > 20:
            return item[key]
    # Try conversation format
    if "conversations" in item and isinstance(item["conversations"], list):
        parts = []
        for msg in item["conversations"]:
            if isinstance(msg, dict) and "value" in msg:
                parts.append(str(msg["value"]))
        return " ".join(parts)
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="MinHash Data Leakage Detection: eval benchmarks vs training data"
    )
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Jaccard similarity threshold for flagging leakage (default: 0.85)")
    parser.add_argument("--num-perm", type=int, default=128,
                        help="Number of MinHash permutations (default: 128)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print all pairwise similarities, not just violations")
    parser.add_argument("--eval-dir", type=str, default=str(_EVALS_DIR))
    parser.add_argument("--training-dir", type=str, default=str(_TRAINING_DATA_DIR))
    args = parser.parse_args()

    print("🔍 DPDP Eval Suite Data Leakage Detection")
    print(f"   Threshold: Jaccard similarity > {args.threshold}")
    print(f"   Eval data:     {args.eval_dir}")
    print(f"   Training data: {args.training_dir}")
    print()

    # Extract texts
    eval_texts = extract_eval_texts(Path(args.eval_dir))
    training_texts = extract_training_texts(Path(args.training_dir))

    if not eval_texts:
        print("⚠️ No eval texts found. Nothing to check.")
        return 0
    if not training_texts:
        print("⚠️ No training texts found. Nothing to check against.")
        return 0

    print(f"   Eval items:     {len(eval_texts)}")
    print(f"   Training items: {len(training_texts)}")
    print()

    # Build MinHash signatures
    print("🔧 Building MinHash signatures...")
    eval_sigs = []
    for item in eval_texts:
        mh = MinHash(num_perm=args.num_perm)
        mh.update(item["text"])
        eval_sigs.append((item, mh))

    training_sigs = []
    for item in training_texts:
        mh = MinHash(num_perm=args.num_perm)
        mh.update(item["text"])
        training_sigs.append((item, mh))

    # Compare all eval items against all training items
    print("🔎 Scanning for leakage...")
    violations = []
    for eval_item, eval_mh in eval_sigs:
        for train_item, train_mh in training_sigs:
            sim = MinHash.jaccard(eval_mh, train_mh)
            if sim > args.threshold:
                violations.append({
                    "eval_source": eval_item["source"],
                    "train_source": train_item["source"],
                    "jaccard_similarity": round(sim, 4),
                    "eval_text_preview": eval_item["text"][:150],
                    "train_text_preview": train_item["text"][:150]
                })
            elif args.verbose and sim > 0.5:
                print(f"   ℹ️ {eval_item['source']} ↔ {train_item['source']}: {sim:.4f}")

    # Report
    print()
    print("═" * 70)
    if violations:
        print(f"🚨 DATA LEAKAGE DETECTED: {len(violations)} pair(s) exceed threshold {args.threshold}")
        print("═" * 70)
        for v in violations:
            print(f"\n   ❌ Eval:  {v['eval_source']}")
            print(f"      Train: {v['train_source']}")
            print(f"      Jaccard Similarity: {v['jaccard_similarity']}")
            print(f"      Eval preview:  {v['eval_text_preview']}...")
            print(f"      Train preview: {v['train_text_preview']}...")
        print()
        print("❌ HARD-FAIL: Evaluation data has leaked into training data.")
        print("   Any certification run using this eval data is INVALID.")
        print("   Remove or replace the contaminated eval items before re-running.")

        # Write violation report
        report_path = Path(args.eval_dir) / "reports" / "data_leakage_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "threshold": args.threshold,
            "eval_items_scanned": len(eval_texts),
            "training_items_scanned": len(training_texts),
            "violations_found": len(violations),
            "violations": violations
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"   📄 Violation report: {report_path}")

        return 1
    else:
        print("✅ NO DATA LEAKAGE DETECTED")
        print("═" * 70)
        print(f"   Scanned {len(eval_texts)} eval items against {len(training_texts)} training items.")
        print(f"   No pair exceeded Jaccard similarity threshold of {args.threshold}.")
        print("   Evaluation benchmarks are safe to use for certification.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
