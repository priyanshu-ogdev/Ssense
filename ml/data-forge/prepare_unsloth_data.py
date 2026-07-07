#!/usr/bin/env python3
"""
prepare_unsloth_data.py – The Schema Translation Layer (Production Sealed)

Transforms structural ChatML arrays from the GAN Forge into the strict 
JSONL formatting expected by Unsloth and Hugging Face TRL 2026.
Optimized with multi-threaded disk I/O and strict data integrity auditing.
"""

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_sft_file(file_path):
    """Worker function to read and validate a single SFT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            pair = json.load(f)
        if "messages" in pair and isinstance(pair["messages"], list):
            return json.dumps(pair, ensure_ascii=False), None
        return None, "MISSING_MESSAGES_KEY"
    except json.JSONDecodeError as e:
        return None, f"JSON_DECODE_ERR: {str(e)}"
    except Exception as e:
        return None, f"OS_ERR: {str(e)}"

def process_dpo_file(file_path):
    """Worker function to read, validate, and structure a single DPO file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            pair = json.load(f)
        if "prompt" in pair and "chosen" in pair and "rejected" in pair:
            unsloth_record = {
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"]
            }
            return json.dumps(unsloth_record, ensure_ascii=False), None
        return None, "MISSING_DPO_KEYS"
    except json.JSONDecodeError as e:
        return None, f"JSON_DECODE_ERR: {str(e)}"
    except Exception as e:
        return None, f"OS_ERR: {str(e)}"

def prepare_sft_data(input_dir="./training-pairs/sft", output_file="../slm-training/data/sft_data.jsonl"):
    print("⚡ Aligning SFT data for Unsloth via High-Throughput Workers...")
    if not os.path.exists(input_dir):
        print(f"⚠️ Warning: SFT input directory '{input_dir}' not found. Skipping.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    if not files:
        print("ℹ️ No JSON files found in SFT directory.")
        return

    processed_count = 0
    corrupted_count = 0
    errors = {}

    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Utilizing ThreadPoolExecutor to prevent OS disk blocking during high-volume reads
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(process_sft_file, f): f for f in files}
            for future in as_completed(future_to_file):
                line, err = future.result()
                if line:
                    out_f.write(line + '\n')
                    processed_count += 1
                else:
                    corrupted_count += 1
                    errors[err] = errors.get(err, 0) + 1
                
    print(f"✅ SFT Alignment Complete: {processed_count}/{len(files)} pairs written to {output_file}")
    if corrupted_count > 0:
        print(f"❌ Alert: Dropped {corrupted_count} corrupted/invalid SFT records. Anomaly breakdown: {errors}")

def prepare_dpo_data(input_dir="./training-pairs/dpo", output_file="../slm-training/data/dpo_data.jsonl"):
    print("⚡ Aligning DPO data for Unsloth via High-Throughput Workers...")
    if not os.path.exists(input_dir):
        print(f"⚠️ Warning: DPO input directory '{input_dir}' not found. Skipping.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    if not files:
        print("ℹ️ No JSON files found in DPO directory.")
        return

    processed_count = 0
    corrupted_count = 0
    errors = {}

    with open(output_file, 'w', encoding='utf-8') as out_f:
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(process_dpo_file, f): f for f in files}
            for future in as_completed(future_to_file):
                line, err = future.result()
                if line:
                    out_f.write(line + '\n')
                    processed_count += 1
                else:
                    corrupted_count += 1
                    errors[err] = errors.get(err, 0) + 1
                
    print(f"✅ DPO Alignment Complete: {processed_count}/{len(files)} pairs written to {output_file}")
    if corrupted_count > 0:
        print(f"❌ Alert: Dropped {corrupted_count} corrupted/invalid DPO records. Anomaly breakdown: {errors}")
        if corrupted_count > (len(files) * 0.05): # Over 5% data loss triggers an explicit pipeline warning
            print("🚨 CRITICAL: Data corruption rate exceeds 5% threshold. Audit GAN Forge pipeline output.")

if __name__ == "__main__":
    print("⚙️ Initializing Ssense Data Alignment Layer (Async I/O Engine)...")
    prepare_sft_data()
    prepare_dpo_data()
    print("🚀 Data is structurally sealed. Ready for Unsloth Execution.")