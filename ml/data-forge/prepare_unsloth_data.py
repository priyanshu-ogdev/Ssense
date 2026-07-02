#!/usr/bin/env python3
"""
prepare_unsloth_data.py – The Schema Translation Layer (Final Sealed Version)

Transforms structural ChatML arrays from the GAN Forge into the strict 
JSONL formatting expected by Unsloth and Hugging Face TRL 2026.
"""

import os
import json

def prepare_sft_data(input_dir="./training-pairs/sft", output_file="./data/sft_data.jsonl"):
    print("Aligning SFT data for Unsloth...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    processed_count = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(input_dir):
            if filename.endswith('.json'):
                with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                    try:
                        pair = json.load(f)
                    except json.JSONDecodeError:
                        continue
                
                # GAN Forge already formatted this as a perfect ChatML "messages" array.
                # Unsloth expects {"messages": [{"role": "...", "content": "..."}]}
                if "messages" in pair:
                    out_f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                    processed_count += 1
                
    print(f"✅ SFT Alignment Complete: {processed_count} pairs written to {output_file}")

def prepare_dpo_data(input_dir="./training-pairs/dpo", output_file="./data/dpo_data.jsonl"):
    print("Aligning DPO data for Unsloth...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    processed_count = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(input_dir):
            if filename.endswith('.json'):
                with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                    try:
                        pair = json.load(f)
                    except json.JSONDecodeError:
                        continue
                
                # FIXED: The GAN Forge now outputs the native TRL 0.17.0 / Unsloth format.
                # We no longer need to slice the arrays.
                if "prompt" in pair and "chosen" in pair and "rejected" in pair:
                    unsloth_record = {
                        "prompt": pair["prompt"],
                        "chosen": pair["chosen"],
                        "rejected": pair["rejected"]
                    }
                    out_f.write(json.dumps(unsloth_record, ensure_ascii=False) + '\n')
                    processed_count += 1
                
    print(f"✅ DPO Alignment Complete: {processed_count} pairs written to {output_file}")

if __name__ == "__main__":
    print("⚙️ Initializing Ssense Data Alignment Layer...")
    prepare_sft_data()
    prepare_dpo_data()
    print("🚀 Data is structurally sealed. Ready for Unsloth Execution.")