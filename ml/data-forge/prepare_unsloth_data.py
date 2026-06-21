#!/usr/bin/env python3
"""
prepare_unsloth_data.py – The Schema Translation Layer (Final Sealed Version)

Transforms raw synthetic outputs from the GAN Forge into the strict 
ChatML dictionaries expected by Unsloth and Hugging Face TRL.
"""

import os
import json

# THE CRYPTOGRAPHIC KEY
SYSTEM_PROMPT = "You are a strict DPDP Regulatory Auditor enforcing the Indian Digital Personal Data Protection (DPDP) Act 2023 and Rules 2025. Output ONLY valid JSON matching the dpdp_schema."

# Load the DPDP Act context (Must be identical to what gan_forge.py used)
# Assuming you saved the law text to a file during the forge phase
LAW_CONTEXT = ""
if os.path.exists("./dpdp_act_and_rules_2025.txt"):
    with open("./dpdp_act_and_rules_2025.txt", 'r', encoding='utf-8') as f:
        LAW_CONTEXT = f.read()

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
                
                # SFT Schema: Unsloth expects a single 'messages' array
                unsloth_record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": pair.get('input', '')}, # GAN forge already injected the law here
                        {"role": "assistant", "content": pair.get('output', '')}
                    ]
                }
                out_f.write(json.dumps(unsloth_record) + '\n')
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
                
                # CRITICAL FIX: Reconstruct the full prompt with the DPDP Act
                # to prevent distribution shift between SFT and DPO phases.
                raw_policy = pair.get('prompt', '')
                full_user_prompt = f"[CONTEXT: THE LAW]\n{LAW_CONTEXT}\n\n[SYNTHESIZED POLICY]\n{raw_policy}"
                
                # DPO Schema: Unsloth/TRL expects prompt, chosen, and rejected as lists of dicts
                unsloth_record = {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": full_user_prompt}
                    ],
                    "chosen": [
                        {"role": "assistant", "content": pair.get('chosen', '')}
                    ],
                    "rejected": [
                        {"role": "assistant", "content": pair.get('rejected', '')}
                    ]
                }
                out_f.write(json.dumps(unsloth_record) + '\n')
                processed_count += 1
                
    print(f"✅ DPO Alignment Complete: {processed_count} pairs written to {output_file}")

if __name__ == "__main__":
    print("⚙️ Initializing Ssense Data Alignment Layer...")
    prepare_sft_data()
    prepare_dpo_data()
    print("🚀 Data is structurally sealed. Ready for Unsloth Execution.")