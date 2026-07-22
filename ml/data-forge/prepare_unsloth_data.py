#!/usr/bin/env python3
"""
prepare_unsloth_data.py – Dual-Model Schema Translation Layer (Production Sealed)

Transforms structural JSON/ChatML files from the GAN Forge into two distinct,
strictly formatted Unsloth / Hugging Face TRL 2026 datasets:
1. Forensic Legal Auditor Model -> audit_sft_data.jsonl & audit_dpo_data.jsonl
2. Conversational Legal Chatbot Model -> chatbot_sft_data.jsonl & chatbot_dpo_data.jsonl

Optimized with:
- Multi-threaded disk I/O & absolute directory anchoring
- Character purity / unicode corruption (\ufffd/\u200b) firewall
- Exact EOS termination (.strip() on assistant turns guaranteeing clean closure before <|im_end|>)
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def strip_assistant_turn(msg):
    """Strip trailing whitespace right at assistant content boundaries for exact EOS alignment."""
    if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
        return {"role": msg["role"], "content": msg["content"].strip()}
    return msg

def process_sft_file(file_path):
    """Worker function to read, validate character purity, clean EOS boundaries, and verify SFT file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='strict') as f:
            raw_text = f.read()
            
        # Character Purity Firewall: Reject unicode replacement or zero-width space characters
        if "\ufffd" in raw_text or "\u200b" in raw_text:
            return None, "UNICODE_CORRUPTION_ERR (\ufffd or \u200b detected)"
            
        pair = json.loads(raw_text)
        if "messages" in pair and isinstance(pair["messages"], list):
            # Clean assistant turns so exact closing bracket '}' is immediately followed by <|im_end|>
            cleaned_messages = [strip_assistant_turn(m) for m in pair["messages"]]
            unsloth_record = {"messages": cleaned_messages}
            serialized = json.dumps(unsloth_record, ensure_ascii=False)
            if "\ufffd" in serialized or "\u200b" in serialized:
                return None, "UNICODE_CORRUPTION_IN_SERIALIZATION"
            return serialized, None
        return None, "MISSING_MESSAGES_KEY"
    except UnicodeDecodeError as e:
        return None, f"UNICODE_DECODE_ERR: {str(e)}"
    except json.JSONDecodeError as e:
        return None, f"JSON_DECODE_ERR: {str(e)}"
    except Exception as e:
        return None, f"OS_ERR: {str(e)}"

def process_dpo_file(file_path):
    """Worker function to read, validate character purity, clean EOS boundaries, and structure DPO file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='strict') as f:
            raw_text = f.read()
            
        # Character Purity Firewall: Reject unicode replacement or zero-width space characters
        if "\ufffd" in raw_text or "\u200b" in raw_text:
            return None, "UNICODE_CORRUPTION_ERR (\ufffd or \u200b detected)"
            
        pair = json.loads(raw_text)
        if "prompt" in pair and "chosen" in pair and "rejected" in pair and isinstance(pair["chosen"], list) and isinstance(pair["rejected"], list):
            cleaned_chosen = [strip_assistant_turn(m) for m in pair["chosen"]]
            cleaned_rejected = [strip_assistant_turn(m) for m in pair["rejected"]]
            unsloth_record = {
                "prompt": pair["prompt"],
                "chosen": cleaned_chosen,
                "rejected": cleaned_rejected
            }
            serialized = json.dumps(unsloth_record, ensure_ascii=False)
            if "\ufffd" in serialized or "\u200b" in serialized:
                return None, "UNICODE_CORRUPTION_IN_SERIALIZATION"
            return serialized, None
        return None, "MISSING_DPO_KEYS"
    except UnicodeDecodeError as e:
        return None, f"UNICODE_DECODE_ERR: {str(e)}"
    except json.JSONDecodeError as e:
        return None, f"JSON_DECODE_ERR: {str(e)}"
    except Exception as e:
        return None, f"OS_ERR: {str(e)}"

def resolve_directory(path_str):
    """Resolve directory path against Cwd first, then relative to BASE_DIR."""
    if os.path.exists(path_str):
        return path_str
    joined = os.path.join(BASE_DIR, path_str)
    if os.path.exists(joined):
        return joined
    return path_str

def resolve_output_file(path_str):
    """Resolve output file path relative to BASE_DIR if not absolute."""
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(BASE_DIR, path_str)

def process_dataset_dir(input_dir, output_file, mode="sft", dataset_name="Dataset"):
    """Core multi-threaded worker engine to process a specific dataset directory into an Unsloth JSONL file."""
    print(f"[{dataset_name.upper()}] Aligning {dataset_name} data for Unsloth via High-Throughput Workers...")
    resolved_in = resolve_directory(input_dir)
    resolved_out = resolve_output_file(output_file)
    os.makedirs(os.path.dirname(resolved_out), exist_ok=True)
    
    if not os.path.exists(resolved_in):
        print(f"[INFO] Input directory '{resolved_in}' not found yet. No {dataset_name} files to process.")
        return

    files = [os.path.join(resolved_in, f) for f in os.listdir(resolved_in) if f.endswith('.json')]
    if not files:
        print(f"[INFO] No JSON files found in {resolved_in}.")
        return

    worker_fn = process_sft_file if mode == "sft" else process_dpo_file
    processed_count = 0
    corrupted_count = 0
    errors = {}

    with open(resolved_out, 'w', encoding='utf-8') as out_f:
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(worker_fn, f): f for f in files}
            for future in as_completed(future_to_file):
                line, err = future.result()
                if line:
                    out_f.write(line + '\n')
                    processed_count += 1
                else:
                    corrupted_count += 1
                    errors[err] = errors.get(err, 0) + 1
                
    print(f"[SUCCESS] {dataset_name} Complete: {processed_count}/{len(files)} pairs written to {resolved_out}")
    if corrupted_count > 0:
        print(f"[ALERT] Dropped {corrupted_count} corrupted/invalid {dataset_name} records. Anomaly breakdown: {errors}")
        if corrupted_count > (len(files) * 0.05):
            print(f"[CRITICAL] {dataset_name} data corruption rate exceeds 5% threshold. Audit pipeline output.")

# --- Model 1: Forensic Legal Auditor Pipelines ---
def prepare_audit_sft_data(input_dir="./training-pairs/sft", output_file="../slm-training/data/audit_sft_data.jsonl"):
    process_dataset_dir(input_dir, output_file, mode="sft", dataset_name="Audit SFT")

def prepare_audit_dpo_data(input_dir="./training-pairs/dpo", output_file="../slm-training/data/audit_dpo_data.jsonl"):
    process_dataset_dir(input_dir, output_file, mode="dpo", dataset_name="Audit DPO")

# --- Model 2: Conversational Legal Chatbot Pipelines ---
def prepare_chatbot_sft_data(input_dir="./training-pairs/chatbot-sft", output_file="../slm-training/data/chatbot_sft_data.jsonl"):
    process_dataset_dir(input_dir, output_file, mode="sft", dataset_name="Chatbot SFT")

def prepare_chatbot_dpo_data(input_dir="./training-pairs/chatbot-dpo", output_file="../slm-training/data/chatbot_dpo_data.jsonl"):
    process_dataset_dir(input_dir, output_file, mode="dpo", dataset_name="Chatbot DPO")

# Backward-compatible master wrappers
def prepare_sft_data():
    prepare_audit_sft_data()
    prepare_chatbot_sft_data()

def prepare_dpo_data():
    prepare_audit_dpo_data()
    prepare_chatbot_dpo_data()

if __name__ == "__main__":
    print("[INIT] Initializing Ssense Dual-Model Data Alignment Layer (Async I/O Engine)...")
    print("\n--- Model 1: Forensic Legal Auditor Dataset ---")
    prepare_audit_sft_data()
    prepare_audit_dpo_data()
    print("\n--- Model 2: Conversational Legal Chatbot Dataset ---")
    prepare_chatbot_sft_data()
    prepare_chatbot_dpo_data()
    print("\n[READY] Both Audit and Chatbot datasets are structurally sealed and stored in slm-training/data.")