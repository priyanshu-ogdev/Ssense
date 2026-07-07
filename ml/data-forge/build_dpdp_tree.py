#!/usr/bin/env python3
"""
build_dpdp_tree.py – Generates dpdp_act_tree.json from the raw law text.
Run this ONCE to bridge the ML data layer to the Rust network layer.
"""

import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

LAW_TEXT_PATH = "./dpdp_act_and_rules_2025.txt"
TREE_OUTPUT = "./dpdp_act_tree.json"
MODEL_PATH = "../models/Qwen2-72B-Instruct-FP8" 

with open(LAW_TEXT_PATH, "r", encoding="utf-8") as f:
    law_text = f.read()

messages = [
    {"role": "system", "content": "You are a senior legal engineer specializing in the DPDP Act."},
    {"role": "user", "content": f"""Given the full text of the DPDP Act 2023 and Rules 2025, produce a JSON object that maps every section, sub-section, and rule that mandates a specific user-facing data practice to its enforcement action.

CRITICAL FORMATTING: 
For the keys, you MUST use the exact strict citation format: "DPDP Act Sec X(Y)" or "DPDP Rules 2025 Rule X(Y)". Do not deviate from this format.

For each key, map it to the enforcement parameters.
Law text:\n{law_text}"""}
]

tree_schema = {
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "violation_type": {  
                "type": "string",
                "enum": [
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
                ]
            },
            "network_action": {  
                "type": "string",
                "enum": [
                    "BLOCK_THIRD_PARTY", 
                    "STRIP_TELEMETRY_HEADER", 
                    "SPOOF_HARDWARE_API", 
                    "INJECT_GPC_SIGNAL", 
                    "WARN_USER_ONLY"
                ]
            },
            "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
            "description": {"type": "string"}
        },
        "required": ["violation_type", "network_action", "severity", "description"]
    }
}

print("Loading 72B model for legal parsing...")
llm = LLM(
    model=MODEL_PATH, 
    quantization="fp8", 
    tensor_parallel_size=1,       
    max_model_len=32768,          # ✅ MATHEMATICAL FIX: Capped at model's native limit
    gpu_memory_utilization=0.75,  # ✅ UMA SAFETY FIX: Protects the OS from OOM panics
    kv_cache_dtype="fp8",         
    enable_chunked_prefill=True   
)

guided_params = StructuredOutputsParams(json=tree_schema)
# ✅ MATH FIX: Dropped max_tokens to 8192 so Input (14k) + Output (8k) = 22k (Safely under 32k)
params = SamplingParams(temperature=0.0, max_tokens=8192, structured_outputs=guided_params)

print("Generating Deterministic Enforcement Tree...")
# ✅ SYNTAX FIX: Pass 'messages' directly. vLLM handles the list of dicts.
output = llm.chat(messages=messages, sampling_params=params)
tree_raw_text = output[0].outputs[0].text.strip()

try:
    tree = json.loads(tree_raw_text)
    with open(TREE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"✅ dpdp_act_tree.json generated flawlessly with {len(tree)} enforceable clauses.")
    print("Move this file to ssense/apps/browser-core/src-tauri/src/config/ for the Rust compiler.")
except json.JSONDecodeError as e:
    print(f"❌ JSON Parsing Error: {e}")