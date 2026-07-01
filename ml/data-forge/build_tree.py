#!/usr/bin/env python3
"""
build_dpdp_tree.py – Generates dpdp_act_tree.json from the raw law text.
Run this ONCE to bridge the ML data layer to the Rust network layer.
"""

import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

LAW_TEXT_PATH = "./dpdp_act_and_rules_2025.txt"
TREE_OUTPUT = "./dpdp_act_tree.json"

with open(LAW_TEXT_PATH, "r", encoding="utf-8") as f:
    law_text = f.read()

# 1. FORMAT AS PROPER CHAT MESSAGES
messages = [
    {"role": "system", "content": "You are a senior legal engineer specializing in the DPDP Act."},
    {"role": "user", "content": f"""Given the full text of the DPDP Act 2023 and Rules 2025, produce a JSON object that maps every section, sub-section, and rule that mandates a specific user-facing data practice to its enforcement action.

CRITICAL FORMATTING: 
For the keys, you MUST use the exact strict citation format: "DPDP Act Sec X(Y)" or "DPDP Rules 2025 Rule X(Y)". Do not deviate from this format.

For each key, map it to the enforcement parameters.
Law text:\n{law_text}"""}
]

# Define the dynamic JSON schema to strictly enforce the output structure
tree_schema = {
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "violation_type": {  # FIXED to match SLM Audit Schema
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
            "network_action": {  # FIXED to match SLM Audit Schema
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
    model="/path/to/your/72B-FP8-Model", 
    quantization="fp8", 
    tensor_parallel_size=2, 
    max_model_len=16384
)

# Use Guided Decoding to guarantee pure JSON output (No markdown backticks generated)
guided_params = GuidedDecodingParams(json=tree_schema)
params = SamplingParams(temperature=0.0, max_tokens=10000, guided_decoding=guided_params)

print("Generating Deterministic Enforcement Tree...")
# Note the [messages] wrapper required by the vLLM 0.24 batch API
output = llm.chat(messages=[messages], sampling_params=params)
tree_raw_text = output[0].outputs[0].text.strip()

# Validate and save directly (No string stripping required)
try:
    tree = json.loads(tree_raw_text)
    with open(TREE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"✅ dpdp_act_tree.json generated flawlessly with {len(tree)} enforceable clauses.")
    print("Move this file to ssense/apps/browser-core/src-tauri/src/config/ for the Rust compiler.")
except json.JSONDecodeError as e:
    print(f"❌ JSON Parsing Error: {e}")