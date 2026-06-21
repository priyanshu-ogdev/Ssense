#!/usr/bin/env python3
"""
build_dpdp_tree.py – Generates dpdp_act_tree.json from the raw law text.
Run this ONCE to bridge the ML data layer to the Rust network layer.
"""

import json
from vllm import LLM, SamplingParams

LAW_TEXT_PATH = "./dpdp_act_and_rules_2025.txt"
TREE_OUTPUT = "./dpdp_act_tree.json"

with open(LAW_TEXT_PATH, "r", encoding="utf-8") as f:
    law_text = f.read()

# 1. FORMAT AS PROPER CHAT MESSAGES
messages = [
    {"role": "system", "content": "You are a senior legal engineer specializing in the DPDP Act."},
    {"role": "user", "content": f"""Given the full text of the DPDP Act 2023 and Rules 2025, produce a JSON object that maps every section, sub-section, and rule that mandates a specific user-facing data practice to its enforcement action.

For each key (the exact statute reference string, e.g., "DPDP Act Sec 8(3)"), include:
- violation_category (one of: "CONSENT_NOT_FREE_OR_SPECIFIC", "PURPOSE_LIMITATION_VIOLATION", "NOTICE_INADEQUATE", "DATA_RETENTION_LIMIT_EXCEEDED", "CHILD_CONSENT_VIOLATION", "SECURITY_SAFEGUARDS_MISSING", "GRIEVANCE_REDRESSAL_INADEQUATE", "BREACH_NOTIFICATION_FAILURE")
- required_network_action (one of: "BLOCK_THIRD_PARTY", "STRIP_TELEMETRY_HEADER", "SPOOF_HARDWARE_API", "INJECT_GPC_SIGNAL", "WARN_USER_ONLY")
- severity ("LOW", "MEDIUM", "HIGH", "CRITICAL")
- description (A short string explaining the violation)

Output ONLY valid JSON, no explanation, no markdown blocks.

Law text:
{law_text}
"""}
]

print("Loading 72B model for legal parsing...")
llm = LLM(
    model="/path/to/your/72B-FP8-Model", 
    quantization="fp8", 
    tensor_parallel_size=2, 
    max_model_len=16384
)
params = SamplingParams(temperature=0.0, max_tokens=8192) # Increased tokens to ensure large trees aren't cut off

print("Generating Deterministic Enforcement Tree...")
output = llm.chat(messages=messages, sampling_params=params)
tree_raw_text = output[0].outputs[0].text.strip()

# 2. SAFELY STRIP MARKDOWN BACKTICKS
if tree_raw_text.startswith("```json"):
    tree_raw_text = tree_raw_text[7:]
if tree_raw_text.startswith("```"):
    tree_raw_text = tree_raw_text[3:]
if tree_raw_text.endswith("```"):
    tree_raw_text = tree_raw_text[:-3]

tree_raw_text = tree_raw_text.strip()

# Validate and save
try:
    tree = json.loads(tree_raw_text)
    with open(TREE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2)
    print(f"✅ dpdp_act_tree.json generated flawlessly with {len(tree)} enforceable clauses.")
    print("Move this file to ssense/apps/browser-core/src-tauri/src/config/ for the Rust compiler.")
except json.JSONDecodeError as e:
    print(f"❌ JSON Parsing Error: {e}")
    print("Raw output was:")
    print(tree_raw_text)