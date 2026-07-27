#!/usr/bin/env python3
"""
build_dpdp_tree.py – Generates dpdp_act_tree.json from the raw law text.
Run this ONCE to bridge the ML data layer to the Rust network layer.
"""

import json
import os
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

LAW_TEXT_PATH = "./dpdp_act_and_rules_2025.txt"
TREE_OUTPUT = "./dpdp_act_tree.json"
MODEL_PATH = os.getenv("TEACHER_MODEL_PATH", "../models/Qwen2-72B-Instruct-FP8") 

with open(LAW_TEXT_PATH, "r", encoding="utf-8") as f:
    law_text = f.read()

# ═══════════════════════════════════════════════════════════════════════════
# STRICT DPDP TREE GENERATION PROMPT
# ═══════════════════════════════════════════════════════════════════════════
DPDP_TREE_PROMPT = """[TASK]
Analyze the provided law text and produce a strict JSON object that maps every actionable section, sub-section, and rule to its deterministic enforcement parameters. This JSON will be directly compiled into a Rust network interceptor.

[STRICT FORMATTING & ENUM RULES]
1. KEYS: You MUST use the exact citation format: "DPDP Act Sec X(Y)" or "DPDP Rules 2025 Rule X(Y)". Do not invent subsections if they do not exist in the text.
2. FOREIGN LAW BLOCK: NEVER cite GDPR, CCPA, or any non-Indian law.
3. ENUMS: For each key, you MUST map it to EXACTLY ONE enum value from the following lists. Do not invent new strings, do not use lowercase, and do not add spaces.

VIOLATION_TYPE ENUMS (16 TOTAL):
- PURPOSE_LIMITATION_VIOLATION
- CONSENT_NOT_FREE_OR_SPECIFIC
- LEGITIMATE_USES_ABUSE
- NOTICE_INADEQUATE
- DATA_RETENTION_LIMIT_EXCEEDED
- CHILD_CONSENT_VIOLATION
- SECURITY_SAFEGUARDS_MISSING
- BREACH_NOTIFICATION_FAILURE
- PROCESSOR_ACCOUNTABILITY_VIOLATION
- GRIEVANCE_REDRESSAL_INADEQUATE
- SDF_OBLIGATIONS_MISSING
- CROSS_BORDER_TRANSFER_VIOLATION
- CONSENT_MANAGER_OBSTRUCTION
- LANGUAGE_ACCESSIBILITY
- ALGORITHMIC_PROFILING_SDF
- RIGHTS_IMPLEMENTATION_VIOLATION

NETWORK_ACTION ENUMS:
- BLOCK_THIRD_PARTY
- STRIP_TELEMETRY_HEADER
- SPOOF_HARDWARE_API
- INJECT_GPC_SIGNAL
- WARN_USER_ONLY

SEVERITY ENUMS:
- LOW
- MEDIUM
- HIGH
- CRITICAL

[OUTPUT SCHEMA]
{
  "DPDP Act Sec X(Y)": {
    "violation_type": "EXACT_ENUM_FROM_ABOVE",
    "network_action": "EXACT_ENUM_FROM_ABOVE",
    "severity": "EXACT_ENUM_FROM_ABOVE",
    "description": "A concise, 1-sentence technical description of what the network interceptor should look for or block."
  }
}

Generate the complete JSON mapping now. Output ONLY the raw JSON object. Do not use markdown backticks.

[LAW TEXT]
""" + law_text

messages = [
    {"role": "system", "content": "You are a senior legal engineer and Rust backend architect specializing in the DPDP Act 2023 and Rules 2025."},
    {"role": "user", "content": DPDP_TREE_PROMPT}
]

# ═══════════════════════════════════════════════════════════════════════════
# UPDATED JSON SCHEMA (16 CATEGORIES)
# ═══════════════════════════════════════════════════════════════════════════
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
                    "LEGITIMATE_USES_ABUSE",
                    "NOTICE_INADEQUATE", 
                    "DATA_RETENTION_LIMIT_EXCEEDED", 
                    "CHILD_CONSENT_VIOLATION", 
                    "SECURITY_SAFEGUARDS_MISSING", 
                    "BREACH_NOTIFICATION_FAILURE",
                    "PROCESSOR_ACCOUNTABILITY_VIOLATION",
                    "GRIEVANCE_REDRESSAL_INADEQUATE", 
                    "SDF_OBLIGATIONS_MISSING", 
                    "CROSS_BORDER_TRANSFER_VIOLATION",
                    "CONSENT_MANAGER_OBSTRUCTION",
                    "LANGUAGE_ACCESSIBILITY",
                    "ALGORITHMIC_PROFILING_SDF",
                    "RIGHTS_IMPLEMENTATION_VIOLATION"
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
    max_model_len=32768,          
    gpu_memory_utilization=0.75,  
    kv_cache_dtype="fp8",         
    enable_chunked_prefill=True   
)

guided_params = StructuredOutputsParams(json=tree_schema)
params = SamplingParams(temperature=0.0, max_tokens=8192, structured_outputs=guided_params)

print("Generating Deterministic Enforcement Tree...")
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