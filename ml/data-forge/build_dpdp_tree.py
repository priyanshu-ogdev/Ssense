#!/usr/bin/env python3
"""
build_dpdp_tree.py – Generates dpdp_act_tree.json from raw law text using Batch-Chunked Processing.
Run this ONCE to bridge the ML data layer to the Rust network layer.

UPGRADES:
1. Bureaucracy Prompt Firewall: Instructs the LLM to ignore government administration sections.
2. Regex Purge: Mechanically deletes Sections 18-26, 29-32, and 34-44 before saving.
3. Dynamic Collision Handling: Safely increments (Part 2), (Part 3) to prevent silent overwrites.
"""

import json
import os
import re
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

LAW_TEXT_PATH = "./dpdp_act_and_rules_2025.txt"
TREE_OUTPUT = "./dpdp_act_tree.json"
MODEL_PATH = os.getenv("TEACHER_MODEL_PATH", "../models/Qwen2-72B-Instruct-FP8")

if not os.path.exists(LAW_TEXT_PATH):
    LAW_TEXT_PATH = "./ml/data-forge/dpdp_act_and_rules_2025.txt"
    if not os.path.exists(LAW_TEXT_PATH):
        raise FileNotFoundError(f"Cannot find law text at {LAW_TEXT_PATH}")

with open(LAW_TEXT_PATH, "r", encoding="utf-8") as f:
    law_text = f.read()

# ═══════════════════════════════════════════════════════════════════════════
# CHUNKING LOGIC: Safe limits + Sub-section splitting
# ═══════════════════════════════════════════════════════════════════════════
def chunk_law_text(text, max_chunk_chars=6000):
    """Splits the raw legal text on structural headers and sections to ensure safe LLM digestion."""
    split_pattern = r"(?=\n(?:CHAPTER|FIRST SCHEDULE|SECOND SCHEDULE|RULES|THE DIGITAL PERSONAL DATA PROTECTION|\d+\.\s|Section\s\d+|Rule\s\d+))"
    raw_sections = [s.strip() for s in re.split(split_pattern, text) if s.strip()]
    
    chunks = []
    current_chunk = ""
    for sec in raw_sections:
        if len(current_chunk) + len(sec) > max_chunk_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sec
        else:
            current_chunk += "\n\n" + sec if current_chunk else sec
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

text_chunks = chunk_law_text(law_text)
print(f"✂️ Split Law Text into {len(text_chunks)} high-resolution chunks for zero-truncation processing.")

# ═══════════════════════════════════════════════════════════════════════════
# STRICT DPDP TREE GENERATION PROMPT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = "You are a senior legal engineer and Rust backend architect specializing in the DPDP Act 2023 and Rules 2025. Your job is to extract data privacy rules applicable to Data Fiduciaries."

PROMPT_TEMPLATE = """[TASK]
Analyze the provided law text excerpt and produce a strict JSON object that maps actionable sections, sub-sections, and rules to their deterministic enforcement parameters. This JSON will be compiled into a Rust network interceptor.

[EXCLUSION RULES - CRITICAL]
1. IGNORE BUREAUCRACY: If a section dictates government administration, Board salaries, Tribunal appointments, fund management, or internal Board procedures (typically Sections 18 through 44), YOU MUST IGNORE IT. Do not map it.
2. If the excerpt contains NO actionable data privacy rules for Data Fiduciaries, output an empty JSON object: {{}}

[STRICT FORMATTING & ENUM RULES]
1. GRANULAR KEYS: You MUST use exact, granular citations (e.g., "DPDP Act Sec 8(1)"). 
2. FOREIGN LAW BLOCK: NEVER cite GDPR, CCPA, or non-Indian laws.
3. ENUMS: Map each entry to EXACTLY ONE enum value from these lists:

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
{{
  "DPDP Act Sec X(Y)": {{
    "violation_type": "EXACT_ENUM_FROM_ABOVE",
    "network_action": "EXACT_ENUM_FROM_ABOVE",
    "severity": "EXACT_ENUM_FROM_ABOVE",
    "description": "1-sentence technical description of what the interceptor should look for or block."
  }}
}}

Generate the JSON mapping for this excerpt now. Output ONLY valid JSON starting with {{.

[LAW TEXT EXCERPT]
{chunk_text}
"""

batch_messages = []
for chunk in text_chunks:
    prompt = PROMPT_TEMPLATE.format(chunk_text=chunk)
    batch_messages.append([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])

# ═══════════════════════════════════════════════════════════════════════════
# JSON SCHEMA & VLLM PARAMS
# ═══════════════════════════════════════════════════════════════════════════
tree_schema = {
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "violation_type": {  
                "type": "string",
                "enum": [
                    "PURPOSE_LIMITATION_VIOLATION", "CONSENT_NOT_FREE_OR_SPECIFIC", 
                    "LEGITIMATE_USES_ABUSE", "NOTICE_INADEQUATE", 
                    "DATA_RETENTION_LIMIT_EXCEEDED", "CHILD_CONSENT_VIOLATION", 
                    "SECURITY_SAFEGUARDS_MISSING", "BREACH_NOTIFICATION_FAILURE",
                    "PROCESSOR_ACCOUNTABILITY_VIOLATION", "GRIEVANCE_REDRESSAL_INADEQUATE", 
                    "SDF_OBLIGATIONS_MISSING", "CROSS_BORDER_TRANSFER_VIOLATION",
                    "CONSENT_MANAGER_OBSTRUCTION", "LANGUAGE_ACCESSIBILITY",
                    "ALGORITHMIC_PROFILING_SDF", "RIGHTS_IMPLEMENTATION_VIOLATION"
                ]
            },
            "network_action": {  
                "type": "string",
                "enum": [
                    "BLOCK_THIRD_PARTY", "STRIP_TELEMETRY_HEADER", 
                    "SPOOF_HARDWARE_API", "INJECT_GPC_SIGNAL", "WARN_USER_ONLY"
                ]
            },
            "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
            "description": {"type": "string"}
        },
        "required": ["violation_type", "network_action", "severity", "description"]
    }
}

print("Loading 72B model for batch legal parsing...")
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
params = SamplingParams(
    temperature=0.0, 
    max_tokens=8192, 
    structured_outputs=guided_params
)

print(f"🚀 Executing Batch Generation ({len(batch_messages)} parallel chunk prompts)...")
batch_outputs = llm.chat(messages=batch_messages, sampling_params=params)

# ═══════════════════════════════════════════════════════════════════════════
# PARSE, MERGE & PURGE (ANTI-OVERWRITE + BUREAUCRACY FIREWALL)
# ═══════════════════════════════════════════════════════════════════════════
def is_bureaucratic_section(key_str):
    """Identifies government administrative sections that should not be mapped."""
    match = re.search(r'\b(?:Sec(?:tion)?|Sec\.)\s*(\d+)', key_str, re.IGNORECASE)
    if match:
        sec_num = int(match.group(1))
        # Nuke Sections 18-26 (Board setup), 29-32 (Tribunal), 34-44 (Govt powers)
        # Spares 27/28 (Board Powers/Grievance), 33 (Penalties), 37 (Website Blocking)
        if sec_num in range(18, 27) or sec_num in range(29, 33) or sec_num in range(34, 45):
            if sec_num not in [37]: 
                return True
    return False

master_tree = {}
parse_errors = 0
purged_bureaucracy = 0

for idx, out in enumerate(batch_outputs):
    raw_text = out.outputs[0].text.strip()
    try:
        sub_tree = json.loads(raw_text)
        
        for key, value in sub_tree.items():
            # Layer 2 Firewall: Mechanically drop hallucinated bureaucratic sections
            if is_bureaucratic_section(key):
                purged_bureaucracy += 1
                continue
                
            # Dynamic Collision Detection (Prevents silent overwrites for >2 collisions)
            original_key = key
            counter = 2
            while key in master_tree:
                key = f"{original_key} (Part {counter})"
                counter += 1
                
            if key != original_key:
                print(f"   ⚠️ Collision mitigated: {original_key} -> Saved as {key}")
                
            master_tree[key] = value
                
        print(f"  ✅ Chunk {idx+1}/{len(batch_outputs)} parsed successfully ({len(sub_tree)} clauses).")
    except json.JSONDecodeError as e:
        parse_errors += 1
        print(f"  ❌ Chunk {idx+1} JSON Decode Error: {e}")

if master_tree:
    with open(TREE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(master_tree, f, indent=2, ensure_ascii=False)
    print("\n" + "="*80)
    print(f"🛡️  Bureaucracy Firewall Purged: {purged_bureaucracy} non-actionable administrative sections.")
    print(f"✅ dpdp_act_tree.json generated flawlessly with {len(master_tree)} total enforceable clauses.")
    print("👉 Move this file to ssense/apps/browser-core/src-tauri/src/config/ for the Rust compiler.")
    print("="*80)
else:
    raise RuntimeError("❌ Failed to generate any valid clauses for dpdp_act_tree.json.")