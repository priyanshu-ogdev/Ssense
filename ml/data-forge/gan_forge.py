#!/usr/bin/env python3
"""
run_gan_forge.py – Ultimate Production GAN Forge

Optimized and verified for:
- Python 3.12 + Transformers 5.5.3 + vLLM 0.24.0
- Native host environments (Container-free execution safety)
- Robust error-trapped text and JSON processing
"""

import math
import os
import json
import glob
import random
import re
import sys
from tqdm import tqdm
from vllm import LLM, SamplingParams
# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0: EXTRACT LAW TEXT FROM PDFs
# ═══════════════════════════════════════════════════════════════════════════
PDF_ACT = "./DPDP_Act_2023.pdf"
PDF_RULES = "./DPDP_Rules_2025.pdf"
LAW_TEXT_PATH = "./dpdp_act_and_rules_2025.txt"

def build_law_text():
    if os.path.exists(LAW_TEXT_PATH):
        print(f"✅ Law text already exists at {LAW_TEXT_PATH}. Skipping.")
        return
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF required: pip install PyMuPDF")
        sys.exit(1)
        
    if not os.path.exists(PDF_ACT) or not os.path.exists(PDF_RULES):
        raise FileNotFoundError("Place both DPDP_Act_2023.pdf and DPDP_Rules_2025.pdf in this directory.")
        
    def extract(pdf_path):
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text("text") for page in doc)
        
    print("Extracting Act 2023...")
    act_text = extract(PDF_ACT)
    print("Extracting Rules 2025...")
    rules_text = extract(PDF_RULES)
    
    combined = f"=== DIGITAL PERSONAL DATA PROTECTION ACT 2023 ===\n\n{act_text}\n\n=== DIGITAL PERSONAL DATA PROTECTION RULES 2025 ===\n\n{rules_text}\n"
    with open(LAW_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"✅ Law merged -> {LAW_TEXT_PATH} (~{len(combined.split())} words).")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: CONFIGURATION & PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════
RAW_POLICIES_DIR = "./raw-policies"
INDIAN_SEEDS_DIR = "./indian-seeds"
SFT_OUTPUT_DIR = "./training-pairs/sft"
DPO_OUTPUT_DIR = "./training-pairs/dpo"

# Relative route from data-forge up to the shared schema directory
SCHEMA_PATH = "../../libs/contracts/schemas/dpdp_schema.json"

# Routes up to the adjacent models directory you just downloaded to
MODEL_PATH = "../models/Qwen2-72B-Instruct-FP8"

os.makedirs(SFT_OUTPUT_DIR, exist_ok=True)
os.makedirs(DPO_OUTPUT_DIR, exist_ok=True)

build_law_text()
with open(LAW_TEXT_PATH, "r", encoding="utf-8") as f:
    DPDP_LAW_TEXT = f.read()

if not os.path.exists(SCHEMA_PATH):
    raise FileNotFoundError(f"Missing required JSON contract schema at: {SCHEMA_PATH}")

def filter_english(text, threshold=0.3):
    devanagari_re = re.compile(r'[\u0900-\u097F]')
    lines = []
    for line in text.splitlines():
        if not line.strip() or len(line) == 0:
            lines.append(line)
            continue
        deva_chars = len(devanagari_re.findall(line))
        if (deva_chars / len(line)) < threshold:
            lines.append(line)
    return '\n'.join(lines)

indian_seeds = [open(f, "r", encoding="utf-8").read()
                for f in glob.glob(os.path.join(INDIAN_SEEDS_DIR, "*.txt"))]
if not indian_seeds:
    raise RuntimeError("No Indian seeds found. Run fetch_indian_seeds.py first.")

raw_policies = []
for f in glob.glob(os.path.join(RAW_POLICIES_DIR, "*.txt")):
    with open(f, "r", encoding="utf-8") as fh:
        text = fh.read()
    cleaned = filter_english(text)
    if len(cleaned) > 2000:
        raw_policies.append(cleaned)

if not raw_policies:
    raise RuntimeError("No valid raw policies found after English filtering verification.")

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    dpdp_schema = json.load(f)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: VLLM ENGINE & PROMPTS
# ═══════════════════════════════════════════════════════════════════════════
print("Initializing 72B FP8 vLLM Engine...")
llm = LLM(
    model=MODEL_PATH,
    quantization="fp8",
    tensor_parallel_size=2,
    max_model_len=16384,
    gpu_memory_utilization=0.95,
)

gen_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=8192)

# Fixed vLLM 0.24.x direct dictionary decoupling configuration
judge_params = SamplingParams(
    temperature=0.1,
    top_p=0.5,
    max_tokens=2048,
    guided_decoding={"json": dpdp_schema}
)

SYNTHESIZER_PROMPT = """[CONTEXT: THE LAW]
[LAW_INJECTION]

[CONTEXT: STYLISTIC ANCHOR]
[SEED_INJECTION]

[TASK]
Using the provided global policy as a base, synthesize a deceptive Indian privacy policy.
CRITICAL: Inject a deliberate violation of the DPDP Act specifically targeting:
[TARGET_VIOLATION]

You MUST plan your approach inside explicit <strategy>...</strategy> XML tags first, then output the final policy text.

[GLOBAL BASE POLICY]
[RAW_POLICY_INJECTION]"""

REFLEXION_EXPLICIT_PROMPT = """[CONTEXT: THE LAW]
[LAW_INJECTION]

[SYSTEM FEEDBACK]
The Regulatory Auditor missed your violation (score >= 90 or empty violations).
Audit details: [AUDIT_FEEDBACK]

[TASK]
Rewrite the policy to make the DPDP violation slightly more legally explicit, while keeping the deceptive corporate tone. 
Plan your approach inside explicit <strategy>...</strategy> XML tags first, then output the revised policy text.

[FAILED POLICY]
[FAILED_POLICY_INJECTION]"""

REFLEXION_SUBTLE_PROMPT = """[CONTEXT: THE LAW]
[LAW_INJECTION]

[SYSTEM FEEDBACK]
The Auditor caught your violation but scored it as overly obvious (subtlety_score < 70).
Make the violation more subtle and buried deep under complex legal jargon, while retaining the illegality.
Plan your approach inside explicit <strategy>...</strategy> XML tags first, then output the revised policy text.

[FAILED POLICY]
[FAILED_POLICY_INJECTION]"""

JUDGE_PROMPT = """You are a strict DPDP Regulatory Auditor.
[CONTEXT: THE LAW]
[LAW_INJECTION]

[TASK]
Analyze the following policy. Generate the JSON audit, including 'dpdp_trust_score', 'violations', and 'subtlety_score' (0-100, tracking how cleverly the violation is hidden).

[SYNTHESIZED POLICY]
[POLICY_INJECTION]"""

# ═══════════════════════════════════════════════════════════════════════════
# CORE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════
def extract_policy(text):
    """Aggressively split out Chain-of-Thought planning blocks to protect pipeline context."""
    if "</strategy>" in text:
        return text.split("</strategy>")[-1].strip()
    if "<strategy>" in text:
        return ""  # Trigger validation failure and auto-regeneration pass
    return text.strip()

TARGET_VIOLATIONS = [
    "Section 6: Consent architecture must be free, specific, and explicitly informed without bundling.",
    "Section 8: Data retention periods exceeded or deliberately tracking unmapped background metadata.",
    "Section 9: Processing personal data of children without explicit and verifiable parental authorization loops.",
    "Section 16: Obstruction of clear, multi-channel user grievance redressal or right-to-erase mechanisms."
]

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: VECTORIZED GAN LOOP
# ═══════════════════════════════════════════════════════════════════════════
def run_gan_forge():
    BATCH_SIZE = 50
    MAX_REFLEXION_STEPS = 3
    total_batches = math.ceil(len(raw_policies) / BATCH_SIZE)

    MAX_RAW_CHARS = 6000
    MAX_SEED_CHARS = 2000
    MAX_POLICY_CHARS_REFLEXION = 15000

    LAZY_AUDIT = {
        "dpdp_trust_score": 100, 
        "violations": [],
        "subtlety_score": 100, 
        "summary": "Universal fallback validation baseline."
    }

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(raw_policies))

        needed = []
        for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
            sft_file = os.path.join(SFT_OUTPUT_DIR, f"sft_{batch_idx:03d}_{local_idx:03d}.json")
            dpo_file = os.path.join(DPO_OUTPUT_DIR, f"dpo_{batch_idx:03d}_{local_idx:03d}.json")
            if not os.path.exists(sft_file) and not os.path.exists(dpo_file):
                needed.append((local_idx, global_idx))

        if not needed:
            print(f"⏭️ Batch {batch_idx} already complete.")
            continue

        print(f"🔥 Processing Batch {batch_idx}: Executing {len(needed)} items...")
        active_raws = [raw_policies[g] for _, g in needed]
        active_seeds = random.choices(indian_seeds, k=len(active_raws))
        local_ids = [l for l, _ in needed]
        targets = [random.choice(TARGET_VIOLATIONS) for _ in active_raws]

        # Initial Vectorized Generation Pass
        gen_messages = [
            [{"role": "system", "content": "Adversarial corporate counsel."},
             {"role": "user", "content": SYNTHESIZER_PROMPT
                .replace("[LAW_INJECTION]", DPDP_LAW_TEXT)
                .replace("[SEED_INJECTION]", seed[:MAX_SEED_CHARS])
                .replace("[TARGET_VIOLATION]", tgt)
                .replace("[RAW_POLICY_INJECTION]", raw[:MAX_RAW_CHARS])}]
            for seed, raw, tgt in zip(active_seeds, active_raws, targets)
        ]
        gen_out = llm.chat(messages=gen_messages, sampling_params=gen_params)
        current_policies = [extract_policy(o.outputs[0].text.strip()) for o in gen_out]

        completed = set()

        for step in range(MAX_REFLEXION_STEPS):
            remaining = [i for i in range(len(current_policies)) if i not in completed]
            if not remaining:
                break

            print(f"   ↳ Reflexion Iteration {step+1}/{MAX_REFLEXION_STEPS}: Syncing {len(remaining)} threads...")

            # Strict Structured Auditing via Decoupled guided_decoding
            judge_msgs = [
                [{"role": "system", "content": "Strict DPDP Auditor."},
                 {"role": "user", "content": JUDGE_PROMPT
                    .replace("[LAW_INJECTION]", DPDP_LAW_TEXT)
                    .replace("[POLICY_INJECTION]", current_policies[i])}]
                for i in remaining
            ]
            audit_outputs = llm.chat(messages=judge_msgs, sampling_params=judge_params)

            parsed = {}
            for idx, out in zip(remaining, audit_outputs):
                try:
                    parsed[idx] = json.loads(out.outputs[0].text.strip())
                except (json.JSONDecodeError, AttributeError, KeyError):
                    parsed[idx] = {"dpdp_trust_score": 100, "violations": [], "subtlety_score": 100}

            explicit_heal, subtle_heal = [], []
            explicit_idx, subtle_idx = [], []

            for i in remaining:
                audit = parsed[i]
                score = audit.get("dpdp_trust_score", 0)
                viols = audit.get("violations", [])
                subtlety = audit.get("subtlety_score", 100)
                local_id = local_ids[i]

                caught = (score < 90 and len(viols) > 0)

                if caught:
                    if step == 0 and subtlety < 70:
                        subtle_heal.append([
                            {"role": "system", "content": "Adversarial corporate counsel."},
                            {"role": "user", "content": REFLEXION_SUBTLE_PROMPT
                                .replace("[LAW_INJECTION]", DPDP_LAW_TEXT)
                                .replace("[FAILED_POLICY_INJECTION]", current_policies[i][:MAX_POLICY_CHARS_REFLEXION])}
                        ])
                        subtle_idx.append(i)
                        continue

                    if step == 0:
                        sft = {"messages": [
                            {"role": "system", "content": "Strict DPDP Auditor."},
                            {"role": "user", "content": f"[CONTEXT: THE LAW]\n{DPDP_LAW_TEXT}\n\n[TASK]\nAnalyze:\n{current_policies[i]}"},
                            {"role": "assistant", "content": json.dumps(audit)}
                        ]}
                        with open(os.path.join(SFT_OUTPUT_DIR, f"sft_{batch_idx:03d}_{local_id:03d}.json"), "w", encoding="utf-8") as f:
                            json.dump(sft, f, ensure_ascii=False)
                    else:
                        dpo = {
                            "chosen": [
                                {"role": "system", "content": "Strict DPDP Auditor."},
                                {"role": "user", "content": f"[CONTEXT: THE LAW]\n{DPDP_LAW_TEXT}\n\n[TASK]\nAnalyze:\n{current_policies[i]}"},
                                {"role": "assistant", "content": json.dumps(audit)}
                            ],
                            "rejected": [
                                {"role": "system", "content": "Strict DPDP Auditor."},
                                {"role": "user", "content": f"[CONTEXT: THE LAW]\n{DPDP_LAW_TEXT}\n\n[TASK]\nAnalyze:\n{current_policies[i]}"},
                                {"role": "assistant", "content": json.dumps(LAZY_AUDIT)}
                            ]
                        }
                        with open(os.path.join(DPO_OUTPUT_DIR, f"dpo_{batch_idx:03d}_{local_id:03d}.json"), "w", encoding="utf-8") as f:
                            json.dump(dpo, f, ensure_ascii=False)
                    completed.add(i)

                else:  # Missed or bypassed evaluation state
                    if step < MAX_REFLEXION_STEPS - 1:
                        explicit_heal.append([
                            {"role": "system", "content": "Adversarial corporate counsel."},
                            {"role": "user", "content": REFLEXION_EXPLICIT_PROMPT
                                .replace("[LAW_INJECTION]", DPDP_LAW_TEXT)
                                .replace("[AUDIT_FEEDBACK]", json.dumps(audit))
                                .replace("[FAILED_POLICY_INJECTION]", current_policies[i][:MAX_POLICY_CHARS_REFLEXION])}
                        ])
                        explicit_idx.append(i)
                    else:
                        sft = {"messages": [
                            {"role": "system", "content": "Strict DPDP Auditor."},
                            {"role": "user", "content": f"[CONTEXT: THE LAW]\n{DPDP_LAW_TEXT}\n\n[TASK]\nAnalyze:\n{current_policies[i]}"},
                            {"role": "assistant", "content": json.dumps(audit)}
                        ]}
                        with open(os.path.join(SFT_OUTPUT_DIR, f"sft_{batch_idx:03d}_{local_id:03d}.json"), "w", encoding="utf-8") as f:
                            json.dump(sft, f, ensure_ascii=False)
                        completed.add(i)

            # Vectorized Annealed Re-Generation
            heal_temp = max(0.6, 0.8 - 0.1 * step)
            heal_params = SamplingParams(temperature=heal_temp, top_p=0.9, max_tokens=8192)

            if explicit_heal:
                out_explicit = llm.chat(messages=explicit_heal, sampling_params=heal_params)
                for idx, o in zip(explicit_idx, out_explicit):
                    current_policies[idx] = extract_policy(o.outputs[0].text.strip())

            if subtle_heal:
                out_subtle = llm.chat(messages=subtle_heal, sampling_params=heal_params)
                for idx, o in zip(subtle_idx, out_subtle):
                    current_policies[idx] = extract_policy(o.outputs[0].text.strip())

    print("✅ GAN forge complete. Datasets compiled under utf-8 targets.")

if __name__ == "__main__":
    run_gan_forge()