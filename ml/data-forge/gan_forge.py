#!/usr/bin/env python3
"""
run_gan_forge.py – Full GAN data‑generation pipeline

1. Parses DPDP Act & Rules PDFs → single .txt (if missing)
2. Runs the Asymmetric GAN Forge using the 72B vLLM engine
   to produce SFT + DPO pairs in Axolotl‑ready format.
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
    """Extract text from PDFs and merge into LAW_TEXT_PATH."""
    if os.path.exists(LAW_TEXT_PATH):
        print(f"✅ Law text already exists at {LAW_TEXT_PATH}. Skipping PDF extraction.")
        return

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF is required. Install it: pip install PyMuPDF")
        sys.exit(1)

    if not os.path.exists(PDF_ACT) or not os.path.exists(PDF_RULES):
        raise FileNotFoundError(
            "Place DPDP_Act_2023.pdf and DPDP_Rules_2025.pdf in this directory."
        )

    def extract_text(pdf_path):
        doc = fitz.open(pdf_path)
        blocks = []
        for page in doc:
            blocks.append(page.get_text("text"))
        return "\n".join(blocks)

    print("Extracting DPDP Act 2023...")
    act_text = extract_text(PDF_ACT)
    print("Extracting DPDP Rules 2025...")
    rules_text = extract_text(PDF_RULES)

    combined = (
        "=== DIGITAL PERSONAL DATA PROTECTION ACT 2023 ===\n\n"
        f"{act_text}\n\n"
        "=== DIGITAL PERSONAL DATA PROTECTION RULES 2025 ===\n\n"
        f"{rules_text}\n"
    )

    with open(LAW_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"✅ Law merged into {LAW_TEXT_PATH} (~{len(combined.split())} words).")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: GAN FORGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
RAW_POLICIES_DIR = "./raw-policies"
INDIAN_SEEDS_DIR = "./indian-seeds"
SFT_OUTPUT_DIR = "./training-pairs/sft"
DPO_OUTPUT_DIR = "./training-pairs/dpo"
SCHEMA_PATH = "../../libs/contracts/schemas/dpdp_schema.json"  # adjust as needed
MODEL_PATH = "/path/to/Qwen2-72B-Instruct-FP8"

os.makedirs(SFT_OUTPUT_DIR, exist_ok=True)
os.makedirs(DPO_OUTPUT_DIR, exist_ok=True)

# Load law text (guaranteed to exist after phase 0)
with open(LAW_TEXT_PATH, "r", encoding="utf-8") as f:
    DPDP_LAW_TEXT = f.read()

# ── Language filter for raw policies ──
def filter_english(text, threshold=0.3):
    """
    Remove lines that are predominantly Devanagari (Hindi).
    Returns the cleaned English-only text.
    """
    devanagari_re = re.compile(r'[\u0900-\u097F]')
    lines = []
    for line in text.splitlines():
        # Keep empty lines (preserve paragraph breaks)
        if not line.strip():
            lines.append(line)
            continue
        total_chars = len(line)
        deva_chars = len(devanagari_re.findall(line))
        if (deva_chars / total_chars) < threshold:
            lines.append(line)
    return '\n'.join(lines)

# Load Indian seeds (assumed English – no filtering)
indian_seeds = [open(f, "r", encoding="utf-8").read()
                for f in glob.glob(os.path.join(INDIAN_SEEDS_DIR, "*.txt"))]
if not indian_seeds:
    raise RuntimeError("No Indian seeds found. Run fetch_indian_seeds.py first.")

# Load raw policies, filtering out non-English lines and too-small documents
raw_policies = []
for f in glob.glob(os.path.join(RAW_POLICIES_DIR, "*.txt")):
    with open(f, "r", encoding="utf-8") as fh:
        raw_text = fh.read()
    cleaned = filter_english(raw_text)
    # Keep only if the remaining English text is substantial
    if len(cleaned) > 2000:
        raw_policies.append(cleaned)

if not raw_policies:
    raise RuntimeError("No valid raw policies found (after English filtering). Check your ./raw-policies folder.")

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    dpdp_schema = json.load(f)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: VLLM ENGINE & PROMPTS (Chat Template Ready)
# ═══════════════════════════════════════════════════════════════════════════
print("Initializing 72B FP8 vLLM Engine with extended context...")
llm = LLM(
    model=MODEL_PATH,
    quantization="fp8",
    tensor_parallel_size=2,
    max_model_len=16384,               # 16k to comfortably fit law + policy
    gpu_memory_utilization=0.95,
)

# Generator params – longer output to avoid mid-sentence truncation
gen_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=8192)

# Judge params – strict, with native guided_json
judge_params = SamplingParams(
    temperature=0.1,
    top_p=0.5,
    max_tokens=2048,
    guided_json=dpdp_schema,
)

# Prompts
SYNTHESIZER_PROMPT = """[CONTEXT: THE LAW]
[LAW_INJECTION]

[CONTEXT: STYLISTIC ANCHOR]
[SEED_INJECTION]

[TASK]
Using the provided global policy as a base, synthesize a deceptive Indian privacy policy. 
CRITICAL: You must secretly inject a deliberate violation of the DPDP Act/Rules deep inside the text.
Output ONLY the synthesized policy text.

[GLOBAL BASE POLICY]
[RAW_POLICY_INJECTION]"""

REFLEXION_PROMPT = """[CONTEXT: THE LAW]
[LAW_INJECTION]

[SYSTEM FEEDBACK]
The Regulatory Auditor gave the following audit:
[AUDIT_FEEDBACK]

As you can see, the DPDP violation you attempted to inject was completely missed (score >= 90 or no violations found).
[TASK]
Rewrite the policy. Make the DPDP violation slightly more legally explicit so a strict auditor will catch it, but maintain the deceptive corporate tone.
Output ONLY the revised policy text.

[FAILED POLICY]
[FAILED_POLICY_INJECTION]"""

JUDGE_PROMPT = """You are a strict DPDP Regulatory Auditor.
[CONTEXT: THE LAW]
[LAW_INJECTION]

[TASK]
Analyze the following synthesized privacy policy. Generate the strict JSON compliance audit identifying the hidden violation.

[SYNTHESIZED POLICY]
[POLICY_INJECTION]"""

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: ASYMMETRIC GAN LOOP (Chat‑based, Resilient, Axolotl‑Ready)
# ═══════════════════════════════════════════════════════════════════════════
def run_gan_forge():
    BATCH_SIZE = 50
    total_batches = math.ceil(len(raw_policies) / BATCH_SIZE)

    # Resume logic: skip already‑completed batches
    existing_files = glob.glob(os.path.join(SFT_OUTPUT_DIR, "sft_*.json"))
    start_batch = 0
    if existing_files:
        highest = max(int(os.path.basename(f).split('_')[1]) for f in existing_files)
        start_batch = highest + 1
        print(f"Resuming GAN Forge from batch {start_batch}...")

    # Safe truncation limits (tuned for 16k context)
    MAX_RAW_CHARS = 6000
    MAX_SEED_CHARS = 2000
    MAX_POLICY_CHARS_REFLEXION = 15000  # fully fits inside 16k window

    # Hard‑coded “lazy” audit for DPO negative baseline
    l_audit = {
        "dpdp_trust_score": 100,
        "violations": [],
        "summary": "The policy appears to be fully compliant with the DPDP Act 2023."
    }

    for batch_idx in tqdm(range(start_batch, total_batches), desc="GAN Forging Batches"):
        batch_raw = raw_policies[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        batch_seeds = random.choices(indian_seeds, k=len(batch_raw))

        # ── Generator (chat API) ──
        gen_messages = [
            [
                {"role": "system", "content": "You are an adversarial corporate legal counsel."},
                {"role": "user", "content": SYNTHESIZER_PROMPT
                    .replace("[LAW_INJECTION]", DPDP_LAW_TEXT)
                    .replace("[SEED_INJECTION]", seed[:MAX_SEED_CHARS])
                    .replace("[RAW_POLICY_INJECTION]", raw[:MAX_RAW_CHARS])
                }
            ]
            for seed, raw in zip(batch_seeds, batch_raw)
        ]
        gen_outputs = llm.chat(messages=gen_messages, sampling_params=gen_params)
        policies = [out.outputs[0].text.strip() for out in gen_outputs]

        # ── Strict Judge (chat API) ──
        judge_messages = [
            [
                {"role": "system", "content": "You are a strict DPDP Regulatory Auditor."},
                {"role": "user", "content": JUDGE_PROMPT
                    .replace("[LAW_INJECTION]", DPDP_LAW_TEXT)
                    .replace("[POLICY_INJECTION]", p)
                }
            ]
            for p in policies
        ]
        audit_outputs = llm.chat(messages=judge_messages, sampling_params=judge_params)
        audits = []
        for out in audit_outputs:
            try:
                audits.append(json.loads(out.outputs[0].text))
            except json.JSONDecodeError:
                audits.append({"dpdp_trust_score": 100, "violations": []})

        # ── Reflexion Routing ──
        reflexion_messages = []
        reflexion_indices = []

        for i, (policy, audit) in enumerate(zip(policies, audits)):
            if audit.get("dpdp_trust_score", 0) >= 90 or len(audit.get("violations", [])) == 0:
                audit_feedback = json.dumps(audit, indent=2)
                reflexion_messages.append([
                    {"role": "system", "content": "You are an adversarial corporate legal counsel."},
                    {"role": "user", "content": REFLEXION_PROMPT
                        .replace("[LAW_INJECTION]", DPDP_LAW_TEXT)
                        .replace("[AUDIT_FEEDBACK]", audit_feedback)
                        .replace("[FAILED_POLICY_INJECTION]", policy[:MAX_POLICY_CHARS_REFLEXION])
                    }
                ])
                reflexion_indices.append(i)
            else:
                # Save SFT pair in Axolotl‑standard messages format
                pair = {
                    "messages": [
                        {"role": "system", "content": "You are a strict DPDP Regulatory Auditor."},
                        {"role": "user", "content": f"[CONTEXT: THE LAW]\n{DPDP_LAW_TEXT}\n\n[TASK]\nAnalyze the following privacy policy:\n{policy}"},
                        {"role": "assistant", "content": json.dumps(audit)}
                    ]
                }
                fname = os.path.join(SFT_OUTPUT_DIR, f"sft_{batch_idx:03d}_{i:03d}.json")
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(pair, f)

        # ── Healing Loop & DPO ──
        if reflexion_messages:
            ref_outputs = llm.chat(messages=reflexion_messages, sampling_params=gen_params)
            fixed_policies = [out.outputs[0].text.strip() for out in ref_outputs]

            # Strict re‑audit (chat)
            re_judge_messages = [
                [
                    {"role": "system", "content": "You are a strict DPDP Regulatory Auditor."},
                    {"role": "user", "content": JUDGE_PROMPT
                        .replace("[LAW_INJECTION]", DPDP_LAW_TEXT)
                        .replace("[POLICY_INJECTION]", p)
                    }
                ]
                for p in fixed_policies
            ]
            re_audit_outputs = llm.chat(messages=re_judge_messages, sampling_params=judge_params)
            fixed_audits = [json.loads(out.outputs[0].text) for out in re_audit_outputs]

            # Construct DPO pairs (Axolotl format, hard‑coded rejected baseline)
            for idx, f_audit, fixed_policy in zip(reflexion_indices, fixed_audits, fixed_policies):
                if f_audit.get("dpdp_trust_score", 0) < 90:
                    dpo_pair = {
                        "chosen": [
                            {"role": "system", "content": "You are a strict DPDP Regulatory Auditor."},
                            {"role": "user", "content": f"[CONTEXT: THE LAW]\n{DPDP_LAW_TEXT}\n\n[TASK]\nAnalyze this policy:\n{fixed_policy}"},
                            {"role": "assistant", "content": json.dumps(f_audit)}
                        ],
                        "rejected": [
                            {"role": "system", "content": "You are a strict DPDP Regulatory Auditor."},
                            {"role": "user", "content": f"[CONTEXT: THE LAW]\n{DPDP_LAW_TEXT}\n\n[TASK]\nAnalyze this policy:\n{fixed_policy}"},
                            {"role": "assistant", "content": json.dumps(l_audit)}
                        ]
                    }
                    fname = os.path.join(DPO_OUTPUT_DIR, f"dpo_{batch_idx:03d}_{idx:03d}.json")
                    with open(fname, "w", encoding="utf-8") as f:
                        json.dump(dpo_pair, f)

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    build_law_text()
    print("Initiating Ssense Asymmetric GAN Forge...")
    run_gan_forge()
    print("✅ Forging complete. SFT and DPO pairs saved in Axolotl‑ready format.")