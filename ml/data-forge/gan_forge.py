#!/usr/bin/env python3
"""
run_gan_forge.py – Ultimate Production GAN Forge (Final Sealed Version)
"""

import math
import os
import json
import glob
import random
import re
import sys
import itertools
from collections import defaultdict
from difflib import SequenceMatcher
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0: MODULAR IMPORTS & PROMPT LOADING
# ═══════════════════════════════════════════════════════════════════════════
from prompts.target_violations import TARGET_VIOLATIONS, ATOMIC_STATUTES, SEMANTIC_KEYWORD_MAP
from prompts.edge_case_templates import EDGE_CASE_TEMPLATES

def load_prompt(filename: str) -> str:
    with open(os.path.join("prompts", filename), "r", encoding="utf-8") as f:
        return f.read()

SYNTHESIZER_PROMPT = load_prompt("synthesizer_prompt.txt")
JUDGE_PROMPT = load_prompt("judge_prompt.txt")
REFLEXION_EXPLICIT_PROMPT = load_prompt("reflexion_explicit_prompt.txt")
REFLEXION_SUBTLE_PROMPT = load_prompt("reflexion_subtle_prompt.txt")
CHATBOT_QA_SFT_PROMPT = load_prompt("chatbot_qa_prompt.txt")
CHATBOT_QA_DPO_PROMPT = load_prompt("chatbot_qa_dpo_prompt.txt")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: CONFIGURATION & PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════
PDF_ACT = "./DPDP_Act_2023.pdf"
PDF_RULES = "./DPDP_Rules_2025.pdf"
LAW_TEXT_PATH = "./dpdp_act_and_rules_2025.txt"

RAW_POLICIES_DIR = "./raw-policies"
INDIAN_SEEDS_DIR = "./indian-seeds"
SFT_OUTPUT_DIR = "./training-pairs/sft"
DPO_OUTPUT_DIR = "./training-pairs/dpo"
CHATBOT_SFT_DIR = "./training-pairs/chatbot-sft"
CHATBOT_DPO_DIR = "./training-pairs/chatbot-dpo"

SCHEMA_PATH = "../../libs/contracts/schemas/dpdp_schema.json"
MODEL_PATH = "../models/Qwen2-72B-Instruct-FP8"

TARGET_AUDIT_POLICIES = 12000
TARGET_CHATBOT_PAIRS = 3000
BATCH_SIZE = 50
MAX_REFLEXION_STEPS = 3

for d in [SFT_OUTPUT_DIR, DPO_OUTPUT_DIR, CHATBOT_SFT_DIR, CHATBOT_DPO_DIR]:
    os.makedirs(d, exist_ok=True)

def build_law_text():
    if os.path.exists(LAW_TEXT_PATH): return
    try: import fitz
    except ImportError: sys.exit("PyMuPDF required: pip install PyMuPDF")
    if not os.path.exists(PDF_ACT) or not os.path.exists(PDF_RULES):
        raise FileNotFoundError("Place both DPDP PDFs in this directory.")
    act_text = "\n".join(page.get_text("text") for page in fitz.open(PDF_ACT))
    rules_text = "\n".join(page.get_text("text") for page in fitz.open(PDF_RULES))
    combined = f"=== DIGITAL PERSONAL DATA PROTECTION ACT 2023 ===\n\n{act_text}\n\n=== DIGITAL PERSONAL DATA PROTECTION RULES 2025 ===\n\n{rules_text}\n"
    with open(LAW_TEXT_PATH, "w", encoding="utf-8") as f: f.write(combined)

build_law_text()
with open(LAW_TEXT_PATH, "r", encoding="utf-8") as f:
    DPDP_LAW_TEXT = f.read()

if not os.path.exists(SCHEMA_PATH):
    raise FileNotFoundError(f"Missing schema: {SCHEMA_PATH}")

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

def filter_english(text, threshold=0.05):
    lines = []
    for line in text.splitlines():
        if not line.strip(): continue
        deva_chars = len(DEVANAGARI_RE.findall(line))
        if (deva_chars / len(line)) < threshold: lines.append(line)
    return '\n'.join(lines)

DPDP_LAW_TEXT = filter_english(DPDP_LAW_TEXT, threshold=0.05)

indian_seeds = [open(f, "r", encoding="utf-8").read() for f in glob.glob(os.path.join(INDIAN_SEEDS_DIR, "*.txt"))]
if not indian_seeds: raise RuntimeError("No Indian seeds found.")

raw_policies = []
for f in glob.glob(os.path.join(RAW_POLICIES_DIR, "*.txt")):
    with open(f, "r", encoding="utf-8") as fh:
        cleaned = filter_english(fh.read())
        if len(cleaned) > 2000: raw_policies.append(cleaned)
if not raw_policies: raise RuntimeError("No valid raw policies found.")

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    dpdp_schema = json.load(f)

SUBTLETY_LEVELS = {
    "blatant": "Make the violation obvious and easy to detect. Use clear, unambiguous language that directly contradicts DPDP requirements.",
    "moderate": "Make the violation moderately subtle. Bury it within corporate jargon but keep it detectable by a careful reader.",
    "subtle": "Make the violation highly subtle. Hide it within complex legal language, exception clauses, or technical terminology.",
    "masterful": "Make the violation masterfully hidden. Embed it within seemingly compliant language, use legal loopholes, or frame it as necessary business practice."
}

INDUSTRIES = {
    "tech": "technology company providing SaaS products and digital services",
    "finance": "banking and financial services institution (NBFC, insurance, fintech)",
    "healthcare": "healthcare provider handling sensitive medical and health data",
    "ecommerce": "e-commerce platform processing transactions and customer data",
    "education": "educational institution managing student and academic data",
    "telecom": "telecommunications service provider handling call records and location data"
}

CHATBOT_PERSONAS = [
    "An angry citizen whose data was leaked in a recent breach.",
    "A Data Protection Officer (DPO) at a mid-sized Indian fintech.",
    "A law student researching Indian privacy jurisprudence.",
    "A corporate compliance lawyer advising a multinational client.",
    "A journalist trying to file an RTI request for a politician's data.",
    "A parent concerned about their child's data being tracked by an EdTech app.",
    "A small business owner trying to understand SDF thresholds."
]

CHATBOT_SCENARIOS = [
    "The exact penalty schedule and maximum fines for data breaches (Section 33 & Schedule).",
    "The step-by-step process of filing a complaint with the Data Protection Board.",
    "The composition, appointment, and removal criteria of Board members (Section 18-20).",
    "The powers of the Board to summon evidence and conduct inquiries (Section 21).",
    "The appellate process: How to appeal a Board decision to the TDSAT (Section 29).",
    "The legal definition and distinction between Data Principal and Data Fiduciary (Section 2).",
    "The scope of the Act: How it applies to offline data that is later digitized (Section 3).",
    "Government and law enforcement exemptions for investigating offenses (Section 17).",
    "How Section 44 of the DPDP Act overrides the RTI Act 2005 regarding personal information.",
    "The legal obligations and liabilities of Data Processors acting on behalf of Fiduciaries (Section 8(8)).",
    "The specific duties of Significant Data Fiduciaries (SDFs) regarding DPIAs and audits (Section 10).",
    "The rights of Data Principals to nominate a representative in case of death or incapacity (Section 15).",
    "The mechanics and interoperability requirements of Consent Managers (Rule 4).",
    "The exact timelines and mechanisms for data breach notification to the Board and users (Rule 7)."
]
SUBTLE_TECHNIQUE_MAP = {
    "consent": "Implied Consent Trap: Use phrases like 'continued use constitutes consent', 'presumptive consent', or 'integral to our ecosystem services'.",
    "retention": "Future-Proofing Loophole: Attribute indefinite retention to 'legacy system dependencies', 'statutory audit requirements', or 'business continuity protocols'.",
    "children": "Ambiguous Age Verification: State that you 'do not knowingly collect data from children' without specifying a verifiable consent mechanism, or rely on 'self-certification'.",
    "grievance": "Procedural Friction: Require 'notarized documentation', 'registered post', or a '60-day internal triage period' to process any rights requests.",
    "notice": "Vagueness Shield: Replace concrete disclosures with aspirational fluff like 'We strive to maintain a trusted environment' or 'Privacy is paramount to our goals'.",
    "security": "Delegated Liability: Claim you use 'industry-standard measures' but explicitly disclaim liability for third-party vendor breaches.",
    "sdf": "Trade Secret Exemption: Claim your core algorithms are 'proprietary trade secrets' and therefore exempt from external algorithmic auditing or DPIA disclosures.",
    "crossborder": "Global Infrastructure Veil: State that data is processed in 'jurisdictions that meet international best practices' without naming specific countries or safeguards.",
    "legitimate_uses_abuse": "Legitimate Use Overreach: Claim that voluntarily providing data constitutes 'deemed consent' for secondary commercial marketing.",
    "processor_accountability": "Vendor Shield: Explicitly state that the Data Fiduciary is 'not responsible for third-party breaches' or 'vendor is solely liable'.",
    "breach_notification": "Internal Triage Delay: State that notification will occur 'within 72 hours of the conclusive completion of our internal forensics triage'.",
    "consent_manager": "Cryptographic Blockade: Refuse Consent Manager integration by citing 'cryptographic integrity' and forcing users to use the 'native application dashboard'.",
    "language_accessibility": "Legal Precision Shield: State that 'legally binding notices are maintained exclusively in English to ensure absolute legal precision'.",
    "algorithmic_profiling": "Black Box Exemption: Hide behind 'proprietary trade secrets' to avoid disclosing automated decision-making logic.",
    "rights_implementation": "Nominee Invalidation: State that 'accounts and data rights are strictly non-transferable' and refuse to recognize post-mortem nominees."
}

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: DYNAMIC CONTEXT & VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════
law_cache = {}
def extract_relevant_law(law_text, target_violation):
    if target_violation in law_cache: return law_cache[target_violation]
    keywords = []
    target_lower = target_violation.lower()
    
    if "section 6" in target_lower or "consent" in target_lower or "rule 5" in target_lower:
        keywords = ["Section 6", "Consent", "Notice", "Bundling", "Rule 5", "verifiable"]
    elif "section 8" in target_lower or "retention" in target_lower or "rule 8" in target_lower:
        keywords = ["Section 8", "Retention", "Erase", "Storage", "Metadata", "Rule 8", "time period"]
    elif "section 9" in target_lower or "children" in target_lower or "rule 10" in target_lower:
        keywords = ["Section 9", "Children", "Parental", "Verifiable", "Rule 10", "guardian"]
    elif "section 11" in target_lower or "section 16" in target_lower or "grievance" in target_lower or "rule 12" in target_lower:
        keywords = ["Section 11", "Section 16", "Grievance", "Redressal", "Appeal", "Rule 12", "Rule 13", "Rule 14"]
    elif "section 5" in target_lower or "notice" in target_lower or "rule 3" in target_lower or "rule 4" in target_lower:
        keywords = ["Section 5", "Notice", "Rule 3", "Rule 4", "clear and plain"]
    elif "security" in target_lower or "rule 7" in target_lower:
        keywords = ["Section 8", "security safeguards", "Rule 7", "technical", "organizational"]
    elif "section 10" in target_lower or "sdf" in target_lower or "rule 13" in target_lower:
        keywords = ["Section 10", "Significant Data Fiduciary", "DPO", "Data Protection Impact", "Rule 13"]
    elif "section 16" in target_lower or "cross-border" in target_lower or "rule 15" in target_lower:
        keywords = ["Section 16", "transfer", "outside the territory", "Rule 15", "foreign"]
    elif "section 33" in target_lower or "penalty" in target_lower or "schedule" in target_lower:
        keywords = ["Section 33", "Penalty", "Schedule", "fine", "crore"]
    elif "section 29" in target_lower or "tdsat" in target_lower or "appeal" in target_lower:
        keywords = ["Section 29", "TDSAT", "Appellate", "Tribunal"]
    elif "section 17" in target_lower or "exemption" in target_lower:
        keywords = ["Section 17", "Exemption", "State", "security of India"]
    elif "section 44" in target_lower or "rti" in target_lower:
        keywords = ["Section 44", "RTI", "Right to Information"]
        
    relevant_chunks = []
    for p in law_text.split('\n\n'):
        if any(kw.lower() in p.lower() for kw in keywords): relevant_chunks.append(p.strip())
            
    result = "\n\n".join(relevant_chunks[:15]) if relevant_chunks else law_text[:8000]
    law_cache[target_violation] = result
    return result

normalized_policy_cache = {}
def get_normalized_policy(policy_text):
    policy_hash = hash(policy_text)
    if policy_hash not in normalized_policy_cache:
        normalized_policy_cache[policy_hash] = ' '.join(policy_text.lower().split())
    return normalized_policy_cache[policy_hash]

def validate_audit_quality(audit: dict, policy_text: str) -> tuple:
    violations = audit.get("violations", [])
    if not violations: return True, ""
    normalized_policy = get_normalized_policy(policy_text)
    
    for v in violations:
        vtype = v.get("violation_type", "")
        evidence = v.get("evidence_quote", "")
        statute = v.get("statute_reference", "")
        statute_lower = statute.lower()
        
        # 1. Fictional/Foreign law check (avoiding DPDP false positive)
        if "digital personal data protection" not in statute_lower and "dpdp" not in statute_lower:
            if any(marker in statute_lower for marker in ["20xx", "gdpr", "ccpa", "privacy rights act", "hipaa", "lgpd", "pdpa"]):
                return False, f"Fictional/Foreign law: {statute}"
        
        # 2. Generic statute check (using imported ATOMIC_STATUTES)
        if re.match(r'^(Section|Rule)\s+\d+$', statute, re.IGNORECASE):
            if statute_lower not in ATOMIC_STATUTES:
                return False, f"Generic statute: {statute}"
        
        if evidence:
            # 3. Commentary trap
            if any(marker in evidence.lower() for marker in ["the policy does not", "no mention of", "does not explicitly", "the policy lacks", "does not specify", "does not mention", "[no mention"]):
                return False, f"Commentary in evidence: {evidence[:40]}..."
            
            normalized_evidence = ' '.join(evidence.lower().split())
            if normalized_evidence not in normalized_policy:
                if not _fuzzy_match(normalized_evidence, normalized_policy, threshold=0.75):
                    return False, f"Hallucinated evidence: {evidence[:50]}..."
        
        # 4. Semantic map (using imported SEMANTIC_KEYWORD_MAP, OR logic)
        if vtype in SEMANTIC_KEYWORD_MAP:
            evidence_lower = evidence.lower() if evidence else ""
            kw = SEMANTIC_KEYWORD_MAP[vtype]
            primary_match = any(k in evidence_lower for k in kw["primary"])
            secondary_count = sum(1 for k in kw["secondary"] if k in evidence_lower)
            if not primary_match and secondary_count < 2:
                return False, f"Semantic mismatch: {evidence[:40]}..."
    
    return True, ""

def _fuzzy_match(quote: str, text: str, threshold: float = 0.85) -> bool:
    if not quote or not text: return False
    matcher = SequenceMatcher(None, quote, text, autojunk=False)
    return matcher.ratio() >= threshold

def extract_policy(text):
    if "</strategy>" in text: policy_text = text.split("</strategy>")[-1].strip()
    elif "<strategy>" in text: return ""
    else: policy_text = text.strip()
    
    policy_text = re.sub(r'<!--.*?-->', '', policy_text, flags=re.DOTALL)
    policy_text = re.sub(r'<[^>]*(?:violation|clause|note|instruction|deceptive|strategy|approach)[^>]*>', '', policy_text, flags=re.IGNORECASE)
    policy_text = re.sub(r'\[.*?(?:violation|note|disclaimer|fictional|deceptive|instruction|omission|clause|educational|real-world|strategy|revised|policy|end)[^\]]*\]', '', policy_text, flags=re.IGNORECASE)
    deceptive_labels = r'^\s*\d+\.\s*(?:Concealment|Complexity|Delay|Inaccessibility|Documentation|Misdirection|Deception|Obfuscation)[^\n]*\n'
    policy_text = re.sub(deceptive_labels, '', policy_text, flags=re.IGNORECASE | re.MULTILINE)
    policy_text = re.split(r'(?:END OF POLICY|End of Policy|\[END OF)', policy_text, flags=re.IGNORECASE)[0]
    policy_text = re.sub(r'^\s*\[(?:DECEPTIVE|REVISED|STRATEGY)[^\]]*\]\s*', '', policy_text, flags=re.IGNORECASE)
    policy_text = re.sub(r'\n\s*\n\s*\n', '\n\n', policy_text)
    return policy_text.strip()

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: VLLM ENGINE & SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════
print("Initializing 72B FP8 vLLM Engine...")
llm = LLM(
    model=MODEL_PATH, quantization="fp8", tensor_parallel_size=1,
    max_model_len=16384, gpu_memory_utilization=0.80, max_num_seqs=50,
    kv_cache_dtype="fp8", enable_prefix_caching=True, enable_chunked_prefill=True
)

gen_params = SamplingParams(temperature=1.05, top_p=0.95, max_tokens=6144)
judge_params = SamplingParams(temperature=0.1, top_p=0.5, max_tokens=2048, structured_outputs=StructuredOutputsParams(json=dpdp_schema))

CHATBOT_SFT_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant"]},
                    "content": {"type": "string", "minLength": 20}
                },
                "required": ["role", "content"],
                "additionalProperties": False
            },
            "minItems": 4, "maxItems": 8
        }
    },
    "required": ["messages"],
    "additionalProperties": False
}

CHATBOT_DPO_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "array", "items": {"type": "object", "properties": {"role": {"type": "string", "enum": ["user"]}, "content": {"type": "string"}}}},
        "chosen": {"type": "array", "items": {"type": "object", "properties": {"role": {"type": "string", "enum": ["assistant"]}, "content": {"type": "string"}}}},
        "rejected": {"type": "array", "items": {"type": "object", "properties": {"role": {"type": "string", "enum": ["assistant"]}, "content": {"type": "string"}}}}
    },
    "required": ["prompt", "chosen", "rejected"]
}

chatbot_sft_params = SamplingParams(temperature=0.85, top_p=0.95, max_tokens=4096, structured_outputs=StructuredOutputsParams(json=CHATBOT_SFT_SCHEMA))
chatbot_dpo_params = SamplingParams(temperature=0.85, top_p=0.95, max_tokens=4096, structured_outputs=StructuredOutputsParams(json=CHATBOT_DPO_SCHEMA))

def build_audit_matrix():
    matrix = []
    combinations = list(itertools.product(TARGET_VIOLATIONS.keys(), SUBTLETY_LEVELS.keys(), INDUSTRIES.keys()))
    for i in range(TARGET_AUDIT_POLICIES):
        category, subtlety, industry = combinations[i % len(combinations)]
        matrix.append({
            "index": i, "base_policy": random.choice(raw_policies), "seed": random.choice(indian_seeds),
            "target_violation": random.choice(TARGET_VIOLATIONS[category]), "target_category": category,
            "subtlety_level": subtlety, "industry": industry,
            "edge_template": random.choice(EDGE_CASE_TEMPLATES) if random.random() < 0.20 else None
        })
    random.shuffle(matrix)
    matrix.sort(key=lambda x: x["target_violation"])
    return matrix

def build_chatbot_matrix():
    matrix = []
    for i in range(TARGET_CHATBOT_PAIRS):
        scenario = random.choice(CHATBOT_SCENARIOS)
        matrix.append({
            "index": i, "persona": random.choice(CHATBOT_PERSONAS),
            "scenario": scenario, "law_chunk": extract_relevant_law(DPDP_LAW_TEXT, scenario)
        })
    return matrix

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: GENERATION LOOPS
# ═══════════════════════════════════════════════════════════════════════════
def build_synthesizer_prompt(item):
    law_chunk = extract_relevant_law(DPDP_LAW_TEXT, item["target_violation"])
    edge_case_injection = item["edge_template"]["prompt"] if item.get("edge_template") else "None. Generate a standard deceptive policy."
    return SYNTHESIZER_PROMPT \
        .replace("[LAW_INJECTION]", law_chunk) \
        .replace("[SEED_INJECTION]", item["seed"][:2000]) \
        .replace("[INDUSTRY_INJECTION]", INDUSTRIES.get(item["industry"], "")) \
        .replace("[TARGET_VIOLATION]", item["target_violation"]) \
        .replace("[SUBTLETY_INSTRUCTION]", SUBTLETY_LEVELS.get(item["subtlety_level"], "")) \
        .replace("[EDGE_CASE_INJECTION]", edge_case_injection) \
        .replace("[RAW_POLICY_INJECTION]", item["base_policy"][:6000])

def _save_training_pair(batch_idx, local_idx, item, policy_text, audit, step, lazy_audit):
    law_chunk = extract_relevant_law(DPDP_LAW_TEXT, item["target_violation"])
    if step == 0:
        sft = {"messages": [
            {"role": "system", "content": "Strict DPDP Auditor."},
            {"role": "user", "content": f"[CONTEXT: THE LAW]\n{law_chunk}\n\n[TASK]\nAnalyze:\n{policy_text}"},
            {"role": "assistant", "content": json.dumps(audit, ensure_ascii=False)}
        ]}
        with open(os.path.join(SFT_OUTPUT_DIR, f"sft_{batch_idx:03d}_{local_idx:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(sft, f, ensure_ascii=False)
        return "sft"
    else:
        dpo = {
            "prompt": [{"role": "system", "content": "Strict DPDP Auditor."}, {"role": "user", "content": f"[CONTEXT: THE LAW]\n{law_chunk}\n\n[TASK]\nAnalyze:\n{policy_text}"}],
            "chosen": [{"role": "assistant", "content": json.dumps(audit, ensure_ascii=False)}],
            "rejected": [{"role": "assistant", "content": json.dumps(lazy_audit, ensure_ascii=False)}]
        }
        with open(os.path.join(DPO_OUTPUT_DIR, f"dpo_{batch_idx:03d}_{local_idx:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(dpo, f, ensure_ascii=False)
        return "dpo"

def run_audit_forge():
    print("\n" + "="*70)
    print("⚖️ INITIATING AUDIT FORGE (16,000 Pairs)")
    print("="*70)
    matrix = build_audit_matrix()
    total_batches = math.ceil(len(matrix) / BATCH_SIZE)
    LAZY_AUDIT = {"global_legal_reasoning": "The auditor failed to identify any violations.", "dpdp_trust_score": 100, "violations": [], "subtlety_score": 100}
    stats = {"total_generated": 0, "saved_sft": 0, "saved_dpo": 0, "dropped_auditor_fail": 0, "early_exit_perfect": 0}

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(matrix))
        batch = matrix[batch_start:batch_end]
        needed = []
        for local_idx, item in enumerate(batch):
            sft_file = os.path.join(SFT_OUTPUT_DIR, f"sft_{batch_idx:03d}_{local_idx:03d}.json")
            dpo_file = os.path.join(DPO_OUTPUT_DIR, f"dpo_{batch_idx:03d}_{local_idx:03d}.json")
            if not os.path.exists(sft_file) and not os.path.exists(dpo_file): needed.append((local_idx, item))
        if not needed: continue
        
        gen_messages = [[{"role": "system", "content": "Adversarial corporate counsel."}, {"role": "user", "content": build_synthesizer_prompt(item)}] for _, item in needed]
        gen_out = llm.chat(messages=gen_messages, sampling_params=gen_params)
        if len(gen_out) != len(needed): continue
        
        local_indices = [idx for idx, _ in needed]
        current_policies = {idx: extract_policy(out.outputs[0].text.strip()) or "" for idx, out in zip(local_indices, gen_out)}
        stats["total_generated"] += len(needed)
        completed = set()
        
        for step in range(MAX_REFLEXION_STEPS):
            remaining = [i for i in local_indices if i not in completed and current_policies.get(i)]
            if not remaining: break
            
            judge_msgs = [[{"role": "system", "content": "Strict DPDP Auditor."}, {"role": "user", "content": JUDGE_PROMPT.replace("[LAW_INJECTION]", extract_relevant_law(DPDP_LAW_TEXT, matrix[batch_start + i]["target_violation"])).replace("[POLICY_INJECTION]", current_policies[i])}] for i in remaining]
            audit_outputs = llm.chat(messages=judge_msgs, sampling_params=judge_params)
            
            parsed = {}
            for idx, out in zip(remaining, audit_outputs):
                try:
                    audit = json.loads(out.outputs[0].text.strip())
                    is_valid, error_msg = validate_audit_quality(audit, current_policies[idx])
                    if not is_valid: audit["auditor_failed"] = True; audit["validation_error"] = error_msg
                    parsed[idx] = audit
                except Exception:
                    parsed[idx] = {"auditor_failed": True, "dpdp_trust_score": 0, "violations": [], "subtlety_score": 0, "validation_error": "JSON parse error"}
            
            explicit_heal, subtle_heal = [], []
            explicit_idx, subtle_idx = [], []
            
            for i in remaining:
                audit = parsed[i]
                item = matrix[batch_start + i]
                if audit.get("auditor_failed"):
                    stats["dropped_auditor_fail"] += 1; completed.add(i); continue
                
                score = audit.get("dpdp_trust_score", 0)
                viols = audit.get("violations", [])
                subtlety = audit.get("subtlety_score", 100)
                caught = (score < 90 and len(viols) > 0)
                
                # Early exit for perfect, highly subtle policies
                if caught and score < 20 and subtlety > 85:
                    result = _save_training_pair(batch_idx, i, item, current_policies[i], audit, step, LAZY_AUDIT)
                    if result == "sft": stats["saved_sft"] += 1
                    elif result == "dpo": stats["saved_dpo"] += 1
                    stats["early_exit_perfect"] += 1; completed.add(i); continue
                
                if caught:
                    # Subtle Reflexion: Make it harder to detect
                    if step == 0 and subtlety < 70:
                        tech_category = item["target_category"]
                        obfuscation_tech = SUBTLE_TECHNIQUE_MAP.get(tech_category, "Corporate Vagueness: Use aspirational language instead of concrete commitments.")
                        
                        subtle_heal.append([{
                            "role": "system", "content": "Adversarial corporate counsel."
                        }, {
                            "role": "user", "content": REFLEXION_SUBTLE_PROMPT
                                .replace("[LAW_INJECTION]", extract_relevant_law(DPDP_LAW_TEXT, item["target_violation"]))
                                .replace("[TARGET_VIOLATION]", item["target_violation"])
                                .replace("[OBFUSCATION_TECHNIQUE]", obfuscation_tech)
                                .replace("[INDUSTRY_INJECTION]", INDUSTRIES.get(item["industry"], ""))
                                .replace("[SEED_INJECTION]", item["seed"][:1000]) # Truncated to save context
                                .replace("[FAILED_POLICY_INJECTION]", current_policies[i][:15000])
                        }])
                        subtle_idx.append(i); continue
                        
                    # Save to SFT/DPO
                    result = _save_training_pair(batch_idx, i, item, current_policies[i], audit, step, LAZY_AUDIT)
                    if result == "sft": stats["saved_sft"] += 1
                    elif result == "dpo": stats["saved_dpo"] += 1
                    completed.add(i)
                else:
                    # Explicit Reflexion: Make the violation undeniable
                    if step < MAX_REFLEXION_STEPS - 1:
                        explicit_heal.append([{
                            "role": "system", "content": "Adversarial corporate counsel."
                        }, {
                            "role": "user", "content": REFLEXION_EXPLICIT_PROMPT
                                .replace("[LAW_INJECTION]", extract_relevant_law(DPDP_LAW_TEXT, item["target_violation"]))
                                .replace("[TARGET_VIOLATION]", item["target_violation"])
                                .replace("[AUDIT_FEEDBACK]", json.dumps(audit))
                                .replace("[INDUSTRY_INJECTION]", INDUSTRIES.get(item["industry"], ""))
                                .replace("[SEED_INJECTION]", item["seed"][:1000]) # Truncated to save context
                                .replace("[FAILED_POLICY_INJECTION]", current_policies[i][:15000])
                        }])
                        explicit_idx.append(i)
                    else: 
                        completed.add(i) # Drop if max reflexion reached
            
            heal_temp = max(0.65, 0.95 - 0.15 * step)
            heal_params = SamplingParams(temperature=heal_temp, top_p=0.9, max_tokens=4096)
            if explicit_heal:
                out_explicit = llm.chat(messages=explicit_heal, sampling_params=heal_params)
                for idx, o in zip(explicit_idx, out_explicit): current_policies[idx] = extract_policy(o.outputs[0].text.strip())
            if subtle_heal:
                out_subtle = llm.chat(messages=subtle_heal, sampling_params=heal_params)
                for idx, o in zip(subtle_idx, out_subtle): current_policies[idx] = extract_policy(o.outputs[0].text.strip())
                
        print(f"   ✅ Audit Batch {batch_idx + 1}/{total_batches} | SFT: {stats['saved_sft']} | DPO: {stats['saved_dpo']} | Dropped: {stats['dropped_auditor_fail']}")

def run_chatbot_forge():
    print("\n" + "="*70)
    print("🤖 INITIATING CHATBOT QA FORGE (4,000 Pairs)")
    print("="*70)
    matrix = build_chatbot_matrix()
    total_batches = math.ceil(len(matrix) / BATCH_SIZE)
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(matrix))
        batch = matrix[batch_start:batch_end]
        
        # SFT Generation
        sft_messages = [[{"role": "system", "content": "You are an expert Indian legal AI."}, {"role": "user", "content": CHATBOT_QA_SFT_PROMPT.replace("[LAW_CHUNK]", item["law_chunk"]).replace("[PERSONA_INJECTION]", item["persona"]).replace("[SCENARIO_INJECTION]", item["scenario"])}] for item in batch]
        sft_out = llm.chat(messages=sft_messages, sampling_params=chatbot_sft_params)
        for idx, out in zip(range(batch_start, batch_end), sft_out):
            sft_file = os.path.join(CHATBOT_SFT_DIR, f"qa_sft_{idx:04d}.json")
            if not os.path.exists(sft_file):
                try:
                    raw_text = out.outputs[0].text.strip()
                    if raw_text.startswith("```"): raw_text = re.sub(r'^```json\n|```$', '', raw_text, flags=re.MULTILINE).strip()
                    with open(sft_file, "w", encoding="utf-8") as f: json.dump(json.loads(raw_text), f, ensure_ascii=False)
                except Exception: pass

        # DPO Generation
        dpo_messages = [[{"role": "system", "content": "You are synthesizing legal AI training data."}, {"role": "user", "content": CHATBOT_QA_DPO_PROMPT.replace("[LAW_CHUNK]", item["law_chunk"]).replace("[PERSONA_INJECTION]", item["persona"]).replace("[SCENARIO_INJECTION]", item["scenario"])}] for item in batch]
        dpo_out = llm.chat(messages=dpo_messages, sampling_params=chatbot_dpo_params)
        for idx, out in zip(range(batch_start, batch_end), dpo_out):
            dpo_file = os.path.join(CHATBOT_DPO_DIR, f"qa_dpo_{idx:04d}.json")
            if not os.path.exists(dpo_file):
                try:
                    with open(dpo_file, "w", encoding="utf-8") as f: json.dump(json.loads(out.outputs[0].text.strip()), f, ensure_ascii=False)
                except Exception: pass
                    
        print(f"   ✅ Chatbot Batch {batch_idx + 1}/{total_batches} complete.")

if __name__ == "__main__":
    run_audit_forge()
    run_chatbot_forge()
    print("\n" + "="*70)
    print("✅ COMPLETE DUAL-TRACK FORGE FINISHED")
    print("="*70)