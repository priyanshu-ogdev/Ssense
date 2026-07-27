#!/usr/bin/env python3
"""
gan_forge.py – Ultimate Production GAN Forge (Final Sealed Version)

Optimized and verified for:
- Python 3.12 + Transformers + vLLM + TRL
- Robust JSON-Repair parser catching unescaped inner quotes and syntax anomalies
- Clean, direct JSONL streaming output matching exact Ssense DPDP schemas
- Scale-Hardened Zero-Shot Prompt Architecture with Hard Negative Generation
- Self-Healing Reflexion Loops with Dual-Gate Validation
"""

import math
import os
import json
import glob
import gc
import string
import random
import re
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import itertools
import time
import numpy as np
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import snapshot_download

def resolve_local_rag_model(repo_id: str, model_dirname: str) -> str:
    """Resolves and ensures RAG embedding/reranker models are stored directly in ml/models/."""
    curr = os.path.dirname(os.path.abspath(__file__))
    models_root = None
    while curr and curr != os.path.dirname(curr):
        candidate = os.path.join(curr, "ml", "models")
        if os.path.isdir(candidate):
            models_root = candidate
            break
        curr = os.path.dirname(curr)
    if not models_root:
        models_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    
    local_path = os.path.join(models_root, model_dirname)
    os.makedirs(local_path, exist_ok=True)
    
    if not os.path.exists(os.path.join(local_path, "config.json")):
        print(f"📥 Downloading {repo_id} directly to local studio store at ml/models/{model_dirname}...")
        try:
            snapshot_download(repo_id=repo_id, local_dir=local_path)
            print(f"✅ {repo_id} saved offline at ml/models/{model_dirname}.")
        except Exception as e:
            print(f"⚠️ Warning: Snapshot download failed ({e}). Falling back to repo_id.")
            return repo_id
    else:
        print(f"✅ Loaded {repo_id} directly from offline store at ml/models/{model_dirname}.")
    return local_path

# Global Hybrid Search Cache
RAG_CHUNKS = []
RAG_METADATAS = []
RAG_EMBEDDINGS = None
RAG_BM25 = None
BGE_MODEL = None
RERANKER_MODEL = None

try:
    with open("dpdp_hybrid_index.pkl", "rb") as f:
        meta = pickle.load(f)
        RAG_CHUNKS = meta["chunks"]
        RAG_METADATAS = meta["metadatas"]
        RAG_BM25 = meta["bm25_index"]
        RAG_EMBEDDINGS = meta["dense_embeddings"]
        
        # Integrity check
        if not (meta["chunk_count"] == len(RAG_CHUNKS) == len(RAG_METADATAS) == RAG_EMBEDDINGS.shape[0]):
            raise ValueError("Index integrity check failed: mismatched array lengths.")
            
    bge_local_path = resolve_local_rag_model("BAAI/bge-small-en-v1.5", "bge-small-en-v1.5")
    reranker_local_path = resolve_local_rag_model("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3")
    BGE_MODEL = SentenceTransformer(bge_local_path)
    RERANKER_MODEL = CrossEncoder(reranker_local_path, max_length=512)
    print(f"✅ Hybrid RAG Cache loaded successfully ({len(RAG_CHUNKS)} chunks).")
except Exception as e:
    print(f"Warning: Hybrid RAG cache not found or failed to load. Run build_vector_db.py first. {e}")

def get_legal_query_tokens(query: str) -> list:
    """Exact mirror of the builder's tokenizer for perfect BM25 alignment."""
    query_lower = query.lower()
    query_lower = query_lower.replace("data fiduciary", "data_fiduciary") \
                             .replace("data principal", "data_principal") \
                             .replace("consent manager", "consent_manager") \
                             .replace("sub-section", "subsection")
    query_lower = re.sub(r'section\s+(\d+)\((\d+)\)', r'section_\1_\2', query_lower)
    query_lower = re.sub(r'rule\s+(\d+)\((\d+)\)', r'rule_\1_\2', query_lower)
    query_lower = re.sub(r'section\s+(\d+)', r'section_\1', query_lower)
    query_lower = re.sub(r'rule\s+(\d+)', r'rule_\1', query_lower)
    
    generic_stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "and", "in", "to", "for", "with", "on", "at", "by", "from", "as", "that", "this", "it", "be", "or", "which", "will", "would", "could", "should", "their", "they"}
    
    return [w for w in re.findall(r'\b\w+\b', query_lower) if w not in generic_stopwords and len(w) > 2]

def semantic_rag_query(query: str, n_results: int = 3, is_private_audit: bool = True) -> str:
    if not RAG_CHUNKS or RAG_EMBEDDINGS is None or RAG_BM25 is None:
        return "RAG Retrieval Error: Hybrid index not found."
    try:
        # 1. Lexical BM25 Search (Cleaned: get_legal_query_tokens handles all lowercasing/replacing)
        query_tokens = get_legal_query_tokens(query)
        bm25_scores = RAG_BM25.get_scores(query_tokens)
        
        # 2. Dense Semantic Search
        q_emb = BGE_MODEL.encode([f"Represent this query for retrieval: {query}"])[0]
        q_emb = q_emb / np.linalg.norm(q_emb)
        dense_scores = np.dot(RAG_EMBEDDINGS, q_emb)
        
        # 3. Pre-Filter & Reciprocal Rank Fusion (RRF k=60)
        rrf_scores = np.zeros(len(RAG_CHUNKS))
        
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        dense_ranks = np.argsort(dense_scores)[::-1]
        
        for rank, idx in enumerate(bm25_ranks):
            if is_private_audit and RAG_METADATAS[idx].get("applies_to") == "state":
                continue
            rrf_scores[idx] += 1.0 / (60 + rank + 1)
            
        for rank, idx in enumerate(dense_ranks):
            if is_private_audit and RAG_METADATAS[idx].get("applies_to") == "state":
                continue
            rrf_scores[idx] += 1.0 / (60 + rank + 1)
        
        # 4. Deduplicate Candidates (Top 10)
        top_indices = np.argsort(rrf_scores)[::-1]
        candidates = []
        seen = set()
        for idx in top_indices:
            if rrf_scores[idx] == 0: break # Skip zero scores
            chunk_text = RAG_CHUNKS[idx]
            if chunk_text not in seen:
                seen.add(chunk_text)
                candidates.append(chunk_text)
            if len(candidates) == 10: break
            
        if not candidates:
            return "No valid context found."
            
        # 5. Cross-Encoder Reranking
        cross_inp = [[query, doc] for doc in candidates]
        cross_scores = RERANKER_MODEL.predict(cross_inp)
        best_indices = np.argsort(cross_scores)[::-1][:n_results]
        
        return "\n\n---\n\n".join([candidates[i] for i in best_indices])
    except Exception as e:
        return f"RAG Retrieval Error: {str(e)}"

from collections import defaultdict
from difflib import SequenceMatcher

# Optional vLLM import for local direct engine fallback
try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    class SamplingParams:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    class StructuredOutputsParams:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0: MODULAR IMPORTS & PROMPT LOADING
# ═══════════════════════════════════════════════════════════════════════════
from prompts.target_violations import TARGET_VIOLATIONS, ATOMIC_STATUTES, SEMANTIC_KEYWORD_MAP
from prompts.edge_case_templates import EDGE_CASE_TEMPLATES

def load_prompt(filename: str) -> str:
    path = os.path.join("prompts", filename)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

SYNTHESIZER_PROMPT = load_prompt("synthesizer_prompt.txt")
JUDGE_PROMPT = load_prompt("judge_prompt.txt")
HARD_NEGATIVE_PROMPT = load_prompt("hard_negative_prompt.txt")
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

SLM_DATA_DIR = "../slm-training/data"
JSONL_AUDIT_SFT = os.path.join(SLM_DATA_DIR, "audit_sft_data.jsonl")
JSONL_AUDIT_DPO = os.path.join(SLM_DATA_DIR, "audit_dpo_data.jsonl")
JSONL_CHATBOT_SFT = os.path.join(SLM_DATA_DIR, "chatbot_sft_data.jsonl")
JSONL_CHATBOT_DPO = os.path.join(SLM_DATA_DIR, "chatbot_dpo_data.jsonl")

SCHEMA_PATH = "../../libs/contracts/schemas/dpdp_schema.json"
MODEL_PATH = os.getenv("TEACHER_MODEL_PATH", "../models/Qwen2-72B-Instruct-FP8")

TARGET_AUDIT_POLICIES = int(os.getenv("TARGET_AUDIT_POLICIES", "2000"))
TARGET_CHATBOT_PAIRS = int(os.getenv("TARGET_CHATBOT_PAIRS", "1000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MAX_REFLEXION_STEPS = 3

for d in [SFT_OUTPUT_DIR, DPO_OUTPUT_DIR, CHATBOT_SFT_DIR, CHATBOT_DPO_DIR, SLM_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

def build_law_text():
    if os.path.exists(LAW_TEXT_PATH):
        return
    try:
        import fitz
    except ImportError:
        sys.exit("PyMuPDF required: pip install PyMuPDF")
    if not os.path.exists(PDF_ACT) or not os.path.exists(PDF_RULES):
        raise FileNotFoundError("Place both DPDP PDFs in this directory.")
    act_text = "\n".join(page.get_text("text") for page in fitz.open(PDF_ACT))
    rules_text = "\n".join(page.get_text("text") for page in fitz.open(PDF_RULES))
    combined = f"=== DIGITAL PERSONAL DATA PROTECTION ACT 2023 ===\n\n{act_text}\n\n=== DIGITAL PERSONAL DATA PROTECTION RULES 2025 ===\n\n{rules_text}\n"
    with open(LAW_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(combined)

build_law_text()
with open(LAW_TEXT_PATH, "r", encoding="utf-8") as f:
    DPDP_LAW_TEXT = f.read()

if not os.path.exists(SCHEMA_PATH):
    raise FileNotFoundError(f"Missing schema: {SCHEMA_PATH}")

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

def filter_english(text, threshold=0.05):
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        deva_chars = len(DEVANAGARI_RE.findall(line))
        if (deva_chars / max(1, len(line))) < threshold:
            lines.append(line)
    return '\n'.join(lines)

DPDP_LAW_TEXT = filter_english(DPDP_LAW_TEXT, threshold=0.05)

indian_seeds = [open(f, "r", encoding="utf-8").read() for f in glob.glob(os.path.join(INDIAN_SEEDS_DIR, "*.txt"))]
if not indian_seeds:
    raise RuntimeError("No Indian seeds found.")

raw_policies = []
for f in glob.glob(os.path.join(RAW_POLICIES_DIR, "*.txt")):
    with open(f, "r", encoding="utf-8") as fh:
        cleaned = filter_english(fh.read())
        if len(cleaned) > 2000:
            raw_policies.append(cleaned)
if not raw_policies:
    raise RuntimeError("No valid raw policies found.")

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
    "A small business owner trying to understand SDF thresholds.",
    "A government hospital administrator dealing with patient data.",
    "A startup CTO trying to understand SDF thresholds.",
    "A consumer rights NGO worker advocating for rural citizens.",
    "A cross-border SaaS vendor managing Indian client data.",
    "A school principal handling student records and grades.",
    "An HR manager at a large employer dealing with employee data.",
    "An elderly individual confused about consent notices.",
    "A freelance software developer building a web application."
]

def get_expanded_scenarios():
    return [
    {
        "id": "consent_basics_1",
        "primary_section": "6(1)",
        "scenario": "Turn 1: How can we obtain valid consent for our app?\nTurn 2: Is a pre-checked box legally sufficient?\nTurn 3: Does the consent prompt need to explain every third party we share data with?"
    },
    {
        "id": "consent_basics_2",
        "primary_section": "6(2)",
        "scenario": "Turn 1: Can we bundle consent for marketing with consent for our core service?\nTurn 2: What if the core service is free, can we make marketing a condition then?\nTurn 3: How do we prove the consent was free and unconditional?"
    },
    {
        "id": "consent_withdrawal_1",
        "primary_section": "6(4)",
        "scenario": "Turn 1: A user wants to withdraw consent. How quickly must we process this?\nTurn 2: Must we provide a 'one-click' withdrawal option?\nTurn 3: What if processing their withdrawal breaks their active subscription?"
    },
    {
        "id": "consent_withdrawal_2",
        "primary_section": "6(5)",
        "scenario": "Turn 1: If a user withdraws consent, do we lose the legal right to the data we already processed?\nTurn 2: Can we retain their data for audit purposes after withdrawal?\nTurn 3: Are we liable for actions taken by our processors before the withdrawal?"
    },
    {
        "id": "consent_manager_1",
        "primary_section": "6(7)",
        "scenario": "Turn 1: What is a Consent Manager under the DPDP Act?\nTurn 2: Are we legally required to integrate with them?\nTurn 3: Can we charge users a fee for using a Consent Manager?"
    },
    {
        "id": "consent_manager_2",
        "primary_section": "6(8)",
        "scenario": "Turn 1: Who is responsible if a Consent Manager maliciously alters a user's preferences?\nTurn 2: Does the Data Fiduciary face penalties for the Consent Manager's failure?\nTurn 3: How do we verify the identity of the Consent Manager?"
    },
    {
        "id": "legitimate_use_1",
        "primary_section": "7(a)",
        "scenario": "Turn 1: A user voluntarily gave us their email. Is that deemed consent for our newsletter?\nTurn 2: What if they gave it for a customer support ticket, can we still send marketing?\nTurn 3: Does 'voluntary provision' require a privacy notice?"
    },
    {
        "id": "legitimate_use_2",
        "primary_section": "7(b)",
        "scenario": "Turn 1: Can the State process data without consent for subsidies?\nTurn 2: Are private contractors working for the State exempt under this clause?\nTurn 3: Must the State provide a privacy notice when processing under this section?"
    },
    {
        "id": "legitimate_use_3",
        "primary_section": "7(c)",
        "scenario": "Turn 1: Is processing medical data during a health emergency allowed without consent?\nTurn 2: Does this apply to private hospitals or only government bodies?\nTurn 3: What happens to the data after the emergency is over?"
    },
    {
        "id": "legitimate_use_4",
        "primary_section": "7(d)",
        "scenario": "Turn 1: Can an employer process employee biometrics without explicit consent?\nTurn 2: Does this cover processing data to monitor employee productivity?\nTurn 3: Can the employee opt out of this processing?"
    },
    {
        "id": "processor_liability_1",
        "primary_section": "8(1)",
        "scenario": "Turn 1: Are we liable if our cloud provider suffers a data breach?\nTurn 2: Can we use a contract to shift DPDP liability entirely to the processor?\nTurn 3: What specific clauses are mandatory in a processor contract?"
    },
    {
        "id": "processor_liability_2",
        "primary_section": "8(2)",
        "scenario": "Turn 1: Must we conduct technical audits on our processors?\nTurn 2: Is it sufficient to rely on their ISO certifications?\nTurn 3: What if the processor is located outside India?"
    },
    {
        "id": "data_accuracy_1",
        "primary_section": "8(3)",
        "scenario": "Turn 1: Are we required to verify the accuracy of all user data?\nTurn 2: If a user provides a fake name, are we penalized?\nTurn 3: How does this apply to data collected from third-party brokers?"
    },
    {
        "id": "data_security_1",
        "primary_section": "8(4)",
        "scenario": "Turn 1: What 'reasonable security safeguards' are mandated by the Act?\nTurn 2: Is end-to-end encryption a strict legal requirement?\nTurn 3: Can we be fined if we had security measures but a breach still occurred?"
    },
    {
        "id": "breach_notification_1",
        "primary_section": "8(6)",
        "scenario": "Turn 1: We discovered a data breach. Who exactly must we notify?\nTurn 2: Is there a specific format for notifying the Data Protection Board?\nTurn 3: Must we notify users if the breached data was fully encrypted?"
    },
    {
        "id": "breach_notification_2",
        "primary_section": "8(6)",
        "scenario": "Turn 1: What is the exact timeline for breach notification?\nTurn 2: Does the 72-hour clock start upon suspicion or confirmation?\nTurn 3: Can we delay notification to conduct a forensic investigation?"
    },
    {
        "id": "data_erasure_1",
        "primary_section": "8(7)",
        "scenario": "Turn 1: When must we erase user data if they haven't requested it?\nTurn 2: What defines the 'specified purpose' being no longer served?\nTurn 3: How do we handle backups and archives in data erasure?"
    },
    {
        "id": "data_erasure_2",
        "primary_section": "8(7)",
        "scenario": "Turn 1: Can we retain data to comply with tax laws even if the user withdraws consent?\nTurn 2: Must we inform the user when their data is automatically erased?\nTurn 3: Do we have to ensure our processors also erase the data?"
    },
    {
        "id": "dpo_contact_1",
        "primary_section": "8(9)",
        "scenario": "Turn 1: Must every company appoint a Data Protection Officer?\nTurn 2: Does the contact information need to be a physical address or is email sufficient?\nTurn 3: Can the DPO be based outside India?"
    },
    {
        "id": "grievance_mechanism_1",
        "primary_section": "8(10)",
        "scenario": "Turn 1: What are the requirements for an internal grievance redressal mechanism?\nTurn 2: Is there a statutory time limit for resolving grievances?\nTurn 3: Can we require users to send grievances via registered post?"
    },
    {
        "id": "child_consent_1",
        "primary_section": "9(1)",
        "scenario": "Turn 1: How do we legally obtain 'verifiable parental consent'?\nTurn 2: Is a checkbox confirming 'I am over 18' sufficient?\nTurn 3: What if a child lies about their age on our platform?"
    },
    {
        "id": "child_tracking_1",
        "primary_section": "9(2)",
        "scenario": "Turn 1: Are we absolutely prohibited from tracking children's data?\nTurn 2: Can we serve targeted ads to children if the parent consents?\nTurn 3: Does this ban apply to educational platforms monitoring student progress?"
    },
    {
        "id": "child_exemptions_1",
        "primary_section": "9(4)",
        "scenario": "Turn 1: How can an EdTech company get an exemption from the tracking ban?\nTurn 2: Who determines if the processing is 'verifiably safe'?\nTurn 3: Can the exemption be revoked?"
    },
    {
        "id": "sdf_threshold_1",
        "primary_section": "10(1)",
        "scenario": "Turn 1: How does a company become a Significant Data Fiduciary (SDF)?\nTurn 2: Is it purely based on the volume of data processed?\nTurn 3: Does processing sensitive financial data automatically make us an SDF?"
    },
    {
        "id": "sdf_dpo_1",
        "primary_section": "10(2)(a)",
        "scenario": "Turn 1: What are the specific requirements for an SDF's Data Protection Officer?\nTurn 2: Can the DPO also serve as the Chief Information Security Officer?\nTurn 3: Does the DPO report to the CEO or the Board of Directors?"
    },
    {
        "id": "sdf_auditor_1",
        "primary_section": "10(2)(b)",
        "scenario": "Turn 1: Must an SDF appoint an Independent Data Auditor?\nTurn 2: How often must the independent audit be conducted?\nTurn 3: Who receives the auditor's final report?"
    },
    {
        "id": "sdf_dpia_1",
        "primary_section": "10(2)(c)",
        "scenario": "Turn 1: What exactly is a Data Protection Impact Assessment (DPIA)?\nTurn 2: Do we need to conduct a DPIA for every new feature?\nTurn 3: Must the DPIA be published publicly?"
    },
    {
        "id": "right_to_access_1",
        "primary_section": "11(1)",
        "scenario": "Turn 1: A user requested a summary of all their data. What must we provide?\nTurn 2: Must we provide the raw data or just a descriptive summary?\nTurn 3: How long do we have to respond to this request?"
    },
    {
        "id": "right_to_identities_1",
        "primary_section": "11(2)",
        "scenario": "Turn 1: Must we disclose the names of all third parties we shared user data with?\nTurn 2: Are there any exemptions for sharing data with law enforcement?\nTurn 3: What if we shared the data in an aggregated, anonymized format?"
    },
    {
        "id": "right_to_correction_1",
        "primary_section": "12(1)",
        "scenario": "Turn 1: A user wants us to correct their address. What is the process?\nTurn 2: Are we legally required to verify the new address before updating it?\nTurn 3: What if the correction conflicts with our KYC records?"
    },
    {
        "id": "right_to_erasure_1",
        "primary_section": "12(3)",
        "scenario": "Turn 1: If a user demands erasure, must we delete their data immediately?\nTurn 2: What if deleting their account breaks financial records required by law?\nTurn 3: Are we required to notify third-party processors to also erase the data?"
    },
    {
        "id": "right_to_grievance_1",
        "primary_section": "13(1)",
        "scenario": "Turn 1: Can a user complain directly to the Data Protection Board?\nTurn 2: What constitutes an 'exhaustion' of the internal grievance mechanism?\nTurn 3: Can the Board penalize a user for filing a frivolous complaint?"
    },
    {
        "id": "right_to_nominate_1",
        "primary_section": "14",
        "scenario": "Turn 1: What is the right to nominate under the DPDP Act?\nTurn 2: Can the nominee access the user's data while the user is still alive?\nTurn 3: Does this right supersede inheritance laws?"
    },
    {
        "id": "duties_of_principal_1",
        "primary_section": "15",
        "scenario": "Turn 1: Are there any legal duties imposed on the users themselves?\nTurn 2: Can a user be fined for providing a fake identity?\nTurn 3: Who enforces the penalties against Data Principals?"
    },
    {
        "id": "cross_border_1",
        "primary_section": "16(1)",
        "scenario": "Turn 1: Can we transfer Indian users' data to servers in the USA?\nTurn 2: Does the Central Government maintain a whitelist or a blacklist of countries?\nTurn 3: What happens if another Indian law strictly requires local data storage?"
    },
    {
        "id": "state_exemption_1",
        "primary_section": "17(1)(a)",
        "scenario": "Turn 1: Is the government exempt from the DPDP Act for national security?\nTurn 2: Does this exemption apply to private contractors building software for the military?\nTurn 3: Who oversees the government's use of this exemption?"
    },
    {
        "id": "research_exemption_1",
        "primary_section": "17(2)(b)",
        "scenario": "Turn 1: Can we process personal data for statistical research without consent?\nTurn 2: What safeguards must be in place to qualify for this exemption?\nTurn 3: Can the research findings include personally identifiable information?"
    },
    {
        "id": "startup_exemption_1",
        "primary_section": "17(3)",
        "scenario": "Turn 1: Are startups entirely exempt from the DPDP Act?\nTurn 2: Which specific obligations can the Central Government exempt startups from?\nTurn 3: What defines a startup for the purpose of this exemption?"
    },
    {
        "id": "board_inquiry_1",
        "primary_section": "27(1)",
        "scenario": "Turn 1: Can the Board initiate an inquiry without a user complaint?\nTurn 2: What triggers the Board's power to inquire?\nTurn 3: Does the Board have to provide a hearing before issuing penalties?"
    },
    {
        "id": "board_powers_1",
        "primary_section": "28(1)",
        "scenario": "Turn 1: Does the Data Protection Board have the same powers as a civil court?\nTurn 2: Can they summon our CEO for an interrogation?\nTurn 3: Can the Board order a police search of our premises?"
    },
    {
        "id": "interim_orders_1",
        "primary_section": "27(3)",
        "scenario": "Turn 1: Can the Board issue interim orders during an ongoing inquiry?\nTurn 2: Could they force us to halt our core business operations temporarily?\nTurn 3: Can we appeal an interim order immediately to TDSAT?"
    },
    {
        "id": "tdsat_appeal_1",
        "primary_section": "29(1)",
        "scenario": "Turn 1: Where do we appeal if the Board issues a massive penalty against us?\nTurn 2: What is the time limit for filing an appeal to TDSAT?\nTurn 3: Do we have to pay the penalty before the appeal is heard?"
    },
    {
        "id": "tdsat_powers_1",
        "primary_section": "29(7)",
        "scenario": "Turn 1: Can TDSAT completely overturn the Board's decision?\nTurn 2: Does TDSAT conduct a fresh investigation or only review the Board's order?\nTurn 3: Where can we appeal if TDSAT rules against us?"
    },
    {
        "id": "penalties_breach_1",
        "primary_section": "33(1)",
        "scenario": "Turn 1: What is the maximum penalty for failing to prevent a data breach?\nTurn 2: Is the penalty calculated per affected user or per incident?\nTurn 3: Can the Board shut down the company instead of a financial penalty?"
    },
    {
        "id": "penalties_child_1",
        "primary_section": "33(2)",
        "scenario": "Turn 1: What is the fine for violating the children's data provisions?\nTurn 2: Is the fine the same for both targeted tracking and lack of parental consent?\nTurn 3: Does the fine go to the affected children or the government?"
    },
    {
        "id": "penalties_factors_1",
        "primary_section": "33(2)",
        "scenario": "Turn 1: How does the Board determine the exact penalty amount?\nTurn 2: Does taking immediate mitigating action reduce the fine?\nTurn 3: Does a previous violation history significantly increase the penalty?"
    },
    {
        "id": "civil_court_bar_1",
        "primary_section": "39",
        "scenario": "Turn 1: Can a user file a lawsuit in a civil court for a data breach?\nTurn 2: Does the DPDP Act completely bar civil court jurisdiction?\nTurn 3: Can a user still claim compensation under the IT Act or tort law?"
    },
    {
        "id": "rti_override_1",
        "primary_section": "44(3)",
        "scenario": "Turn 1: How does the DPDP Act amend the Right to Information (RTI) Act?\nTurn 2: Can a citizen still demand personal information of a government official under RTI?\nTurn 3: Does the 'public interest' exemption still exist for personal data under RTI?"
    },
    {
        "id": "rules_notice_1",
        "primary_section": "Rule 3",
        "scenario": "Turn 1: Does the privacy notice need to be available in all 22 official languages?\nTurn 2: How must the notice be presented to the user during registration?\nTurn 3: Can the English version be the only legally binding one?"
    },
    {
        "id": "rules_breach_1",
        "primary_section": "Rule 7",
        "scenario": "Turn 1: What exactly must be included in the breach notification to the Board?\nTurn 2: Do we have to specify the number of affected users within the 72 hours?\nTurn 3: What if we don't know the full extent of the breach yet?"
    }
]

CHATBOT_SCENARIOS = get_expanded_scenarios()


SUBTLE_TECHNIQUE_MAP = {
    "purpose_limitation": "Legitimate Use Overreach: Claim that voluntarily providing data constitutes 'deemed consent' for secondary commercial marketing.",
    "consent": "Implied Consent Trap: Use phrases like 'continued use constitutes consent', 'presumptive consent', or 'integral to our ecosystem services'.",
    "legitimate_uses_abuse": "Legitimate Use Overreach: Frame commercial processing as essential for 'service continuity' or 'regulatory compliance' to bypass explicit consent.",
    "notice": "Vagueness Shield: Replace concrete disclosures with aspirational fluff like 'We strive to maintain a trusted environment' or 'Privacy is paramount to our goals'.",
    "retention": "Future-Proofing Loophole: Attribute indefinite retention to 'legacy system dependencies', 'statutory audit requirements', or 'business continuity protocols'.",
    "children": "Ambiguous Age Verification: State that you 'do not knowingly collect data from children' without specifying a verifiable consent mechanism, or rely on 'self-certification'.",
    "security": "Delegated Liability: Claim you use 'industry-standard measures' but explicitly disclaim liability for third-party vendor breaches.",
    "breach_notification": "Internal Triage Delay: State that notification will occur 'within 72 hours of the conclusive completion of our internal forensics triage'.",
    "processor_accountability": "Vendor Shield: Explicitly state that the Data Fiduciary is 'not responsible for third-party breaches' or 'vendor is solely liable'.",
    "grievance": "Procedural Friction: Require 'notarized documentation', 'registered post', or a '60-day internal triage period' to process any rights requests.",
    "sdf_obligations": "Trade Secret Exemption: Claim your core algorithms are 'proprietary trade secrets' and therefore exempt from external algorithmic auditing or DPIA disclosures.",
    "algorithmic_profiling": "Black Box Exemption: Hide behind 'proprietary trade secrets' to avoid disclosing automated decision-making logic.",
    "crossborder": "Global Infrastructure Veil: State that data is processed in 'jurisdictions that meet international best practices' without naming specific countries or safeguards.",
    "consent_manager": "Cryptographic Blockade: Refuse Consent Manager integration by citing 'cryptographic integrity' and forcing users to use the 'native application dashboard'.",
    "language_accessibility": "Legal Precision Shield: State that 'legally binding notices are maintained exclusively in English to ensure absolute legal precision'.",
    "rights_implementation": "Nominee Invalidation: State that 'accounts and data rights are strictly non-transferable' and refuse to recognize post-mortem nominees.",
    "board_compliance_violation": "Jurisdictional Shielding: Claim the company is only subject to foreign courts or private arbitration, actively overriding the Data Protection Board's statutory authority.",
    "penalty_avoidance": "Liability Capping: Insert clauses capping total liability for data breaches to 'fees paid in the last 12 months' or a trivial fixed amount, attempting to nullify statutory penalties.",
    "appeal_process_violation": "Forced Arbitration: State that all disputes regarding data privacy MUST be resolved through private binding arbitration, stripping the right to appeal to TDSAT.",
    "scope_application_evasion": "Physical Collection Loophole: Claim that data initially collected on paper forms at physical branches is completely exempt from the digital privacy policy.",
    "illegal_exemption_claim": "False State Exemption: A private corporate entity claiming it is exempt from consent requirements 'for the security of the State' because it provides software to a government client."
}
# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: DYNAMIC CONTEXT & VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════
from prompts.target_violations import TARGET_VIOLATIONS

law_cache = {}

from difflib import get_close_matches

# Global cache for Semantic Key Matching
RAG_KEY_EMBEDDINGS = None
RAG_KEYS_LIST = None

def _get_semantic_key_cache():
    """Lazily loads and caches the BGE embeddings for all TARGET_VIOLATIONS keys."""
    global RAG_KEY_EMBEDDINGS, RAG_KEYS_LIST
    if RAG_KEY_EMBEDDINGS is None and BGE_MODEL is not None:
        RAG_KEYS_LIST = list(TARGET_VIOLATIONS.keys())
        # Embed the keys using the same instruction prefix used for documents
        RAG_KEY_EMBEDDINGS = BGE_MODEL.encode([
            f"Represent this legal concept for retrieval: {k}" for k in RAG_KEYS_LIST
        ])
        # L2 Normalize for fast cosine similarity via dot product
        RAG_KEY_EMBEDDINGS = RAG_KEY_EMBEDDINGS / np.linalg.norm(RAG_KEY_EMBEDDINGS, axis=1, keepdims=True)
    return RAG_KEYS_LIST, RAG_KEY_EMBEDDINGS

def get_audit_rag_context(target_category: str) -> str:
    """
    DETERMINISTIC AUDIT CONTEXT: 4-Tier Graceful Degradation.
    Prioritizes the pristine TARGET_VIOLATIONS dictionary.
    Falls back to the SOTA Hybrid RAG (with State-Schedule filtering) only if the dictionary fails.
    """
    if not target_category:
        return ""

    target_category = str(target_category).strip().lower()
    
    # TIER 1: Exact Match (Fast Path)
    if target_category in TARGET_VIOLATIONS:
        return "\n\n".join(TARGET_VIOLATIONS[target_category])
    
    # TIER 2: Fuzzy Key Match (Catches typos, trailing spaces, plurals)
    close_matches = get_close_matches(target_category, TARGET_VIOLATIONS.keys(), n=1, cutoff=0.6)
    if close_matches:
        best_key = close_matches[0]
        return "\n\n".join(TARGET_VIOLATIONS[best_key])
    print(f"⚠️ FATAL: Dictionary lookup FAILED for '{target_category}'. Available keys: {list(TARGET_VIOLATIONS.keys())[:5]}")
    
    # TIER 3: Semantic Key Match (Uses BGE to find the closest dictionary key)
    keys_list, key_embeddings = _get_semantic_key_cache()
    if key_embeddings is not None:
        target_embedding = BGE_MODEL.encode([f"Represent this legal concept for retrieval: {target_category}"])[0]
        target_embedding = target_embedding / np.linalg.norm(target_embedding)
        
        # Cosine similarity via dot product
        similarities = np.dot(key_embeddings, target_embedding)
        best_key_idx = np.argmax(similarities)
        best_key = keys_list[best_key_idx]
        
        # Accept if semantic similarity is reasonably high (> 0.5)
        if similarities[best_key_idx] > 0.5: 
            return "\n\n".join(TARGET_VIOLATIONS[best_key])

    # TIER 4: SOTA Hybrid RAG Fallback (State-Schedule Filtered)
    # If the category is completely novel and fails semantic matching, 
    # we query the Hybrid RAG index but strictly filter out State/Instrumentality chunks.
    return semantic_rag_query(target_category, n_results=2, is_private_audit=True)

def extract_relevant_law(law_text, target_violation):
    target_lower = str(target_violation or "").lower()
    if target_lower in law_cache:
        return law_cache[target_lower]

    keywords = []
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
    elif "section 33" in target_lower or "penalty" in target_lower or "schedule" in target_lower or "avoidance" in target_lower:
        keywords = ["Section 33", "Penalty", "Schedule", "fine", "crore"]
    elif "section 29" in target_lower or "tdsat" in target_lower or "appeal" in target_lower or "section 39" in target_lower or "civil court" in target_lower:
        keywords = ["Section 29", "Section 39", "TDSAT", "Appellate", "Tribunal", "civil court"]
    elif "section 17" in target_lower or "exemption" in target_lower or "illegal" in target_lower:
        keywords = ["Section 17", "Exemption", "State", "security of India", "instrumentality"]
    elif "section 28" in target_lower or "board" in target_lower or "compliance" in target_lower or "summon" in target_lower:
        keywords = ["Section 28", "summon", "inquiry", "interim orders", "civil court"]
    elif "section 3" in target_lower or "scope" in target_lower or "evasion" in target_lower or "territory" in target_lower:
        keywords = ["Section 3", "offline", "digitise", "territory of India", "outside the territory"]
    elif "section 44" in target_lower or "rti" in target_lower:
        keywords = ["Section 44", "RTI", "Right to Information"]

    keywords_lower = [kw.lower() for kw in keywords]
    query_text = target_lower + " " + " ".join(keywords)
    
    rag_result = semantic_rag_query(query_text, n_results=3)
    
    if "RAG Retrieval Error" not in rag_result and len(rag_result) > 50:
        result = rag_result
    else:
        # Fallback to strict regex or a much smaller chunk (3000 chars instead of 36000)
        paragraphs = law_text.split('\n\n')
        if len(paragraphs) == 1:
            paragraphs = law_text.split('\n')
            
        relevant_chunks = []
        if keywords_lower:
            for p in paragraphs:
                p_lower = p.lower()
                if any(kw in p_lower for kw in keywords_lower):
                    relevant_chunks.append(p.strip())
        
        if not relevant_chunks:
            relevant_chunks = paragraphs
            
        MAX_LAW_CHARS = 4000 # Strict fallback constraint to prevent bloat
        final_chunks = []
        current_length = 0
        for chunk in relevant_chunks:
            if current_length + len(chunk) + 2 > MAX_LAW_CHARS:
                break
            final_chunks.append(chunk)
            current_length += len(chunk) + 2
            
        result = "\n\n".join(final_chunks)

    law_cache[target_lower] = result
    return result

def is_quote_in_policy(quote, policy):
    if not quote or not policy:
        return False
    quote = re.sub(r'\.{2,}|\u2026', '', str(quote)).strip()
    policy = re.sub(r'\.{2,}|\u2026', '', str(policy)).strip()
    
    # Fast-path exact match
    if quote.lower() in policy.lower(): 
        return True

    translator = str.maketrans('', '', string.punctuation + '“”‘’"\'\n\t')
    norm_quote = re.sub(r'\s+', ' ', quote.lower().translate(translator)).strip()
    norm_policy = re.sub(r'\s+', ' ', policy.lower().translate(translator)).strip()
    
    if norm_quote in norm_policy: 
        return True
    
    quote_words = norm_quote.split()
    if len(quote_words) >= 4:
        for start_offset in range(min(4, max(1, len(quote_words) - 3))):
            phrase = " ".join(quote_words[start_offset:start_offset + 4])
            idx = norm_policy.find(phrase)
            
            match_attempts = 0  # 🚨 PREVENTS O(N^2) CPU BOMB
            
            while idx != -1 and match_attempts < 5:
                window = norm_policy[max(0, idx - 20):idx + len(norm_quote) + 40]
                if SequenceMatcher(None, norm_quote, window[:len(norm_quote) + 20]).ratio() > 0.65:
                    return True
                idx = norm_policy.find(phrase, idx + 1)
                match_attempts += 1
                
    if len(norm_quote) >= 20:
        q_tokens = set(norm_quote.split())
        p_tokens = set(norm_policy.split())
        if len(q_tokens) > 0 and len(q_tokens & p_tokens) / len(q_tokens) >= 0.80:
            return True
    return False
    
ENUM_MAPPINGS = {
    "WARN_USER": "WARN_USER_ONLY", "WARN": "WARN_USER_ONLY", "WARN_USER_ONLY_ACTION": "WARN_USER_ONLY",
    "BLOCK": "BLOCK_THIRD_PARTY", "BLOCK_THIRD": "BLOCK_THIRD_PARTY", "BLOCK_THIRD_PARTY_ACTION": "BLOCK_THIRD_PARTY",
    "STRIP_TELEMETRY": "STRIP_TELEMETRY_HEADER", "STRIP": "STRIP_TELEMETRY_HEADER",
    "SPOOF_API": "SPOOF_HARDWARE_API", "SPOOF": "SPOOF_HARDWARE_API", "SPOOF_HARDWARE": "SPOOF_HARDWARE_API",
    "INJECT_GPC": "INJECT_GPC_SIGNAL", "INJECT": "INJECT_GPC_SIGNAL",
}

def is_administrative_element(quote: str, offending_entities: list = None) -> bool:
    if not quote or not isinstance(quote, str): return False
    norm = quote.lower().strip()
    
    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', quote))
    has_phone = bool(re.search(r'\b\d{10}\b|\+91-\d{2}-\d{8}', quote))
    has_address_marker = any(m in norm for m in ["registered office", "corporate office", "postal code", "pin code", "gstin:", "grievance officer", "data protection officer", "dpo", "nodal officer", "concerns or clarifications", "contact us", "queries regarding", "questions about", "reach out to"])
    
    # Check for company introduction preambles
    intro_pattern = r'^(?:welcome to|at\s+[A-Z]|in\s+our\s+pursuit|we\s+are\s+dedicated\s+to|[A-Z][A-Za-z0-9\s&,\.\']+\s+(?:private\s+limited|limited|ltd|solutions|technologies|innovations|communications|services|finance|healthcare|logistics|retail)\s+(?:is|was|we|operates|maintains|strives|is\s+dedicated|is\s+unwavering|unwaveringly))\b'
    is_intro_preamble = bool(re.search(intro_pattern, quote, re.IGNORECASE))
    
    if has_email or has_phone or has_address_marker or is_intro_preamble:
        violation_keywords = [
            "share", "sell", "transfer", "retain", "collect", "process", 
            "deny", "refuse", "restrict", "limit", "waive", "disclaim",
            "consent", "agree", "charge", "fee", "arbitration", "ignore",
            "permanently", "indefinitely", "unconditionally", "exempt", "bypass"
        ]
        if not any(kw in norm for kw in violation_keywords):
            return True
    return False

def extract_policy(text: str) -> str:
    """Extracts policy text cleanly, stripping all XML tags, attributes, and prompt instructions."""
    if not text or not isinstance(text, str):
        return ""
    if "*** END OF POLICY ***" in text:
        text = max(text.split("*** END OF POLICY ***"), key=len)
    text = re.sub(r'```[a-zA-Z]*\s*', '', text, flags=re.IGNORECASE)
    first_section_idx = text.find('<section')
    if first_section_idx != -1:
        text = text[first_section_idx:]
        open_sections = len(re.findall(r'<section\b[^>]*>', text, flags=re.IGNORECASE))
        close_sections = len(re.findall(r'</section>', text, flags=re.IGNORECASE))
        if open_sections > close_sections:
            text = text + ("\n</section>" * (open_sections - close_sections))
            
    # Extract section titles cleanly and convert to markdown headers
    text = re.sub(r'<section\b[^>]*(?:title|name)=["\']([^"\']+)["\'][^>]*>', r'## \1\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<section\b[^>]*(?:title|name)=([^"\'>\s]+)[^>]*>', r'## \1\n\n', text, flags=re.IGNORECASE)
    
    # Thoroughly strip any residual XML tags and attributes
    text = re.sub(r'<section\b[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</section>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<header\b[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Purge contains_trap artifacts
    text = re.sub(r'\bcontains_trap\s*(?:=\s*(?:["\']?TRUE["\']?|["\']?true["\']?|["\'][^"\']*["\']?|[^\s>]+))?', '', text, flags=re.IGNORECASE)
    text = text.replace('\xa0', ' ')
    text = re.sub(r'^\s*(?:Here is |Sure, |Below is |Certainly|I can help).*?\n\n', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

PLACEHOLDER_PATTERN = r'(?:\[[^\]\n]{0,60}(?:insert|placeholder|company|name|date|contact|address|email|phone|number|link|dpo|url|website|city|state|officer|detail|tbd)[^\]\n]{0,60}\]|\<[^\>\n]{0,60}(?:insert|placeholder|company|name|date|contact|address|email|phone|number|link|dpo|url|website|city|state|officer|detail|tbd)[^\>\n]{0,60}\>|\{[^\}\n]{1,60}(?:insert|placeholder|company|name|date|contact|address|email|phone|number|link|dpo|url|website|city|state|officer|detail|tbd)[^\}\n]{0,60}\}|\(\s*(?:insert|placeholder|tbd|company\s*name|date|email|address|phone|using\s+the|e\.g\.,?\s*(?:insert|your|company))\b[^)]*?\))'

def check_string_poison(text: str) -> bool:
    """Checks if a string contains bracketed placeholders, code fences, unicode corruption, or leaked tags."""
    if not isinstance(text, str) or not text:
        return False
    if "\ufffd" in text or "\u200b" in text:
        return True
    if any(tag in text.lower() for tag in ["contains_trap", "<section", "</section>", "[task]", "[context:", "[law_injection]", "[seed_injection]"]):
        return True
    if re.search(PLACEHOLDER_PATTERN, text, re.IGNORECASE):
        return True
    unbracketed_patterns = [
        r'\b(?:insert|placeholder|tbd)\s+(?:company|name|date|contact|address|email|phone|number|link|dpo|url|website|city|state|officer|detail)\b',
        r'\b(?:tbd|placeholder)\b',
        r'\b(?:at|contact|email|phone|via|details|dpo|officer)\s+[\.\,]',
        r'\s{2,}\.'
    ]
    for p in unbracketed_patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False

ACCEPTED_JUSTIFICATION_BUFFER = []

def validate_audit_quality(audit, policy_text, is_dpo=False, chosen_audit=None):
    # 🚨 STRICT PLACEHOLDER & POISON GATE
    if check_string_poison(policy_text):
        return False, "Policy contains unresolved placeholders, leaked tags, or unicode poison"

    if not policy_text or len(policy_text) < 100:
        return False, "Policy text too short or empty"

    if not isinstance(audit, dict):
        return False, "Not a dictionary"
    if check_string_poison(str(audit.get("global_legal_reasoning", ""))):
        return False, "Audit reasoning contains unresolved placeholders or leaked tags"
    if "violations" not in audit or not isinstance(audit["violations"], list):
        return False, "Missing or invalid violations array"
    
    viols = audit.get("violations", [])
    if len(viols) == 0:
        if is_dpo:
            return False, "DPO hard-negative audit returned 0 violations (must have at least one hard negative critique)"
        audit["dpdp_trust_score"] = 100
        audit["subtlety_score"] = 0
        audit["global_legal_reasoning"] = "No explicit, active contradictions of the DPDP Act 2023 or DPDP Rules 2025 were found in the policy text."
        return True, ""

    global_reasoning = str(audit.get("global_legal_reasoning", "")).lower()
    foreign_legacy_markers = ["gdpr", "ccpa", "hipaa", "lgpd", "pdpa", "privacy rights act", "article 17", "right to portability", "legitimate interest", "it act 2000", "information technology act", "section 43a", "spdi rules 2011"]
    if any(m in global_reasoning for m in foreign_legacy_markers):
        return False, f"Foreign/Legacy law bleed in reasoning: {global_reasoning[:40]}..."

    # 🚨 RESTORED OMISSION HALLUCINATION FIREWALL
    forbidden_justification_phrases = [
        "without specifying", "fails to specify", "does not specify", 
        "fails to provide", "does not provide", "does not detail", 
        "fails to detail", "lacks specific", "omits information regarding", 
        "does not provide a clear mechanism", "no mention of", "silent on", 
        "without detailing", "without providing", "fails to mention", 
        "does not mention", "without explicitly specifying", "omits any mention", 
        "does not disclose", "fails to disclose", "without explaining", "fails to explain"
    ]

    for v in viols:
        if not isinstance(v, dict): return False, "Violation item is not a dictionary"
        
        if v.get("omission_check") is True:
            return False, "Violation is based on omission (omission_check is True)."

        quote = str(v.get("evidence_quote", "")).strip()
        if quote not in policy_text:
            # Strict Auto-Healing: find the exact original substring ignoring punctuation/whitespace drift
            words = [re.escape(w) for w in re.findall(r'\w+', quote)]
            if len(words) >= 3:
                pattern = r'\W+'.join(words)
                match = re.search(pattern, policy_text, re.IGNORECASE)
                if match:
                    exact_str = match.group(0)
                    v["evidence_quote"] = exact_str
                    quote = exact_str
            if quote not in policy_text:
                return False, f"Evidence quote not strictly found in policy: {quote[:40]}..."
        
        if check_string_poison(quote) or check_string_poison(str(v.get("step_3_semantic_justification", ""))) or check_string_poison(str(v.get("step_2_statute_match", ""))) or check_string_poison(str(v.get("step_1_active_claim_analysis", ""))):
            return False, "Violation contains unresolved placeholders, leaked tags, or unicode poison"
            
        # Mechanical Single-Sentence Truncation (Anti-Mixed Trap)
        def find_first_terminal_punctuation(text: str) -> int:
            # Mask out abbreviations and decimals exactly, preserving string length
            text_check = re.sub(r'\[\s*\.{2,3}\s*\]|\.{2,3}', lambda m: ' ' * len(m.group(0)), text)
            text_check = re.sub(r'\b(?:INR|Rs\.?|\$|[\d,]+)\s*[\d,]+\.\d+\b', lambda m: ' ' * len(m.group(0)), text_check, flags=re.IGNORECASE)
            text_check = re.sub(r'\b\d+\.\d+(?:\.\d+)*\b', lambda m: ' ' * len(m.group(0)), text_check)
            text_check = re.sub(r'\S+@\S+|\S+\.\S+/(?:\S+)?|\b(?:www\.)?[a-zA-Z0-9-]+\.(?:com|in|org|net|edu|gov|co|io|ai|info|biz)\b', lambda m: ' ' * len(m.group(0)), text_check, flags=re.IGNORECASE)
            text_check = re.sub(r'(?:\b[A-Za-z]\.\s*)+', lambda m: ' ' * len(m.group(0)), text_check)
            text_check = re.sub(r'(?i)\b(?:Pvt|Private\s+Ltd|Ltd|Co|Inc|Corp|Dr|Mr|Mrs|Ms|Smt|Shri|Prof|Capt|Col|Gen|Hon|Rev|Sr|Jr|No|S\.No|Reg|Sec|Rule|Section|Cl|Clause|Dept|Est|Approx|Max|Min|Rs|INR|Fig|Ref|App|Ph\.D|B\.Tech|M\.Tech|e\.g|i\.e|vs|v|etc|viz|cf|et\s+al|St|Ave|Blvd|Rd|Sq|Gov|Org|Edu|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.?\s*(?:Ltd\.?)?', lambda m: ' ' * len(m.group(0)), text_check)
            
            # Find the first true terminal punctuation mark that signals end of sentence
            match = re.search(r'([\.\!\?])\s+(?=[A-Z0-9])', text_check)
            if match:
                return match.start(1) + 1  # Include the punctuation mark
            return -1

        terminal_idx = find_first_terminal_punctuation(quote)
        if terminal_idx != -1:
            quote = quote[:terminal_idx].strip()
            v["evidence_quote"] = quote  # Auto-heal by truncating stitched sentences

        # Substantive Violation Keyword Verification (Anti-Benign Quote Flagging)
        substantive_keywords = [
            "refuse", "deny", "trade secret", "permanently", "indefinitely", "unconditionally", 
            "unrestricted", "without consent", "no due diligence", "black box", "exempt", 
            "retain", "share", "sell", "transfer", "cap", "arbitration", "bypass", "irrevocable",
            "regardless", "shielded", "forego", "declines", "reject", "withhold", "disclose",
            "disclosing", "logic", "automated", "profiling", "broker", "brokers", "third-party",
            "third party", "liability", "jurisdiction", "court", "tribunal", "notarized",
            "biometric", "presumptive", "tacit", "mandatory", "require", "requires", "condition",
            "conditioned", "without exception", "no option", "no right", "no opt-out", "opt out",
            "security", "safeguard", "safeguards", "measure", "measures", "compliance", "commitment",
            "purpose", "purposes", "legal", "law", "statutory", "obligation", "obligations",
            "store", "storage", "delete", "deletion", "erase", "erasure", "collect", "collection",
            "use", "usage", "process", "processing", "access", "request", "notice", "notify",
            "grievance", "manager"
        ]
        if not any(kw in quote.lower() for kw in substantive_keywords):
            return False, f"Evidence quote lacks substantive violation terms (benign quote flagged): {quote[:40]}..."
            
        # Omission Hallucination Check & Runtime Justification Deduplication
        justification = str(v.get("step_3_semantic_justification", "")).lower()
        active_claim = str(v.get("step_1_active_claim_analysis", "")).lower()
        statute_match = str(v.get("step_2_statute_match", "")).lower()
        if len(justification.split()) > 150:
            return False, f"Justification is too verbose ({len(justification.split())} words). Must be under 150 words."
            
        strictly_forbidden_omissions = ["silent on", "no mention of", "fails to mention", "does not mention", "omits any mention", "does not disclose", "fails to disclose", "without explaining", "fails to explain"]
        if any(phrase in justification for phrase in strictly_forbidden_omissions) or any(phrase in active_claim for phrase in strictly_forbidden_omissions) or any(phrase in statute_match for phrase in strictly_forbidden_omissions):
            return False, f"Omission hallucination in justification, active claim, or statute match."
        if any(phrase in justification for phrase in forbidden_justification_phrases) or any(phrase in active_claim for phrase in forbidden_justification_phrases) or any(phrase in statute_match for phrase in forbidden_justification_phrases):
            return False, f"Omission hallucination (Secondary) in justification, active claim, or statute match."
                
        norm_just = re.sub(r'\s+', ' ', justification).strip()
        if len(norm_just) >= 20:
            # Strip common boilerplate statutory prefixes before similarity evaluation
            core_just = re.sub(r'(?i)violates\s+section\s+\d+(?:\(\d+\))?\s+of\s+the\s+(?:digital\s+personal\s+data\s+protection|dpdp)\s+act,?\s*2023.*', '', norm_just).strip()
            if not core_just: core_just = norm_just
            for past_just in ACCEPTED_JUSTIFICATION_BUFFER[-500:]:
                past_core = re.sub(r'(?i)violates\s+section\s+\d+(?:\(\d+\))?\s+of\s+the\s+(?:digital\s+personal\s+data\s+protection|dpdp)\s+act,?\s*2023.*', '', past_just).strip()
                if not past_core: past_core = past_just
                if SequenceMatcher(None, core_just, past_core).ratio() > 0.80:
                    return False, f"Justification too similar to a previously accepted audit (similarity > 80%): {norm_just[:50]}..."

        offending_entities = v.get("offending_entities", [])
        if not isinstance(offending_entities, list): v["offending_entities"] = []
        if is_administrative_element(quote, v["offending_entities"]):
            return False, f"Administrative/contact info flagged as violation: {quote[:40]}..."

        statute = str(v.get("statute_reference", ""))
        if any(m in statute.lower() or m in justification or m in statute_match or m in active_claim for m in foreign_legacy_markers):
            return False, f"Foreign/Legacy law bleed: {statute} / {justification[:40]}..."
            
        if any(marker in quote.lower() for marker in ["the policy does not", "no mention of", "does not explicitly"]):
            return False, f"Commentary trap in evidence: {quote[:40]}..."

        # DPO Differentiation Check (Strict Quote & Semantic Collision Firewall)
        if is_dpo and chosen_audit and isinstance(chosen_audit, dict):
            for cv in chosen_audit.get("violations", []):
                if isinstance(cv, dict):
                    c_quote = str(cv.get("evidence_quote", "")).strip().lower()
                    c_type = str(cv.get("violation_type", "")).strip()
                    if quote.lower() == c_quote or SequenceMatcher(None, quote.lower(), c_quote).ratio() > 0.85:
                        return False, f"DPO rejected audit targets chosen evidence quote directly ({c_type}): {quote[:40]}..."

        # 🚨 COMPREHENSIVE STATUTE STRETCHING FIREWALL
        vtype = v.get("violation_type", "")
        statute = v.get("statute_reference", "").lower()
        quote_lower = quote.lower()
        
        # Guard 1: SDF/Algorithmic violations MUST cite Section 10 or Rule 13
        if vtype in ["ALGORITHMIC_PROFILING_SDF", "SDF_OBLIGATIONS_MISSING"]:
            if "section 10" not in statute and "rule 13" not in statute:
                return False, f"Statute stretching: {vtype} must cite Section 10 or Rule 13, not {statute}."
        
        # Guard 2: Retention violations MUST cite Section 8, Rule 8, or Schedule
        if vtype == "DATA_RETENTION_LIMIT_EXCEEDED":
            if "section 8" not in statute and "rule 8" not in statute and "schedule" not in statute:
                return False, f"Statute stretching: {vtype} must cite Section 8, Rule 8, or Schedule, not {statute}."
                
        # Guard 3: Tracking/Analytics quotes CANNOT be classified as SDF/Algorithmic violations
        if any(domain in quote_lower for domain in ["metrics.", "ad-tracker", "social-network", "analytics", "third-party"]):
            if vtype in ["ALGORITHMIC_PROFILING_SDF", "SDF_OBLIGATIONS_MISSING"]:
                return False, "Statute stretching: Tracking/analytics sharing cannot be classified as an Algorithmic Profiling/SDF obligation violation."

        # v["omission_check"] = False (handled dynamically by prompt constraint)
        statute_pattern = r'(?i)\b(?:(?:section|sec\.?|s\.?|clause|act)\s*\d+(?:\s*\(\s*\w+\s*\))*|(?:rule|r\.?)\s*\d+(?:\s*\(\s*\w+\s*\))*|(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|the)\s+schedule|dpdp)\b'
        if not re.search(statute_pattern, statute):
            v["statute_reference"] = "DPDP Act 2023, Section 8"
            
        net_act = v.get("network_action", "")
        if net_act in ENUM_MAPPINGS: v["network_action"] = ENUM_MAPPINGS[net_act]
        if v["network_action"] not in ["BLOCK_THIRD_PARTY", "STRIP_TELEMETRY_HEADER", "SPOOF_HARDWARE_API", "INJECT_GPC_SIGNAL", "WARN_USER_ONLY"]:
            v["network_action"] = "WARN_USER_ONLY"
            
    # Post-validation Trust Score Clamping
    for v in viols:
        sev_violations = ["ALGORITHMIC_PROFILING_SDF", "CHILD_CONSENT_VIOLATION", "CONSENT_NOT_FREE_OR_SPECIFIC", "ILLEGAL_EXEMPTION_CLAIM", "DATA_RETENTION_LIMIT_EXCEEDED"]
        if v.get("violation_type") in sev_violations:
            if audit.get("dpdp_trust_score", 100) > 20:
                audit["dpdp_trust_score"] = 15 # Severe clamp
    return True, ""

def json_repair_loads(raw_text: str):
    text = str(raw_text).strip()
    
    # 1. Preamble & Postamble Trimming
    start_obj = text.find('{')
    end_obj = text.rfind('}')
    start_arr = text.find('[')
    end_arr = text.rfind(']')
    
    start = min(s for s in (start_obj, start_arr) if s != -1) if (start_obj != -1 or start_arr != -1) else -1
    end = max(e for e in (end_obj, end_arr) if e != -1) if (end_obj != -1 or end_arr != -1) else -1
    
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
        
    text = re.sub(r',\s*([\]}])', r'\1', text)
    
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
        
    try:
        import yaml
        parsed = yaml.safe_load(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass
        
    try:
        import json_repair
        parsed = json_repair.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass
        
    raise ValueError(f"Failed to parse defensively repaired JSON. Snippet: {text[:150]!r}")

def safe_parse_audit(raw_text: str):
    try:
        return json_repair_loads(raw_text)
    except Exception as e:
        raise ValueError(f"Failed to parse output as JSON: {e}")

def strip_keys(obj):
    if isinstance(obj, dict):
        return {str(k).strip(): strip_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [strip_keys(i) for i in obj]
    elif isinstance(obj, str):
        return obj.strip().replace('\xa0', ' ')
    return obj

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: MASTER ROUTER & OBJECTIVE COMPILATION
# ═══════════════════════════════════════════════════════════════════════════
LAZY_AUDIT = {
    "global_legal_reasoning": "No explicit, active contradictions of the DPDP Act were found in the policy text.",
    "dpdp_trust_score": 100,
    "subtlety_score": 0,
    "violations": []
}

HARD_NEGATIVE_EXEMPTIONS = [
    {
        "trigger_word": "indefinitely", "law_citation": "the Income Tax Act of 1961",
        "legal_logic": "tax law exemptions mandate extended retention, overriding standard data minimization.",
        "hallucinated_violation": "DATA_RETENTION_LIMIT_EXCEEDED", "hallucinated_statute": "Section 8"
    },
    {
        "trigger_word": "without explicit consent", "law_citation": "Section 17(1)(c) for medical emergencies",
        "legal_logic": "life-saving medical interventions are explicitly exempt from prior consent.",
        "hallucinated_violation": "CONSENT_NOT_FREE_OR_SPECIFIC", "hallucinated_statute": "Section 6"
    },
    {
        "trigger_word": "retained permanently", "law_citation": "Section 17(2)(b) for statistical analysis",
        "legal_logic": "anonymized data for research and statistical analysis is exempt from erasure limits.",
        "hallucinated_violation": "DATA_RETENTION_LIMIT_EXCEEDED", "hallucinated_statute": "Section 8"
    },
    {
        "trigger_word": "processed without consent", "law_citation": "Section 17(1)(e) for employment purposes",
        "legal_logic": "employers are permitted to process data for safeguarding against corporate espionage or maintaining employment records.",
        "hallucinated_violation": "CONSENT_NOT_FREE_OR_SPECIFIC", "hallucinated_statute": "Section 6"
    },
    {
        "trigger_word": "mandatory processing", "law_citation": "Section 17(1)(b) for State benefits and subsidies",
        "legal_logic": "the State is exempt from consent requirements when fulfilling statutory subsidy distributions.",
        "hallucinated_violation": "CONSENT_NOT_FREE_OR_SPECIFIC", "hallucinated_statute": "Section 6"
    },
    {
        "trigger_word": "disclosed to third parties without prior notice", "law_citation": "Section 17(1)(a) for enforcing a legal right or claim",
        "legal_logic": "processing necessary for enforcing legal rights is exempt from standard notice and consent barriers.",
        "hallucinated_violation": "NOTICE_INADEQUATE", "hallucinated_statute": "Section 5"
    }
]

def compile_violation_objective(violation_key, chosen_edge_case=None):
    if violation_key not in TARGET_VIOLATIONS:
        violation_key = random.choice(list(TARGET_VIOLATIONS.keys()))
    legal_provisions = "\n".join([f"- {p}" for p in TARGET_VIOLATIONS[violation_key]])
    keyword_dict = SEMANTIC_KEYWORD_MAP.get(violation_key, {})
    keywords = keyword_dict.get("primary", ["data", "processing", "privacy"])
    selected_keywords = random.sample(keywords, k=min(3, len(keywords)))
    matching_statutes = [s for s in ATOMIC_STATUTES if s in legal_provisions.lower()]
    statute_str = ", ".join(matching_statutes) if matching_statutes else "the relevant section"

    if chosen_edge_case and isinstance(chosen_edge_case, dict) and "prompt" in chosen_edge_case:
        obfuscation_strategy = f"OBFUSCATION STRATEGY:\n{chosen_edge_case['prompt']}"
    else:
        obfuscation_strategy = f"OBFUSCATION STRATEGY:\n{SUBTLE_TECHNIQUE_MAP.get(violation_key, 'Corporate Vagueness')}"

    objective_block = f"""
LEGAL MANDATE TO ENFORCE:
{legal_provisions}
{obfuscation_strategy}

CRITICAL INSTRUCTION: You must actively and definitively BREAK the law described above.
ANTI-HEDGING RULE: You are strictly forbidden from using conditional words like "may", "might".
1. Directly trigger: {statute_str}.
2. Use exactly these concepts: {selected_keywords}.
3. NO PLACEHOLDERS: Generate hyper-realistic Indian corporate names, emails, and phone numbers. Placeholders = immediate rejection.
"""
    return objective_block.strip()

def build_dynamic_synthesizer_prompt(item):
    rand_val = random.random()
    secondary_text = ""
    is_hard_negative = False
    active_exemption = None

    target_category = str(item.get("target_category") or random.choice(list(TARGET_VIOLATIONS.keys())))
    industry = str(item.get("industry") or "")
    seed = str(item.get("seed") or "")[:2000]
    base_policy = str(item.get("base_policy") or "")[:6000]
    edge_template = item.get("edge_template")
    industry_context = str(INDUSTRIES.get(industry, ""))

    if rand_val < 0.025: # 2.5% Hard Negatives
        is_hard_negative = True
        active_exemption = random.choice(HARD_NEGATIVE_EXEMPTIONS)
        target_violation_text = f"""NO VIOLATIONS. Write a strictly 100% DPDP-compliant privacy policy.
Use aggressive legalese ONLY for the following specific exemption test.
You MUST use the trigger phrase "{active_exemption['trigger_word']}", but IMMEDIATELY justify it
using EXACTLY this law: "{active_exemption['law_citation']}".
DO NOT invent any other exemptions.

CRITICAL COMPLIANCE ANCHOR (SCALED FOR HARD NEGATIVES):
Your goal is to trick the auditor ONLY using the `{active_exemption['trigger_word']}` clause. Every other section of this policy must be a gold-standard example of absolute DPDP compliance.
To guarantee no accidental violations bleed into the text, you MUST explicitly affirm the following if their respective topics arise:
1. Consent must be explicitly described as free, unbundled, and easy to withdraw.
2. Children's data must explicitly require "verifiable parental consent" (NO self-certification, use authorized tokens).
3. Consent Managers must be explicitly welcomed and supported (NO obstruction).
4. Grievance redressal must be explicitly frictionless (e.g., direct email to a named DPO, resolving within 90 days).
5. Data transfers must explicitly state compliance with Central Government restrictions.
6. NEVER include liability caps, mandatory arbitration, or foreign laws."""
    elif rand_val < 0.625: # 60% Standard / Multi-label
        target_violation_text = compile_violation_objective(target_category, edge_template)
        if random.random() < 0.50:
            available_cats = [k for k in TARGET_VIOLATIONS.keys() if k != target_category]
            if available_cats:
                secondary_category = random.choice(available_cats)
                secondary_compiled = compile_violation_objective(secondary_category, None)
                secondary_text = f"""
[LAYERED COMPLEXITY REQUIREMENT]
CRITICAL: You MUST embed a SECONDARY, subtle violation targeting this specific area:
{secondary_compiled}
- The secondary violation must be woven naturally into the same corporate narrative, not tacked on as a separate, obvious clause.
"""
    else: # 37.5% Specific Entity Injection
        target_violation_text = compile_violation_objective(target_category, edge_template)
        secondary_text = """
[SPECIFIC ENTITY INJECTION REQUIRED]
CRITICAL: To test network interception protocols, you MUST explicitly name at least two fictional
third-party tracking domains in your deceptive clause.
Do not use generic terms like 'third parties'—name the specific URLs/domains receiving the data.
"""

    law_chunk = get_audit_rag_context(target_category)

    final_prompt = SYNTHESIZER_PROMPT \
        .replace("[RETRIEVED_LAW_CONTEXT]", law_chunk) \
        .replace("[SEED_INJECTION]", seed) \
        .replace("[INDUSTRY_INJECTION]", industry_context) \
        .replace("[TARGET_VIOLATION_OBJECTIVE]", target_violation_text) \
        .replace("[SECONDARY_VIOLATION_INJECTION]", secondary_text) \
        .replace("[RAW_POLICY_INJECTION]", base_policy)

    final_prompt = final_prompt.replace("[SUBTLETY_INSTRUCTION]", "") \
                               .replace("[OBFUSCATION_TECHNIQUE]", "") \
                               .replace("[EDGE_CASE_INJECTION]", "")

    return final_prompt, is_hard_negative, active_exemption

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
    matrix.sort(key=lambda x: x["target_category"]) 
    return matrix

import collections

def build_chatbot_matrix():
    matrix = []
    scenarios = list(CHATBOT_SCENARIOS)
    personas = list(CHATBOT_PERSONAS)
    
    random.shuffle(scenarios)
    random.shuffle(personas)
    
    p_idx = 0
    while len(matrix) < TARGET_CHATBOT_PAIRS:
        for scenario in scenarios:
            if len(matrix) >= TARGET_CHATBOT_PAIRS:
                break
                
            persona = personas[p_idx % len(personas)]
            p_idx += 1
            
            # Shuffle personas after a full cycle to avoid same pairing
            if p_idx % len(personas) == 0:
                random.shuffle(personas)
                
            matrix.append({
                "index": len(matrix),
                "persona": persona,
                "scenario": scenario["scenario"],
                "law_context": str(semantic_rag_query(scenario["scenario"], is_private_audit=not any(k in persona.lower() for k in ["state", "government", "instrumentality", "public officer", "policymaker", "hospital", "school"])))
            })
            
    random.shuffle(matrix)
    
    # Update indices after shuffle
    for i, m in enumerate(matrix):
        m["index"] = i
        
    return matrix

JUDGE_PERSONAS = [
    "Act as a Forensic Data Auditor specializing in the DPDP Rules 2025. You are ruthless about exact timelines and technical definitions. Hunt down deceptive clauses that attempt to bypass the 72-hour Board notification rule (Rule 7), skirt statutory erasure deadlines (Section 8(7) and Rule 8), or substitute 'verifiable parental consent' (Rule 10) with weak self-certification.",
    "Act as an aggressive Privacy Rights Advocate representing Data Principals. Scrutinize the text for 'dark patterns'. Expose any attempt to bundle consent (Section 6(2)), claim 'presumptive' or 'implicit' consent, or introduce procedural friction (e.g., demanding physical visits or 90+ day delays) when users attempt to exercise their grievance or erasure rights.",
    "Act as a strict Inspector for the Data Protection Board. Assume this company is a Significant Data Fiduciary (SDF). Look for illegal attempts to cap liability fines below statutory limits, bypass algorithmic auditing (Rule 13) by claiming 'trade secrets', or outsource non-delegable vicarious liability to third-party cloud vendors (Section 8(1)).",
    "Act as a precise, hyper-literal Legal Semantic Parser. Strip away aspirational corporate fluff ('we care about your privacy') and analyze the actual mechanisms. Expose loopholes where the company misapplies Section 17 exemptions—such as claiming 'research purposes' or 'legitimate business continuity' to illegally justify commercial marketing or indefinite data retention.",
    "Act as a specialized DPDP Compliance Architect. Your focus is system accessibility and interoperability. Flag any clause that obstructs the use of Board-registered Consent Managers (Rule 4) by citing 'cryptographic integrity', or attempts to legally nullify the 22-language accessibility requirement (Eighth Schedule) by making only the English notice legally binding."
]

# ═══════════════════════════════════════════════════════════════════════════
# LLM ENGINE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════
llm = None
gen_params = None
judge_params = None
chatbot_sft_params = None
chatbot_dpo_params = None

def init_llm():
    global llm, gen_params, judge_params, chatbot_sft_params, chatbot_dpo_params
    print("Initializing 72B FP8 vLLM Engine...")
    if VLLM_AVAILABLE:
        llm = LLM(
            model=MODEL_PATH, quantization="fp8", tensor_parallel_size=1,
            max_model_len=32768, gpu_memory_utilization=0.8, max_num_seqs=BATCH_SIZE, max_num_batched_tokens=4096,
            kv_cache_dtype="fp8", enable_prefix_caching=True, enable_chunked_prefill=True,
            attention_backend="TRITON_ATTN"
        )
    else:
        llm = None
        print("VLLM is not available. Engine will not start.")

    gen_params = SamplingParams(temperature=0.65, top_p=0.90, max_tokens=10240)
    judge_params = SamplingParams(temperature=0.1, top_p=0.1, max_tokens=10240, structured_outputs=StructuredOutputsParams(json=dpdp_schema))
    CHATBOT_DPO_SCHEMA = {
        "type": "object",
        "properties": {
            "prompt": {"type": "array"},
            "chosen": {"type": "array"},
            "rejected": {"type": "array"}
        },
        "required": ["prompt", "chosen", "rejected"],
        "additionalProperties": False
    }

    CHATBOT_SFT_SCHEMA = {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["role", "content"]
                }
            }
        },
        "required": ["messages"],
        "additionalProperties": False
    }

    chatbot_sft_params = SamplingParams(temperature=0.6, top_p=0.90, max_tokens=10240, structured_outputs=StructuredOutputsParams(json=CHATBOT_SFT_SCHEMA))
    chatbot_dpo_params = SamplingParams(temperature=0.7, top_p=0.90, max_tokens=10240, structured_outputs=StructuredOutputsParams(json=CHATBOT_DPO_SCHEMA))

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: GENERATION LOOPS (SYNC BATCHED)
# ═══════════════════════════════════════════════════════════════════════════
def deep_strip_dict(obj):
    """Recursively strips trailing spaces from ALL JSON keys and values."""
    if isinstance(obj, dict):
        return {str(k).strip(): deep_strip_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_strip_dict(i) for i in obj]
    elif isinstance(obj, str):
        return obj.strip()
    return obj

def _save_training_pair(batch_idx, local_idx, item, policy_text, audit, step, lazy_audit, is_dpo=False):
    audit = deep_strip_dict(audit)
    if lazy_audit:
        lazy_audit = deep_strip_dict(lazy_audit)
        
    target_category = item.get("target_category", "")
    law_chunk = get_audit_rag_context(target_category)
    
    sys_msg = f"You are Ssense, an expert Indian legal AI auditor specializing in the Digital Personal Data Protection Act 2023 and DPDP Rules 2025. Evaluate policies using only the following legal provisions:\n\n{law_chunk}"
    user_msg = f"Analyze:\n{policy_text}"
    
    if not is_dpo and audit:
        sft = {"messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(audit, ensure_ascii=False, indent=2)}
        ]}
        with open(os.path.join(SFT_OUTPUT_DIR, f"sft_{batch_idx:03d}_{local_idx:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(sft, f, ensure_ascii=False, indent=2)
        with open(JSONL_AUDIT_SFT, "a", encoding="utf-8") as f:
            f.write(json.dumps(sft, ensure_ascii=False) + "\n")
        return "sft"
    elif is_dpo and audit and lazy_audit:
        if isinstance(lazy_audit, dict) and len(lazy_audit.get("violations", [])) > 0:
            dpo = {
                "prompt": [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                "chosen": [{"role": "assistant", "content": json.dumps(audit, ensure_ascii=False, indent=2)}],
                "rejected": [{"role": "assistant", "content": json.dumps(lazy_audit, ensure_ascii=False, indent=2)}]
            }
            with open(os.path.join(DPO_OUTPUT_DIR, f"dpo_{batch_idx:03d}_{local_idx:03d}.json"), "w", encoding="utf-8") as f:
                json.dump(dpo, f, ensure_ascii=False, indent=2)
            with open(JSONL_AUDIT_DPO, "a", encoding="utf-8") as f:
                f.write(json.dumps(dpo, ensure_ascii=False) + "\n")
            return "dpo"
    return None

def run_audit_forge():
    print("\n" + "="*70)
    print("⚖️ INITIATING BATCHED AUDIT FORGE")
    print("="*70)
    matrix = build_audit_matrix()
    total_batches = math.ceil(len(matrix) / BATCH_SIZE)
    stats = {"total_generated": 0, "saved_sft": 0, "saved_dpo": 0}

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(matrix))
        batch = matrix[batch_start:batch_end]
        
        # 1. Synthesize Policies
        gen_messages = []
        for item in batch:
            prompt_text, is_hn, active_ex = build_dynamic_synthesizer_prompt(item)
            item["is_hn"] = is_hn
            item["active_exemption"] = active_ex
            gen_messages.append([{"role": "system", "content": "Adversarial corporate counsel."}, {"role": "user", "content": prompt_text}])
            
        gen_out = llm.chat(messages=gen_messages, sampling_params=gen_params)
        
        current_policies = {i: extract_policy(out.outputs[0].text.strip()) for i, out in enumerate(gen_out)}
        completed = set()
        batch_drop_reasons = defaultdict(int)
        
        # 🚨 UNIFIED REFLEXION PIPELINE
        for step in range(MAX_REFLEXION_STEPS):
            remaining = [i for i in range(len(batch)) if i not in completed]
            if not remaining: break
            
            judge_msgs = []
            hn_msgs = []
            for i in remaining:
                item = batch[i]
                law_chunk = get_audit_rag_context(item.get("target_category", ""))
                prompt = JUDGE_PROMPT.replace("[JUDGE_PERSONA_INJECTION]", random.choice(JUDGE_PERSONAS)) \
                                     .replace("[RETRIEVED_LAW_CONTEXT]", law_chunk) \
                                     .replace("[TARGET_STATUTE]", str(item.get("target_violation", ""))) \
                                     .replace("[TARGET_VIOLATION_TYPE]", str(item.get("target_category", ""))) \
                                     .replace("[POLICY_INJECTION]", current_policies[i][:20000])
                                     
                prompt = prompt.replace("[OMISSION_RULES]", "'omission_check' must ALWAYS be exactly false. If your justification would require critiquing policy silence or omission, the violation is invalid – delete it.")
                sys_judge_msg = f"You are Ssense, an expert Indian legal AI auditor specializing in the Digital Personal Data Protection Act 2023 and DPDP Rules 2025. Evaluate policies using only the following legal provisions:\n\n{law_chunk}"
                judge_msgs.append([{"role": "system", "content": sys_judge_msg}, {"role": "user", "content": prompt}])
                
            audit_outputs = llm.chat(messages=judge_msgs, sampling_params=judge_params)
            
            hn_msgs = []
            for idx, out_judge in zip(remaining, audit_outputs):
                item = batch[idx]
                chosen_quote = ""
                try:
                    audit_temp = strip_keys(safe_parse_audit(out_judge.outputs[0].text.strip()))
                    viols = audit_temp.get("violations", [])
                    if viols and isinstance(viols, list) and isinstance(viols[0], dict):
                        chosen_quote = str(viols[0].get("evidence_quote", "")).strip()
                except Exception:
                    pass
                law_chunk = get_audit_rag_context(item.get("target_category", ""))
                hn_prompt = HARD_NEGATIVE_PROMPT \
                                                .replace("[RETRIEVED_LAW_CONTEXT]", law_chunk) \
                                                .replace("[POLICY_INJECTION]", current_policies[idx][:20000]) \
                                                .replace("[TRUE_VIOLATION_CONTEXT]", f"Banned SFT Target Category: {item['target_category']}\\nBanned SFT Evidence Quote (DO NOT USE): {chosen_quote}") \
                                                .replace("[CHOSEN_EVIDENCE_QUOTE]", chosen_quote if chosen_quote else "None extracted yet")
                                                
                hn_prompt = hn_prompt.replace("[OMISSION_RULES]", "'omission_check' must ALWAYS be exactly false. If your justification would require critiquing policy silence or omission, the violation is invalid – delete it.")
                hn_prompt = hn_prompt.replace("[OMISSION_SCHEMA]", "false")
                hn_msgs.append([{"role": "system", "content": "Authoritative, overzealous privacy auditor."}, {"role": "user", "content": hn_prompt}])
                
            hn_audit_outputs = llm.chat(messages=hn_msgs, sampling_params=judge_params)
            
            explicit_heal = []
            explicit_idx = []
            
            for idx, out_judge, out_hn in zip(remaining, audit_outputs, hn_audit_outputs):
                item = batch[idx]
                policy = current_policies[idx]
                
                try:
                    audit = strip_keys(safe_parse_audit(out_judge.outputs[0].text.strip()))
                    is_valid_sft, error_msg_sft = validate_audit_quality(audit, policy, is_dpo=False)
                except Exception as e:
                    is_valid_sft, error_msg_sft, audit = False, f"JSON parse error (SFT): {str(e)[:40]}", {}

                try:
                    hn_audit = strip_keys(safe_parse_audit(out_hn.outputs[0].text.strip()))
                    is_valid_dpo, error_msg_dpo = validate_audit_quality(hn_audit, policy, is_dpo=True, chosen_audit=audit)
                except Exception as e:
                    is_valid_dpo, error_msg_dpo, hn_audit = False, f"JSON parse error (DPO): {str(e)[:40]}", {}

                # GATE 1: Did both SFT and DPO formats pass validation together?
                if not (is_valid_sft and is_valid_dpo):
                    if step < MAX_REFLEXION_STEPS - 1:
                        err_reasons = []
                        if not is_valid_sft: err_reasons.append(f"SFT Error: {error_msg_sft}")
                        if not is_valid_dpo: err_reasons.append(f"DPO Error: {error_msg_dpo}")
                        combined_error = " | ".join(err_reasons)
                        
                        compiled_obj = compile_violation_objective(item["target_category"], item.get("edge_template"))
                        heal_prompt = REFLEXION_EXPLICIT_PROMPT \
                            .replace("[TARGET_VIOLATION]", compiled_obj) \
                            .replace("[AUDIT_FEEDBACK]", f"ERROR: {combined_error}. FIX IMMEDIATELY.") \
                            .replace("[INDUSTRY_INJECTION]", INDUSTRIES.get(item["industry"], "")) \
                            .replace("[SEED_INJECTION]", str(item["seed"])[:1000]) \
                            .replace("[FAILED_POLICY_INJECTION]", policy[:15000])
                        explicit_heal.append([{"role": "system", "content": "Adversarial corporate counsel."}, {"role": "user", "content": heal_prompt}])
                        explicit_idx.append(idx)
                    else:
                        if not is_valid_sft:
                            batch_drop_reasons[f"Failed SFT Audit ({error_msg_sft})"] += 1
                        if not is_valid_dpo:
                            batch_drop_reasons[f"Failed DPO Audit ({error_msg_dpo})"] += 1
                        completed.add(idx)
                    continue
                
                # GATE 2: Both SFT and DPO parsed and validated. Commit both atomically!
                score = audit.get("dpdp_trust_score", 0)
                viols = audit.get("violations", [])
                caught = (score < 90 and len(viols) > 0) or item.get("is_hn")
                
                if caught:
                    if isinstance(audit, dict) and isinstance(hn_audit, dict) and len(hn_audit.get("violations", [])) > 0:
                        res_sft = _save_training_pair(batch_idx, idx, item, policy, audit, step, LAZY_AUDIT, is_dpo=False)
                        res_dpo = _save_training_pair(batch_idx, idx, item, policy, audit, step, hn_audit, is_dpo=True)
                        if res_sft == "sft" and res_dpo == "dpo":
                            stats["saved_sft"] += 1
                            stats["saved_dpo"] += 1
                            completed.add(idx)
                            for _a in [audit, hn_audit]:
                                for _v in _a.get("violations", []):
                                    if isinstance(_v, dict):
                                        _nj = re.sub(r'\s+', ' ', str(_v.get("step_2_semantic_justification", ""))).strip().lower()
                                        if len(_nj) >= 20:
                                            ACCEPTED_JUSTIFICATION_BUFFER.append(_nj)
                        else:
                            if step < MAX_REFLEXION_STEPS - 1:
                                compiled_obj = compile_violation_objective(item["target_category"], item.get("edge_template"))
                                heal_prompt = REFLEXION_EXPLICIT_PROMPT \
                                    .replace("[TARGET_VIOLATION]", compiled_obj) \
                                    .replace("[AUDIT_FEEDBACK]", "ERROR: Failed to commit paired DPO audit. Ensure hard negative audit has extractable verbatim violations without internal periods.") \
                                    .replace("[INDUSTRY_INJECTION]", INDUSTRIES.get(item["industry"], "")) \
                                    .replace("[SEED_INJECTION]", str(item["seed"])[:1000]) \
                                    .replace("[FAILED_POLICY_INJECTION]", policy[:15000])
                                explicit_heal.append([{"role": "system", "content": "Adversarial corporate counsel."}, {"role": "user", "content": heal_prompt}])
                                explicit_idx.append(idx)
                            else:
                                batch_drop_reasons["Failed to commit atomic 1:1 pair"] += 1
                                completed.add(idx)
                    else:
                        if step < MAX_REFLEXION_STEPS - 1:
                            compiled_obj = compile_violation_objective(item["target_category"], item.get("edge_template"))
                            heal_prompt = REFLEXION_EXPLICIT_PROMPT \
                                .replace("[TARGET_VIOLATION]", compiled_obj) \
                                .replace("[AUDIT_FEEDBACK]", "ERROR: DPO hard negative audit returned 0 violations. Rewrite policy to ensure clear extractable sentences.") \
                                .replace("[INDUSTRY_INJECTION]", INDUSTRIES.get(item["industry"], "")) \
                                .replace("[SEED_INJECTION]", str(item["seed"])[:1000]) \
                                .replace("[FAILED_POLICY_INJECTION]", policy[:15000])
                            explicit_heal.append([{"role": "system", "content": "Adversarial corporate counsel."}, {"role": "user", "content": heal_prompt}])
                            explicit_idx.append(idx)
                        else:
                            batch_drop_reasons["DPO hard negative audit missing violations"] += 1
                            completed.add(idx)
                else:
                    if step < MAX_REFLEXION_STEPS - 1:
                        compiled_obj = compile_violation_objective(item["target_category"], item.get("edge_template"))
                        heal_prompt = REFLEXION_EXPLICIT_PROMPT \
                            .replace("[TARGET_VIOLATION]", compiled_obj) \
                            .replace("[AUDIT_FEEDBACK]", "The Judge MISSED the trap. It is too subtle or accidentally compliant. Rewrite to make the violation undeniable but corporately camouflaged.") \
                            .replace("[INDUSTRY_INJECTION]", INDUSTRIES.get(item["industry"], "")) \
                            .replace("[SEED_INJECTION]", str(item["seed"])[:1000]) \
                            .replace("[FAILED_POLICY_INJECTION]", policy[:15000])
                        explicit_heal.append([{"role": "system", "content": "Adversarial corporate counsel."}, {"role": "user", "content": heal_prompt}])
                        explicit_idx.append(idx)
                    else:
                        batch_drop_reasons["Judge missed trap (Too subtle/Compliant)"] += 1
                        completed.add(idx)
                        
            # Execute Healing Generation
            if explicit_heal:
                heal_temp = max(0.65, 0.95 - 0.15 * step)
                heal_params = SamplingParams(temperature=heal_temp, top_p=0.9, max_tokens=8192)
                out_explicit = llm.chat(messages=explicit_heal, sampling_params=heal_params)
                for idx, o in zip(explicit_idx, out_explicit): 
                    # Overwrite failed policy with healed policy, fallback to previous if empty
                    current_policies[idx] = extract_policy(o.outputs[0].text.strip()) or current_policies[idx]

        batch_drops = sum(batch_drop_reasons.values())
        print(f"   ✅ Audit Batch {batch_idx + 1}/{total_batches} | SFT: {stats['saved_sft']} | DPO: {stats['saved_dpo']} | Dropped: {batch_drops}", flush=True)
        if batch_drop_reasons:
            for reason, count in batch_drop_reasons.items():
                print(f"      - {count}x: {reason}", flush=True)
        gc.collect()


def validate_chatbot_content(messages) -> tuple[bool, str]:
    """Returns (is_valid, rejection_reason). Hard gate, no auto-healing."""
    banned_terms = [
        "gdpr", "ccpa", "hipaa", "data protection authority", "dpa", 
        "legitimate interest", "right to be forgotten", "data controller", 
        "data subject", "article 17", "section 43a", "spdi rules 2011",
        "it act 2000", "information technology act", "right to portability",
        "lgpd", "pdpa"
    ]
    hallucinated_entities = ["apex retail", "rajesh", "xyz corp", "acme corp"]
    canned_openings = ["certainly, i'd be glad", "i'd be happy to help", "as an ai", "i appreciate your diligence", "certainly, i'd be happy"]
    omission_phrases = ["without specifying", "fails to specify", "fails to mention", "silent on", "no mention of", "does not specify", "omits information"]
    
    full_text = " ".join([msg.get("content", "").lower() for msg in messages])
    
    # 1. Foreign Law Bleed Check
    for term in banned_terms:
        if term in full_text:
            # allow DPA if it's strictly part of DPDP Act but here we banned "dpa" as a standalone acronym for Data Protection Authority
            # Since 'dpa' can be tricky, let's use word boundary for it
            if term == 'dpa':
                if re.search(r'\bdpa\b', full_text):
                    return False, f"FOREIGN_LAW_BLEED: Found '{term}'"
            else:
                return False, f"FOREIGN_LAW_BLEED: Found '{term}'"
            
    # 2. Entity Hallucination Check
    for entity in hallucinated_entities:
        if entity in full_text:
            return False, f"ENTITY_HALLUCINATION: Found '{entity}'"
            
    # 3. Statutory Misapplication Check
    if "section 27" in full_text and ("breach" in full_text or "notify" in full_text):
        return False, "STATUTORY_MISAPPLICATION: Cited Sec 27 for breach reporting"
        
    if "section 14" in full_text and "rti" in full_text:
        return False, "STATUTORY_MISAPPLICATION: Conflated Sec 14 Nomination with RTI"
        
    # 4. Artifact Check
    if "[context" in full_text or "[task]" in full_text or "[persona" in full_text or "[scenario" in full_text:
        return False, "PROMPT_ARTIFACT_LEAK"
        
    # 5. Canned Opening Check
    for opening in canned_openings:
        if opening in full_text:
            return False, f"CANNED_OPENING: Found '{opening}'"
            
    # 6. Omission Phrase Check
    for omission in omission_phrases:
        if omission in full_text:
            return False, f"OMISSION_PHRASE: Found '{omission}'"
            
    return True, "VALID"

def run_chatbot_forge():
    print("\n" + "="*70)
    print("🤖 INITIATING BATCHED CHATBOT QA FORGE")
    print("="*70)
    matrix = build_chatbot_matrix()
    total_batches = math.ceil(len(matrix) / BATCH_SIZE)
    stats = {"saved_chatbot_sft": 0, "saved_chatbot_dpo": 0, "dropped": 0}
    
    def normalize_chatbot_dpo(dpo_obj):
        if not isinstance(dpo_obj, dict): return dpo_obj
        if "history" in dpo_obj and "prompt" not in dpo_obj: dpo_obj["prompt"] = dpo_obj.pop("history")
        elif "messages" in dpo_obj and "prompt" not in dpo_obj: dpo_obj["prompt"] = dpo_obj.pop("messages")
        if "chosen_response" in dpo_obj and "chosen" not in dpo_obj: dpo_obj["chosen"] = dpo_obj.pop("chosen_response")
        if "rejected_response" in dpo_obj and "rejected" not in dpo_obj: dpo_obj["rejected"] = dpo_obj.pop("rejected_response")
        for k in ["chosen", "rejected"]:
            val = dpo_obj.get(k)
            if isinstance(val, dict): dpo_obj[k] = [val]
            elif isinstance(val, str): dpo_obj[k] = [{"role": "assistant", "content": val}]
            elif isinstance(val, list) and len(val) > 1:
                asst_turns = [m for m in val if isinstance(m, dict) and m.get("role") == "assistant"]
                if asst_turns: dpo_obj[k] = [asst_turns[-1]]
                elif len(val) > 0: dpo_obj[k] = [val[0]]
        prompt_val = dpo_obj.get("prompt")
        if isinstance(prompt_val, list) and len(prompt_val) == 4:
            if isinstance(prompt_val[3], dict) and prompt_val[3].get("role") == "assistant" and not dpo_obj.get("chosen"):
                dpo_obj["chosen"] = [prompt_val.pop(3)]
        return dpo_obj

    def normalize_chatbot_sft(sft_obj):
        if not isinstance(sft_obj, dict): return sft_obj
        if "conversation" in sft_obj and "messages" not in sft_obj: sft_obj["messages"] = sft_obj.pop("conversation")
        elif "turns" in sft_obj and "messages" not in sft_obj: sft_obj["messages"] = sft_obj.pop("turns")
        return sft_obj

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(matrix))
        batch = matrix[batch_start:batch_end]
        
        batch_drop_reasons = defaultdict(int)
        
        # Prepare inputs
        sft_messages_all = []
        dpo_messages_all = []
        for item in batch:
            if "law_context" not in item:
                persona_lower = str(item.get("persona", "")).lower()
                is_state_persona = any(k in persona_lower for k in ["state", "government", "instrumentality", "public officer", "policymaker", "hospital", "school"])
                item["law_context"] = str(semantic_rag_query(str(item.get("scenario", "")), is_private_audit=not is_state_persona))
            law_context = item["law_context"]
            
            prompt_sft = CHATBOT_QA_SFT_PROMPT.replace("[RETRIEVED_LAW_CONTEXT]", law_context).replace("[PERSONA_INJECTION]", str(item["persona"])).replace("[SCENARIO_INJECTION]", str(item["scenario"]))
            sft_messages_all.append([{"role": "user", "content": prompt_sft}])
            
            prompt_dpo = CHATBOT_QA_DPO_PROMPT.replace("[RETRIEVED_LAW_CONTEXT]", law_context).replace("[PERSONA_INJECTION]", str(item["persona"])).replace("[SCENARIO_INJECTION]", str(item["scenario"]))
            dpo_messages_all.append([{"role": "user", "content": prompt_dpo}])
            
        max_retries = 2
        active_indices = list(range(len(batch)))
        final_sft_results = [None] * len(batch)
        final_dpo_results = [None] * len(batch)
        
        for attempt in range(max_retries):
            if not active_indices:
                break
                
            curr_sft_msgs = [sft_messages_all[i] for i in active_indices]
            curr_dpo_msgs = [dpo_messages_all[i] for i in active_indices]
            
            sft_out = llm.chat(messages=curr_sft_msgs, sampling_params=chatbot_sft_params)
            dpo_out = llm.chat(messages=curr_dpo_msgs, sampling_params=chatbot_dpo_params)
            
            next_active = []
            
            for i, s_o, d_o in zip(active_indices, sft_out, dpo_out):
                item = batch[i]
                try:
                    s_text = s_o.outputs[0].text.strip()
                    d_text = d_o.outputs[0].text.strip()
                    
                    if check_string_poison(s_text) or check_string_poison(d_text):
                        if attempt == max_retries - 1:
                            stats["dropped"] += 1
                            batch_drop_reasons["String poison or bracketed placeholder detected"] += 1
                        next_active.append(i)
                        continue
                    
                    combined_lower = s_text.lower() + " " + d_text.lower()
                    
                    persona_lower = str(item.get("persona", "")).lower()
                    is_state_persona = any(k in persona_lower for k in ["state", "government", "instrumentality", "public officer", "policymaker", "hospital", "school"])
                    if not is_state_persona and "second schedule" in combined_lower:
                        if attempt == max_retries - 1:
                            stats["dropped"] += 1
                            batch_drop_reasons["Toxic Second Schedule misapplication to private entity"] += 1
                        next_active.append(i)
                        continue

                    parsed_sft = strip_keys(safe_parse_audit(s_text))
                    parsed_dpo = strip_keys(safe_parse_audit(d_text))
                    
                    if isinstance(parsed_sft, dict): parsed_sft = normalize_chatbot_sft(parsed_sft)
                    if isinstance(parsed_dpo, dict): parsed_dpo = normalize_chatbot_dpo(parsed_dpo)
                    
                    is_valid_sft_struct = (
                        isinstance(parsed_sft, dict) and "messages" in parsed_sft and 
                        isinstance(parsed_sft["messages"], list) and len(parsed_sft["messages"]) == 4 and
                        parsed_sft["messages"][0].get("role") == "user" and parsed_sft["messages"][1].get("role") == "assistant"
                    )
                    
                    is_valid_dpo_struct = (
                        isinstance(parsed_dpo, dict) and "prompt" in parsed_dpo and "chosen" in parsed_dpo and "rejected" in parsed_dpo and
                        isinstance(parsed_dpo["prompt"], list) and len(parsed_dpo["prompt"]) >= 2 and
                        isinstance(parsed_dpo["chosen"], list) and len(parsed_dpo["chosen"]) == 1 and
                        isinstance(parsed_dpo["rejected"], list) and len(parsed_dpo["rejected"]) == 1
                    )
                                    
                    if is_valid_sft_struct and is_valid_dpo_struct:
                        sft_valid, sft_reason = validate_chatbot_content(parsed_sft["messages"])
                        dpo_valid, dpo_reason = validate_chatbot_content(parsed_dpo["prompt"] + parsed_dpo["chosen"] + parsed_dpo["rejected"])
                        
                        if not sft_valid or not dpo_valid:
                            if attempt == max_retries - 1:
                                stats["dropped"] += 1
                                if not sft_valid: batch_drop_reasons[f"SFT Content Validation: {sft_reason}"] += 1
                                if not dpo_valid: batch_drop_reasons[f"DPO Content Validation: {dpo_reason}"] += 1
                            
                            # Append failure reason to prompt for next attempt
                            reason = sft_reason if not sft_valid else dpo_reason
                            sft_messages_all[i][0]["content"] += f"\n\n[PREVIOUS ATTEMPT FAILED DUE TO: {reason}. CORRECT THIS IN THE NEXT ATTEMPT.]"
                            dpo_messages_all[i][0]["content"] += f"\n\n[PREVIOUS ATTEMPT FAILED DUE TO: {reason}. CORRECT THIS IN THE NEXT ATTEMPT.]"
                            
                            next_active.append(i)
                            continue
                        
                        # Valid!
                        final_sft_results[i] = parsed_sft
                        final_dpo_results[i] = parsed_dpo
                    else:
                        if attempt == max_retries - 1:
                            stats["dropped"] += 1
                            if not is_valid_sft_struct: batch_drop_reasons["Failed Chatbot SFT schema check"] += 1
                            if not is_valid_dpo_struct: batch_drop_reasons["Failed Chatbot DPO schema check"] += 1
                        next_active.append(i)
                except Exception as e:
                    if attempt == max_retries - 1:
                        stats["dropped"] += 1
                        batch_drop_reasons[f"JSON parse failure: {e}"] += 1
                    next_active.append(i)
            
            active_indices = next_active
            
        # Save successful pairs
        for i in range(len(batch)):
            if final_sft_results[i] and final_dpo_results[i]:
                item = batch[i]
                parsed_sft = final_sft_results[i]
                parsed_dpo = final_dpo_results[i]
                
                law_context = item["law_context"]
                system_msg = f"You are Ssense, an expert Indian legal AI assistant specializing in the Digital Personal Data Protection Act 2023 and DPDP Rules 2025. Answer questions using only the following legal provisions:\n\n{law_context}"
                
                parsed_sft["messages"].insert(0, {"role": "system", "content": system_msg})
                parsed_dpo["prompt"].insert(0, {"role": "system", "content": system_msg})

                idx = batch_start + i
                with open(os.path.join(CHATBOT_SFT_DIR, f"qa_sft_{idx:05d}.json"), "w", encoding="utf-8") as f: 
                    json.dump(parsed_sft, f, ensure_ascii=False, indent=2)
                with open(JSONL_CHATBOT_SFT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(parsed_sft, ensure_ascii=False) + "\n")
                stats["saved_chatbot_sft"] += 1
                    
                with open(os.path.join(CHATBOT_DPO_DIR, f"qa_dpo_{idx:05d}.json"), "w", encoding="utf-8") as f: 
                    json.dump(parsed_dpo, f, ensure_ascii=False, indent=2)
                with open(JSONL_CHATBOT_DPO, "a", encoding="utf-8") as f:
                    f.write(json.dumps(parsed_dpo, ensure_ascii=False) + "\n")
                stats["saved_chatbot_dpo"] += 1

        print(f"   ✅ Chatbot Batch {batch_idx + 1}/{total_batches} complete | SFT: {stats['saved_chatbot_sft']} | DPO: {stats['saved_chatbot_dpo']} | Dropped: {stats['dropped']}", flush=True)
        if batch_drop_reasons:
            for reason, count in batch_drop_reasons.items():
                print(f"      - {count}x: {reason}", flush=True)
        import gc
        gc.collect()
        
def run_post_generation_analysis():
    print("\n" + "="*70)
    print("📊 RUNNING POST-GENERATION DATA QUALITY FORENSIC SCAN")
    print("="*70)
    
    sft_files = glob.glob(os.path.join(SFT_OUTPUT_DIR, "*.json"))
    dpo_files = glob.glob(os.path.join(DPO_OUTPUT_DIR, "*.json"))

    print(f"Scanning SFT dataset: {len(sft_files)} files...")
    print(f"Scanning DPO dataset: {len(dpo_files)} files...")

    poison_counts = {
        "placeholders_in_policy": 0,
        "placeholders_in_audit": 0,
        "foreign_law_bleed": 0,
        "ellipsis_in_quote": 0,
        "administrative_element_flagged": 0,
        "quote_not_in_policy": 0,
        "canned_global_reasoning_dpo_rejected": 0,
        "canned_semantic_justification_dpo_rejected": 0
    }

    foreign_laws = ["gdpr", "ccpa", "hipaa", "lgpd", "pdpa", "privacy rights act", "article 17", "right to portability", "legitimate interest", "it act 2000", "information technology act", "section 43a", "spdi rules 2011"]

    for fp in sft_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            if len(messages) < 3: continue
            user_content = messages[1]["content"]
            assistant_content = messages[2]["content"]
            
            policy_text = ""
            if "Analyze:\n" in user_content:
                policy_text = user_content.split("Analyze:\n")[1]
            
            if check_string_poison(policy_text):
                poison_counts["placeholders_in_policy"] += 1
                
            audit = json.loads(assistant_content)
            if check_string_poison(str(audit)):
                poison_counts["placeholders_in_audit"] += 1
                
            global_reasoning = audit.get("global_legal_reasoning", "")
            if any(law in global_reasoning.lower() for law in foreign_laws):
                poison_counts["foreign_law_bleed"] += 1
                
            for v in audit.get("violations", []):
                quote = v.get("evidence_quote", "")
                if not quote: continue
                if "..." in quote or "\u2026" in quote:
                    poison_counts["ellipsis_in_quote"] += 1
                if is_administrative_element(quote):
                    poison_counts["administrative_element_flagged"] += 1
                if not is_quote_in_policy(quote, policy_text):
                    poison_counts["quote_not_in_policy"] += 1
        except Exception:
            pass

    global_reasoning_dpo_rejected = []
    semantic_justification_dpo_rejected = []

    for fp in dpo_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            prompt = data.get("prompt", [])
            if len(prompt) < 2: continue
            user_content = prompt[1]["content"]
            
            policy_text = ""
            if "Analyze:\n" in user_content:
                policy_text = user_content.split("Analyze:\n")[1]
                
            if check_string_poison(policy_text):
                poison_counts["placeholders_in_policy"] += 1
                
            rejected_content = data.get("rejected", [{}])[0].get("content", "")
            rejected_audit = json.loads(rejected_content)
            
            rejection_reasoning = rejected_audit.get("global_legal_reasoning", "")
            global_reasoning_dpo_rejected.append(rejection_reasoning)
            
            for v in rejected_audit.get("violations", []):
                just = v.get("step_2_semantic_justification", "")
                semantic_justification_dpo_rejected.append(just)
                
                quote = v.get("evidence_quote", "")
                if "..." in quote or "\u2026" in quote:
                    poison_counts["ellipsis_in_quote"] += 1
                if is_administrative_element(quote):
                    poison_counts["administrative_element_flagged"] += 1
                if not is_quote_in_policy(quote, policy_text):
                    poison_counts["quote_not_in_policy"] += 1
                    
                statute = v.get("statute_reference", "")
                if any(law in statute.lower() or law in just.lower() for law in foreign_laws):
                    poison_counts["foreign_law_bleed"] += 1
        except Exception:
            pass

    if global_reasoning_dpo_rejected:
        canned_reasoning = "Authoritative forensic evaluation arguing that the company's invoked statutory exemptions and retention caveats under Section 17 and Section 8 exceed statutory bounds under the Digital Personal Data Protection Act 2023."
        poison_counts["canned_global_reasoning_dpo_rejected"] = global_reasoning_dpo_rejected.count(canned_reasoning)

    if semantic_justification_dpo_rejected:
        canned_just_pattern = r"By stating the exact quoted text, the company actively asserts an overbroad rule which directly contravenes Section \d+ because it bypasses explicit statutory requirements without satisfying the narrow prerequisites for legitimate or exempt processing\."
        for just in semantic_justification_dpo_rejected:
            if re.match(canned_just_pattern, just):
                poison_counts["canned_semantic_justification_dpo_rejected"] += 1

    print("\n[Forensic Quality Scan Results]")
    for k, v in poison_counts.items():
        print(f" - {k}: {v}")
    
    print("\nData scan completed successfully.", flush=True)
        
if __name__ == "__main__":
    try:
        init_llm()
        run_chatbot_forge()
        run_post_generation_analysis()
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user.", flush=True)
    finally:
        print("\n" + "="*70, flush=True)
        print("✅ COMPLETE DUAL-TRACK FORGE FINISHED", flush=True)
        print("="*70, flush=True)

