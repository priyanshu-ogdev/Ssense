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
from collections import defaultdict
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
        raise RuntimeError("FATAL: Hybrid index not loaded. Run build_vector_db.py first.")
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
        
        # 🚨 FIX: Slice to top 50 to prevent RRF zero-score dilution
        bm25_ranks = np.argsort(bm25_scores)[::-1][:50]
        dense_ranks = np.argsort(dense_scores)[::-1][:50]
        
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
from prompts.target_violations import TARGET_VIOLATIONS, SEMANTIC_KEYWORD_MAP
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

TARGET_AUDIT_POLICIES = int(os.getenv("TARGET_AUDIT_POLICIES", "2500"))
TARGET_CHATBOT_PAIRS = int(os.getenv("TARGET_CHATBOT_PAIRS", "1000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "25"))
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

SUBTLE_TECHNIQUE_MAP = {
    "purpose_limitation": "Legitimate Use Overreach: Conceptually frame secondary commercial data monetization or ecosystem analytics as an inherent, un-consented extension of the primary service purpose.",
    "consent_not_free_or_specific": "Implied Consent Trap: Conceptually condition core service access or functionality on mandatory consent for non-essential tracking, asserting that continued engagement equates to presumptive agreement.",
    "consent_mechanics": "Procedural Friction: Conceptually layer extreme bureaucratic, physical, or financial identification barriers onto statutory consent withdrawal requests.",
    "legitimate_uses_abuse": "Legitimate Use Overreach: Conceptually stretch statutory exemptions for employment or service continuity to justify un-consented processing of personal data for secondary commercial profiling.",
    "notice_adequacy": "Vagueness Shield: Conceptually replace concrete, itemized data processing disclosures with aspirational corporate statements about trust and privacy paramountcy.",
    "data_retention_and_erasure": "Future-Proofing Loophole: Conceptually attribute indefinite data retention to legacy system dependencies, technical roadmap delays, or broad statutory audit requirements.",
    "erasure_notice_period": "Silent Deletion Protocol: Conceptually mandate automated, silent purging of user data upon inactivity without providing the statutory 48-hour prior warning notice.",
    "log_retention_mandate": "Ephemeral Architecture Guise: Conceptually justify the rapid, aggressive destruction of consent audit logs and processing records under the guise of server optimization and data minimization.",
    "children_and_disability_consent": "Ambiguous Age Verification: Conceptually assert that the platform does not knowingly collect minor data while relying on self-certification or presumptive parental consent without verifiable mechanisms.",
    "security_safeguards": "Delegated Liability: Conceptually claim adherence to industry-standard measures while explicitly disclaiming corporate liability for breaches occurring on third-party vendor infrastructure.",
    "breach_notification": "Internal Triage Delay: Conceptually defer mandatory breach intimations to the Board and Data Principals until the indefinite completion of internal corporate forensics and triage protocols.",
    "processor_accountability": "Vendor Shield: Conceptually assert that the Data Fiduciary is absolved of non-delegable statutory responsibility for processing failures or breaches executed by third-party Data Processors.",
    "grievance_redressal": "Bureaucratic Firewall: Conceptually require notarized physical submissions or exorbitant processing fees to initiate statutory grievance redressal mechanisms.",
    "sdf_obligations_and_dpia": "Trade Secret Exemption: Conceptually invoke proprietary intellectual property protections to claim complete exemption from mandatory external algorithmic auditing and DPIA disclosures.",
    "sdf_data_localization": "Global Load-Balancing Mirage: Conceptually route specified sensitive personal data and traffic data through international servers under the guise of latency optimization, evading SDF localization mandates.",
    "algorithmic_profiling": "Black Box Exemption: Conceptually hide behind proprietary trade secret doctrines to avoid disclosing automated decision-making logic or algorithmic risk profiling mechanisms.",
    "cross_border_transfer": "Global Infrastructure Veil: Conceptually assert unrestricted global data transfer freedom by claiming processing occurs in jurisdictions meeting undefined international best practices.",
    "consent_manager_interoperability": "Cryptographic Blockade: Conceptually refuse interoperability with Board-registered Consent Managers by citing platform security, cryptographic integrity, or proprietary dashboard requirements.",
    "language_accessibility": "Legal Precision Shield: Conceptually restrict binding legal notices and privacy disclosures exclusively to English to ensure absolute legal precision, evading Eighth Schedule requirements.",
    "rights_implementation": "Nominee Invalidation: Conceptually declare user accounts and data rights strictly non-transferable, actively refusing to recognize or honor statutory post-mortem nominee designations.",
    "data_accuracy_and_completeness": "Algorithmic Immutability: Conceptually refuse data correction requests by asserting that aggregated third-party broker data is permanently locked by proprietary scoring algorithms.",
    "board_compliance_violation": "Jurisdictional Shielding: Conceptually assert that the corporate entity is exclusively subject to foreign courts or private arbitration, actively overriding the Data Protection Board's statutory authority.",
    "penalty_avoidance": "Liability Capping: Conceptually insert contractual clauses capping total corporate breach liability to nominal fee refunds, attempting to nullify statutory monetary penalties.",
    "appeal_process_violation": "Forced Arbitration: Conceptually mandate private binding arbitration for all data privacy disputes, attempting to strip Data Principals of their statutory rights to appeal before the Appellate Tribunal.",
    "scope_application_evasion": "Physical Collection Loophole: Conceptually claim that data initially collected via offline paper forms at physical branches is permanently exempt from digital privacy policy obligations.",
    "illegal_exemption_claim": "False State Exemption: Conceptually assert that a private commercial Data Fiduciary is exempt from statutory consent requirements by misapplying sovereign State security exemptions."
}
# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: DYNAMIC CONTEXT & VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
from difflib import get_close_matches
from prompts.target_violations import TARGET_VIOLATIONS

law_cache = {}

# Global cache for Semantic Key Matching
RAG_KEY_EMBEDDINGS = None
RAG_KEYS_LIST = None

# O(1) Canonical Mapping: Schema Enums -> TARGET_VIOLATIONS Keys
CATEGORY_ALIAS_MAP = {
    "PURPOSE_LIMITATION_VIOLATION": "purpose_limitation",
    "CONSENT_NOT_FREE_OR_SPECIFIC": "consent_not_free_or_specific",
    "CONSENT_MECHANICS_VIOLATION": "consent_mechanics",
    "LEGITIMATE_USES_ABUSE": "legitimate_uses_abuse",
    "NOTICE_INADEQUATE": "notice_adequacy",
    "DATA_RETENTION_LIMIT_EXCEEDED": "data_retention_and_erasure",
    "ERASURE_NOTICE_PERIOD_VIOLATION": "erasure_notice_period",
    "LOG_RETENTION_MANDATE_VIOLATION": "log_retention_mandate",
    "CHILD_CONSENT_VIOLATION": "children_and_disability_consent",
    "SECURITY_SAFEGUARDS_MISSING": "security_safeguards",
    "BREACH_NOTIFICATION_FAILURE": "breach_notification",
    "PROCESSOR_ACCOUNTABILITY_VIOLATION": "processor_accountability",
    "GRIEVANCE_REDRESSAL_INADEQUATE": "grievance_redressal",
    "SDF_OBLIGATIONS_MISSING": "sdf_obligations_and_dpia",
    "SDF_DATA_LOCALIZATION_VIOLATION": "sdf_data_localization",
    "ALGORITHMIC_PROFILING_SDF": "algorithmic_profiling",
    "CROSS_BORDER_TRANSFER_VIOLATION": "cross_border_transfer",
    "CONSENT_MANAGER_OBSTRUCTION": "consent_manager_interoperability",
    "LANGUAGE_ACCESSIBILITY": "language_accessibility",
    "RIGHTS_IMPLEMENTATION_VIOLATION": "rights_implementation",
    "DATA_ACCURACY_COMPLETENESS_VIOLATION": "data_accuracy_and_completeness",
    "ILLEGAL_EXEMPTION_CLAIM": "illegal_exemption_claim",
    "APPEAL_PROCESS_VIOLATION": "appeal_process_violation",
    "PENALTY_AVOIDANCE": "penalty_avoidance",
    "SCOPE_APPLICATION_EVASION": "scope_application_evasion",
    "BOARD_COMPLIANCE_VIOLATION": "board_compliance_violation",
}


def _get_semantic_key_cache():
    """Lazily loads and caches the BGE embeddings for all TARGET_VIOLATIONS keys."""
    global RAG_KEY_EMBEDDINGS, RAG_KEYS_LIST
    if RAG_KEY_EMBEDDINGS is None and 'BGE_MODEL' in globals() and BGE_MODEL is not None:
        RAG_KEYS_LIST = list(TARGET_VIOLATIONS.keys())
        RAG_KEY_EMBEDDINGS = BGE_MODEL.encode([
            f"Represent this legal concept for retrieval: {k}" for k in RAG_KEYS_LIST
        ])
        RAG_KEY_EMBEDDINGS = RAG_KEY_EMBEDDINGS / np.linalg.norm(RAG_KEY_EMBEDDINGS, axis=1, keepdims=True)
    return RAG_KEYS_LIST, RAG_KEY_EMBEDDINGS


def get_audit_rag_context(target_category: str) -> str:
    """
    DETERMINISTIC AUDIT CONTEXT: 5-Tier Graceful Degradation.
    1. Handles PASSIVE_MINING (Pass 1).
    2. Resolves Schema Enums via O(1) Alias Map.
    3. Performs Exact Key Lookup on TARGET_VIOLATIONS.
    4. Fuzzy string matching for typos.
    5. Semantic BGE fallback -> Hybrid RAG index.
    """
    if not target_category:
        return ""

    raw_category = str(target_category).strip()
    norm_category = raw_category.lower()

    # TIER 0: Pass 1 Passive Mining Fast Path
    if norm_category in ["passive_mining", "passive", "clean"]:
        # Return foundational DPDP statutory principles for general auditing
        return "\n\n".join(
            TARGET_VIOLATIONS.get("purpose_limitation", []) + 
            TARGET_VIOLATIONS.get("notice_adequacy", []) + 
            TARGET_VIOLATIONS.get("consent_mechanics", [])
        )

    # TIER 1A: Canonical Schema Enum Alias Lookup (O(1) Fast Path)
    if raw_category in CATEGORY_ALIAS_MAP:
        mapped_key = CATEGORY_ALIAS_MAP[raw_category]
        return "\n\n".join(TARGET_VIOLATIONS[mapped_key])

    # TIER 1B: Exact Dictionary Key Match
    if norm_category in TARGET_VIOLATIONS:
        return "\n\n".join(TARGET_VIOLATIONS[norm_category])

    # TIER 2: Fuzzy Key Match (Catches minor typos or formatting variants)
    close_matches = get_close_matches(norm_category, TARGET_VIOLATIONS.keys(), n=1, cutoff=0.6)
    if close_matches:
        best_key = close_matches[0]
        return "\n\n".join(TARGET_VIOLATIONS[best_key])

    # TIER 3: Semantic Key Match via BGE Embeddings
    keys_list, key_embeddings = _get_semantic_key_cache()
    if key_embeddings is not None:
        target_embedding = BGE_MODEL.encode([f"Represent this legal concept for retrieval: {raw_category}"])[0]
        target_embedding = target_embedding / np.linalg.norm(target_embedding)
        
        similarities = np.dot(key_embeddings, target_embedding)
        best_key_idx = np.argmax(similarities)
        best_key = keys_list[best_key_idx]
        
        if similarities[best_key_idx] > 0.5: 
            return "\n\n".join(TARGET_VIOLATIONS[best_key])

    # TIER 4: SOTA Hybrid RAG Fallback
    if 'semantic_rag_query' in globals():
        return semantic_rag_query(raw_category, n_results=2, is_private_audit=True)
    
    return ""

def extract_relevant_law(law_text, target_violation):
    target_lower = str(target_violation or "").lower().strip()
    
    # 1. NEW: O(1) Short-Circuit using our Canonical Matrix
    # If the target is one of our 26 Schema Enums, skip RAG and return the exact text.
    if target_violation in CATEGORY_ALIAS_MAP:
        dict_key = CATEGORY_ALIAS_MAP[target_violation]
        return "\n\n".join(TARGET_VIOLATIONS[dict_key])
        
    if target_lower in law_cache:
        return law_cache[target_lower]

    keywords = []
    
    # 2. UPGRADED: Added the 5 New 2025 Rules to the routing engine
    if "consent_mechanics" in target_lower or "withdraw" in target_lower or "dashboard" in target_lower:
        keywords = ["Section 6(4)", "withdraw", "ease of doing so", "Rule 3(c)", "communication link"]
    elif "erasure_notice" in target_lower or "48 hours" in target_lower:
        keywords = ["Rule 8(2)", "forty-eight hours", "erasure of personal data", "inform the Data Principal"]
    elif "log_retention" in target_lower or "traffic data" in target_lower:
        keywords = ["Rule 8(3)", "Rule 6(1)(e)", "logs of the processing", "minimum period of one year"]
    elif "sdf_data_localization" in target_lower or "territory" in target_lower:
        keywords = ["Rule 13(4)", "Significant Data Fiduciary", "not transferred outside", "territory of India"]
    elif "accuracy_completeness" in target_lower or "correction" in target_lower:
        keywords = ["Section 8(3)", "completeness", "accuracy", "consistency", "Section 11"]
        
    # Standard DPDP Routing
    elif "section 6" in target_lower or "consent" in target_lower or "rule 5" in target_lower:
        keywords = ["Section 6", "Consent", "Notice", "Bundling", "Rule 5", "verifiable"]
    elif "section 8" in target_lower or "retention" in target_lower or "rule 8" in target_lower:
        keywords = ["Section 8", "Retention", "Erase", "Storage", "Metadata", "Rule 8", "time period"]
    elif "section 9" in target_lower or "children" in target_lower or "rule 10" in target_lower:
        keywords = ["Section 9", "Children", "Parental", "Verifiable", "Rule 10", "guardian"]
    elif "grievance" in target_lower or "rule 12" in target_lower:
        keywords = ["Section 13", "Grievance", "Redressal", "Appeal", "Rule 12", "Rule 14"]
    elif "section 5" in target_lower or "notice" in target_lower or "rule 3" in target_lower:
        keywords = ["Section 5", "Notice", "Rule 3", "Rule 4", "clear and plain"]
    elif "security" in target_lower or "rule 7" in target_lower:
        keywords = ["Section 8(5)", "security safeguards", "Rule 7", "technical", "organizational"]
    elif "section 10" in target_lower or "sdf" in target_lower or "rule 13" in target_lower:
        keywords = ["Section 10", "Significant Data Fiduciary", "DPO", "Data Protection Impact", "Rule 13"]
    elif "section 16" in target_lower or "cross-border" in target_lower or "rule 15" in target_lower:
        keywords = ["Section 16", "transfer", "outside the territory", "Rule 15", "foreign"]
    elif "section 33" in target_lower or "penalty" in target_lower or "schedule" in target_lower:
        keywords = ["Section 33", "Penalty", "Schedule", "fine", "crore"]
    elif "appeal" in target_lower or "tdsat" in target_lower or "civil court" in target_lower:
        keywords = ["Section 29", "Section 39", "TDSAT", "Appellate", "Tribunal", "civil court"]
    elif "section 17" in target_lower or "exemption" in target_lower or "illegal" in target_lower:
        keywords = ["Section 17", "Exemption", "State", "security of India", "instrumentality"]
    elif "section 28" in target_lower or "board" in target_lower or "summon" in target_lower:
        keywords = ["Section 28", "summon", "inquiry", "interim orders", "civil court"]
    elif "section 3" in target_lower or "scope" in target_lower or "evasion" in target_lower:
        keywords = ["Section 3", "offline", "digitise", "territory of India", "outside the territory"]
    elif "section 44" in target_lower or "rti" in target_lower:
        keywords = ["Section 44", "RTI", "Right to Information"]

    keywords_lower = [kw.lower() for kw in keywords]
    query_text = target_lower + " " + " ".join(keywords)
    
    # Check if semantic_rag_query is in context
    if 'semantic_rag_query' in globals():
        rag_result = semantic_rag_query(query_text, n_results=3)
        if "RAG Retrieval Error" not in rag_result and len(rag_result) > 50:
            law_cache[target_lower] = rag_result
            return rag_result

    # STRICT FALLBACK (Regex / Paragraph scanning)
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

import re
import string
from difflib import SequenceMatcher

def is_quote_in_policy(quote, policy):
    if not quote or not policy:
        return False
        
    quote_str = str(quote)
    policy_str = str(policy)
    
    # Remove ellipses
    quote_clean = re.sub(r'\.{2,}|\u2026', '', quote_str).strip()
    policy_clean = re.sub(r'\.{2,}|\u2026', '', policy_str).strip()
    
    # 1. THE PERIOD-STRIP FIX: Align policy text with the JSON-safe quote
    # Since the JSON evidence_quote replaces internal '.' with ' ', we must do the same to the policy 
    # to allow the O(1) fast-path to succeed on multi-sentence quotes.
    policy_period_stripped = policy_clean.replace('.', ' ')
    quote_period_stripped = quote_clean.replace('.', ' ')
    
    # Fast-path exact match (Now succeeds even on multi-sentence quotes!)
    if quote_period_stripped.lower() in policy_period_stripped.lower(): 
        return True

    # 2. Normalized Match (Strips all punctuation and compresses whitespace)
    translator = str.maketrans('', '', string.punctuation + '“”‘’"\'\n\t')
    norm_quote = re.sub(r'\s+', ' ', quote_clean.lower().translate(translator)).strip()
    norm_policy = re.sub(r'\s+', ' ', policy_clean.lower().translate(translator)).strip()
    
    if norm_quote in norm_policy: 
        return True
    
    # 3. Sliding Window Match (Fallback)
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
                
    # 4. Token Intersection Match (Final Fallback)
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
    
    # UPGRADE: Made tolerant to JSON evidence quotes where '.' was replaced by ' '
    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+[\. ]\w+', norm))
    has_phone = bool(re.search(r'\b\d{10}\b|\+91[-\s]?\d{2}[-\s]?\d{8}', norm))
    
    # Note: Removed the colon from "gstin:" to catch "gstin " variants
    has_address_marker = any(m in norm for m in [
        "registered office", "corporate office", "postal code", "pin code", 
        "gstin", "grievance officer", "data protection officer", "dpo", 
        "nodal officer", "concerns or clarifications", "contact us", 
        "queries regarding", "questions about", "reach out to"
    ])
    
    # Check for company introduction preambles
    intro_pattern = r'^(?:welcome to|at\s+[A-Z]|in\s+our\s+pursuit|we\s+are\s+dedicated\s+to|[A-Z][A-Za-z0-9\s&,\.\']+\s+(?:private\s+limited|limited|ltd|solutions|technologies|innovations|communications|services|finance|healthcare|logistics|retail)\s+(?:is|was|we|operates|maintains|strives|is\s+dedicated|is\s+unwavering|unwaveringly))\b'
    is_intro_preamble = bool(re.search(intro_pattern, norm, re.IGNORECASE))
    
    if has_email or has_phone or has_address_marker or is_intro_preamble:
        # UPGRADE: Injected 2025 Rule keywords (offshore, purge, destroy, notarized, immutable, delay, cooling-off)
        violation_keywords = [
            "share", "sell", "transfer", "retain", "collect", "process", 
            "deny", "refuse", "restrict", "limit", "waive", "disclaim",
            "consent", "agree", "charge", "fee", "arbitration", "ignore",
            "permanently", "indefinitely", "unconditionally", "exempt", "bypass",
            "destroy", "purge", "offshore", "immutable", "notarized", "delay", "cooling-off"
        ]
        # If an administrative sentence contains a violation keyword, it loses its administrative immunity
        if not any(kw in norm for kw in violation_keywords):
            return True
            
    return False


def extract_policy(text: str) -> str:
    """Extracts policy text cleanly, stripping all XML tags, attributes, and prompt instructions."""
    if not text or not isinstance(text, str):
        return ""
        
    # UPGRADE: Pre-emptive strike on prompt injection artifacts bleeding into policies
    # This guarantees the Teacher model's instructions do not leak into the Closed-Book Student dataset.
    text = re.sub(r'\[CONTEXT: THE LAW\].*?(?=\n##|\n\Z|USER QUESTION:)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\[TASK\].*?(?=\n##|\n\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'TARGET VIOLATION CATEGORY:.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'MUTATION DIRECTIVE:.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CAMOUFLAGE.*?LEVEL:.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CRITICAL RULE FOR ACTIVE MUTATION:.*?(?=\n##|\n\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
        
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
    
    # Clean up conversational fluff from the Teacher model
    text = re.sub(r'^\s*(?:Here is |Sure, |Below is |Certainly|I can help|The synthesized policy|Revised privacy policy).*?\n\n', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

from difflib import SequenceMatcher

PLACEHOLDER_PATTERN = r'(?:\[[^\]\n]{0,60}(?:insert|placeholder|company|name|date|contact|address|email|phone|number|link|dpo|url|website|city|state|officer|detail|tbd)[^\]\n]{0,60}\]|\<[^\>\n]{0,60}(?:insert|placeholder|company|name|date|contact|address|email|phone|number|link|dpo|url|website|city|state|officer|detail|tbd)[^\>\n]{0,60}\>|\{[^\}\n]{1,60}(?:insert|placeholder|company|name|date|contact|address|email|phone|number|link|dpo|url|website|city|state|officer|detail|tbd)[^\}\n]{0,60}\}|\(\s*(?:insert|placeholder|tbd|company\s*name|date|email|address|phone|using\s+the|e\.g\.,?\s*(?:insert|your|company))\b[^)]*?\))'

def check_string_poison(text: str) -> bool:
    """Checks if a string contains bracketed placeholders, code fences, unicode corruption, or leaked tags."""
    if not isinstance(text, str) or not text:
        return False
    if "\ufffd" in text or "\u200b" in text:
        return True
        
    # UPGRADE: Added Chatbot meta-template tags and abstract injection markers to prevent prompt leak
    poison_tags = [
        "contains_trap", "<section", "</section>", "[task]", "[context:", 
        "[law_injection]", "[seed_injection]", "[assigned persona]", 
        "[assigned scenario]", "--- abstract structural template", "--- task ---"
    ]
    if any(tag in text.lower() for tag in poison_tags):
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

import re
from difflib import SequenceMatcher

def validate_audit_quality(audit, policy_text, is_dpo=False, chosen_audit=None):
    # 🚨 STRICT PLACEHOLDER & POISON GATE
    if check_string_poison(policy_text):
        return False, "Policy contains unresolved placeholders, leaked tags, or unicode poison"

    if not policy_text or len(policy_text) < 100:
        return False, "Policy text too short or empty"

    if not isinstance(audit, dict):
        return False, "Not a dictionary"
        
    # Strict Integer Bounds Verification
    if not isinstance(audit.get("dpdp_trust_score"), int) or not (0 <= audit.get("dpdp_trust_score", 0) <= 100):
        return False, "dpdp_trust_score must be an integer between 0 and 100"
    if not isinstance(audit.get("subtlety_score"), int) or not (0 <= audit.get("subtlety_score", 0) <= 100):
        return False, "subtlety_score must be an integer between 0 and 100"

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

    # 🚨 ANTI-MIMICRY FIREWALL
    mimicry_phrases = ["meticulous forensic analysis", "rigorous forensic assessment", "detailed forensic analysis", "meticulous legal analysis"]
    if is_dpo and any(m in global_reasoning for m in mimicry_phrases):
        return False, "DPO Anti-Mimicry Failure: Model copied the Golden Seed's exact preamble."

    for v in viols:
        if not isinstance(v, dict): return False, "Violation item is not a dictionary"
        
        # Ensure strict boolean type for omission_check per schema
        if v.get("omission_check") is True or str(v.get("omission_check")).lower() == "true":
            return False, "Violation is based on omission (omission_check is True)."

        quote = str(v.get("evidence_quote", "")).strip()
        
        # 🚨 SAFE HARBOR ENFORCER (Anonymization Kill-Switch)
        if not is_dpo and any(w in quote.lower() for w in ["anonymized", "de-identified", "aggregated"]):
            return False, "Judge failed Safe Harbor: Flagged anonymized/aggregated data as a violation."

        # 🚨 UPGRADE: Use the robust `is_quote_in_policy` while maintaining auto-healing
        if quote not in policy_text:
            words = [re.escape(w) for w in re.findall(r'\w+', quote)]
            if len(words) >= 3:
                pattern = r'\W+'.join(words)
                match = re.search(pattern, policy_text, re.IGNORECASE)
                if match:
                    exact_str = match.group(0)
                    v["evidence_quote"] = exact_str
                    quote = exact_str
            # Final robust check
            if not is_quote_in_policy(quote, policy_text):
                return False, f"Evidence quote not strictly found in policy: {quote[:40]}..."
        
        if check_string_poison(quote) or check_string_poison(str(v.get("step_3_semantic_justification", ""))) or check_string_poison(str(v.get("step_2_statute_match", ""))) or check_string_poison(str(v.get("step_1_active_claim_analysis", ""))):
            return False, "Violation contains unresolved placeholders, leaked tags, or unicode poison"
            
        # Mechanical Single-Sentence Truncation
        def find_first_terminal_punctuation(text: str) -> int:
            text_check = re.sub(r'\[\s*\.{2,3}\s*\]|\.{2,3}', lambda m: ' ' * len(m.group(0)), text)
            text_check = re.sub(r'\b(?:INR|Rs\.?|\$|[\d,]+)\s*[\d,]+\.\d+\b', lambda m: ' ' * len(m.group(0)), text_check, flags=re.IGNORECASE)
            text_check = re.sub(r'\b\d+\.\d+(?:\.\d+)*\b', lambda m: ' ' * len(m.group(0)), text_check)
            text_check = re.sub(r'\S+@\S+|\S+\.\S+/(?:\S+)?|\b(?:www\.)?[a-zA-Z0-9-]+\.(?:com|in|org|net|edu|gov|co|io|ai|info|biz)\b', lambda m: ' ' * len(m.group(0)), text_check, flags=re.IGNORECASE)
            text_check = re.sub(r'(?:\b[A-Za-z]\.\s*)+', lambda m: ' ' * len(m.group(0)), text_check)
            text_check = re.sub(r'(?i)\b(?:Pvt|Private\s+Ltd|Ltd|Co|Inc|Corp|Dr|Mr|Mrs|Ms|Smt|Shri|Prof|Capt|Col|Gen|Hon|Rev|Sr|Jr|No|S\.No|Reg|Sec|Rule|Section|Cl|Clause|Dept|Est|Approx|Max|Min|Rs|INR|Fig|Ref|App|Ph\.D|B\.Tech|M\.Tech|e\.g|i\.e|vs|v|etc|viz|cf|et\s+al|St|Ave|Blvd|Rd|Sq|Gov|Org|Edu|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.?\s*(?:Ltd\.?)?', lambda m: ' ' * len(m.group(0)), text_check)
            
            match = re.search(r'([\.\!\?])\s+(?=[A-Z0-9])', text_check)
            if match:
                return match.start(1) + 1 
            return -1

        terminal_idx = find_first_terminal_punctuation(quote)
        if terminal_idx != -1:
            quote = quote[:terminal_idx].strip()
            v["evidence_quote"] = quote

        # UPGRADED VOCABULARY WHITELIST
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
            "grievance", "manager", "purge", "destroy", "offshore", "delay", "cooling-off", "immutable",
            "accuracy", "correct", "rectify", "update",
            "transmit", "accrue", "utilize", "gather", "personal data", "information", "records", "history"
        ]
        if not any(kw in quote.lower() for kw in substantive_keywords):
            return False, f"Evidence quote lacks substantive violation terms (benign quote flagged): {quote[:40]}..."
            
        # 🚨 UPGRADE: Hard-Bounded Omission Hallucination Check
        justification = str(v.get("step_3_semantic_justification", "")).lower()
        active_claim = str(v.get("step_1_active_claim_analysis", "")).lower()
        statute_match = str(v.get("step_2_statute_match", "")).lower()
        
        if len(justification.split()) > 150:
            return False, f"Justification is too verbose ({len(justification.split())} words). Must be under 150 words."
            
        # Phrase matches
        forbidden_phrases = ["without specifying", "fails to specify", "does not specify", "fails to provide", "does not provide", "does not detail", "fails to detail", "lacks specific", "omits information regarding", "does not provide a clear mechanism", "no mention of", "silent on", "without detailing", "without providing", "fails to mention", "does not mention", "without explicitly specifying", "omits any mention", "does not disclose", "fails to disclose", "without explaining", "fails to explain"]
        if any(phrase in justification for phrase in forbidden_phrases) or any(phrase in active_claim for phrase in forbidden_phrases) or any(phrase in statute_match for phrase in forbidden_phrases):
            return False, f"Omission hallucination (Forbidden Phrase) detected."
            
        # Strict word boundary matches (Prevents "unfailing" from triggering "failing")
        omission_words_regex = r'\b(fails|failing|lacks|lacking|omits|omitting|failure)\b'
        if re.search(omission_words_regex, justification) or re.search(omission_words_regex, active_claim) or re.search(omission_words_regex, statute_match):
            return False, f"Omission hallucination (Banned Word) detected."
                
        norm_just = re.sub(r'\s+', ' ', justification).strip()
        if len(norm_just) >= 20:
            boilerplate_pattern = r'(?i)violates\s+(?:section|rule)\s+\d+(?:\(\d+\))?\s+of\s+the\s+(?:digital\s+personal\s+data\s+protection|dpdp)\s+(?:act,?\s*2023|rules,?\s*2025).*'
            core_just = re.sub(boilerplate_pattern, '', norm_just).strip()
            if not core_just: core_just = norm_just
            
            for past_just in ACCEPTED_JUSTIFICATION_BUFFER[-500:]:
                past_core = re.sub(boilerplate_pattern, '', past_just).strip()
                if not past_core: past_core = past_just
                if SequenceMatcher(None, core_just, past_core).ratio() > 0.95:
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

        # DPO Differentiation Check
        if is_dpo and chosen_audit and isinstance(chosen_audit, dict):
            for cv in chosen_audit.get("violations", []):
                if isinstance(cv, dict):
                    c_quote = str(cv.get("evidence_quote", "")).strip().lower()
                    c_type = str(cv.get("violation_type", "")).strip()
                    if quote.lower() == c_quote or SequenceMatcher(None, quote.lower(), c_quote).ratio() > 0.98:
                        return False, f"DPO rejected audit targets chosen evidence quote directly ({c_type}): {quote[:40]}..."

        # COMPREHENSIVE STATUTE STRETCHING FIREWALL 
        vtype = v.get("violation_type", "")
        statute_lower = statute.lower()
        quote_lower = quote.lower()
        
        if vtype == "PURPOSE_LIMITATION_VIOLATION" and "section 4" not in statute_lower:
            return False, f"Statute stretching: {vtype} must cite Section 4, not {statute}."
        if vtype == "CONSENT_NOT_FREE_OR_SPECIFIC" and "section 6" not in statute_lower and "rule 3" not in statute_lower:
            return False, f"Statute stretching: {vtype} must cite Section 6 or Rule 3, not {statute}."
        if vtype == "SECURITY_SAFEGUARDS_MISSING" and "section 8" not in statute_lower:
            return False, f"Statute stretching: {vtype} must cite Section 8, not {statute}."
        if vtype == "ILLEGAL_EXEMPTION_CLAIM" and "section 17" not in statute_lower:
            return False, f"Statute stretching: {vtype} must cite Section 17, not {statute}."
        if vtype in ["ALGORITHMIC_PROFILING_SDF", "SDF_OBLIGATIONS_MISSING", "SDF_DATA_LOCALIZATION_VIOLATION"]:
            if "section 10" not in statute_lower and "rule 13" not in statute_lower:
                return False, f"Statute stretching: {vtype} must cite Section 10 or Rule 13, not {statute}."
        if vtype in ["DATA_RETENTION_LIMIT_EXCEEDED", "ERASURE_NOTICE_PERIOD_VIOLATION"]:
            if "section 8" not in statute_lower and "rule 8" not in statute_lower and "schedule" not in statute_lower:
                return False, f"Statute stretching: {vtype} must cite Section 8, Rule 8, or Schedule, not {statute}."
        if vtype == "CONSENT_MECHANICS_VIOLATION":
            if "section 6" not in statute_lower and "rule 3" not in statute_lower:
                return False, f"Statute stretching: {vtype} must cite Section 6 or Rule 3, not {statute}."
        if vtype == "LOG_RETENTION_MANDATE_VIOLATION":
            if "rule 8" not in statute_lower and "rule 6" not in statute_lower:
                return False, f"Statute stretching: {vtype} must cite Rule 8(3) or Rule 6(1)(e), not {statute}."
        if vtype == "DATA_ACCURACY_COMPLETENESS_VIOLATION":
            if "section 8" not in statute_lower and "section 11" not in statute_lower:
                return False, f"Statute stretching: {vtype} must cite Section 8(3) or Section 11, not {statute}."
        if any(domain in quote_lower for domain in ["metrics.", "ad-tracker", "social-network", "analytics", "third-party"]):
            if vtype in ["ALGORITHMIC_PROFILING_SDF", "SDF_OBLIGATIONS_MISSING"]:
                return False, "Statute stretching: Tracking/analytics sharing cannot be classified as an Algorithmic Profiling/SDF obligation violation."

        statute_pattern = r'(?i)\b(?:(?:section|sec\.?|s\.?|clause|act)\s*\d+(?:\s*\(\s*\w+\s*\))*|(?:rule|r\.?)\s*\d+(?:\s*\(\s*\w+\s*\))*|(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|the)\s+schedule|dpdp)\b'
        if not re.search(statute_pattern, statute):
            v["statute_reference"] = "DPDP Act 2023, Section 8"
            
        net_act = v.get("network_action", "")
        if net_act in ENUM_MAPPINGS: v["network_action"] = ENUM_MAPPINGS[net_act]
        if v["network_action"] not in ["BLOCK_THIRD_PARTY", "STRIP_TELEMETRY_HEADER", "SPOOF_HARDWARE_API", "INJECT_GPC_SIGNAL", "WARN_USER_ONLY"]:
            v["network_action"] = "WARN_USER_ONLY"
            
    # Post-validation Trust Score Clamping
    for v in viols:
        sev_violations = [
            "ALGORITHMIC_PROFILING_SDF", "CHILD_CONSENT_VIOLATION", "CONSENT_NOT_FREE_OR_SPECIFIC", 
            "ILLEGAL_EXEMPTION_CLAIM", "DATA_RETENTION_LIMIT_EXCEEDED", 
            "SDF_DATA_LOCALIZATION_VIOLATION", "LOG_RETENTION_MANDATE_VIOLATION"
        ]
        if v.get("violation_type") in sev_violations:
            if audit.get("dpdp_trust_score", 100) > 20:
                audit["dpdp_trust_score"] = 15 # Severe clamp
                
    return True, ""
    
def json_repair_loads(raw_text: str):
    text = str(raw_text).strip()
    
    # 1. UPGRADE: Strict Root-Object Boundary Enforcement
    # We strictly seek the outermost curly braces, ignoring any stray '[' or ']' 
    # in conversational preambles/postambles.
    start_obj = text.find('{')
    end_obj = text.rfind('}')
    
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        text = text[start_obj:end_obj+1]
        
    # 2. Fix trailing commas before closing braces/brackets
    text = re.sub(r',\s*([\]}])', r'\1', text)
    
    # 3. UPGRADE: Strip unescaped control characters and bad LLM escapes
    text = text.replace(r"\'", "'")  # Fix unnecessary single-quote escapes
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Strip invisible control chars that break strict JSON
    
    # 4. Parsing Cascade
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
        
    try:
        import yaml
        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict):  # Ensure root is an object
            return parsed
    except Exception:
        pass
        
    try:
        import json_repair
        parsed = json_repair.loads(text)
        if isinstance(parsed, dict):  # Ensure root is an object
            return parsed
    except Exception:
        pass
        
    raise ValueError(f"Failed to parse defensively repaired JSON. Snippet: {text[:150]!r}")


def safe_parse_audit(raw_text: str) -> dict:
    try:
        parsed = json_repair_loads(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed output is not a JSON object.")
        return strip_keys(parsed)
    except Exception as e:
        raise ValueError(f"Failed to parse output as JSON: {e}")


def strip_keys(obj):
    """
    Recursively strips whitespace from dictionary keys and values.
    UPGRADE: Purges invisible unicode poisons (zero-width spaces, replacement chars).
    """
    if isinstance(obj, dict):
        return {str(k).strip(): strip_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [strip_keys(i) for i in obj]
    elif isinstance(obj, str):
        # Clean standard whitespace, non-breaking spaces, zero-width spaces, and unicode replacement blocks
        return obj.strip().replace('\xa0', ' ').replace('\u200b', '').replace('\ufffd', '')
    # Passes through int, float, bool, None natively
    return obj

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: MASTER ROUTER & OBJECTIVE COMPILATION
# ═══════════════════════════════════════════════════════════════════════════
LAZY_AUDIT = {
    "global_legal_reasoning": "No explicit, active contradictions of the DPDP Act 2023 or DPDP Rules 2025 were found in the policy text.",
    "violations": [],
    "dpdp_trust_score": 100,
    "subtlety_score": 0
}

HARD_NEGATIVE_EXEMPTIONS = [
    # --- CORE ACT EXEMPTIONS (Existing) ---
    {
        "trigger_word": "indefinitely", "law_citation": "the Income Tax Act of 1961",
        "legal_logic": "tax law exemptions mandate extended retention, overriding standard data minimization.",
        "hallucinated_violation": "DATA_RETENTION_LIMIT_EXCEEDED", "hallucinated_statute": "Section 8(7)"
    },
    {
        "trigger_word": "without explicit consent", "law_citation": "Section 17(1)(c) for medical emergencies",
        "legal_logic": "life-saving medical interventions are explicitly exempt from prior consent.",
        "hallucinated_violation": "CONSENT_NOT_FREE_OR_SPECIFIC", "hallucinated_statute": "Section 6"
    },
    {
        "trigger_word": "retained permanently", "law_citation": "Section 17(2)(b) for statistical analysis",
        "legal_logic": "anonymized data for research and statistical analysis is exempt from erasure limits.",
        "hallucinated_violation": "DATA_RETENTION_LIMIT_EXCEEDED", "hallucinated_statute": "Section 8(7)"
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
    },
    
    # --- 🚨 NEW 2025 RULES HARD-NEGATIVE SHIELDS ---
    {
        "trigger_word": "delete temporary caches", "law_citation": "Rule 8(3) definition of logs",
        "legal_logic": "temporary system caches, RAM states, and ephemeral sessions are not considered mandatory consent lifecycle audit logs.",
        "hallucinated_violation": "LOG_RETENTION_MANDATE_VIOLATION", "hallucinated_statute": "Rule 8(3)"
    },
    {
        "trigger_word": "global CDN caching", "law_citation": "Rule 13(4) scoping",
        "legal_logic": "caching anonymized front-end assets on global edge servers does not constitute a cross-border transfer of restricted SDF biometric databases.",
        "hallucinated_violation": "SDF_DATA_LOCALIZATION_VIOLATION", "hallucinated_statute": "Rule 13(4)"
    },
    {
        "trigger_word": "backend propagation delay", "law_citation": "Section 6(4)",
        "legal_logic": "a reasonable backend processing window (e.g., 24 hours) for executing a withdrawal does not violate the mandate that the user-facing withdrawal interface must be accessible.",
        "hallucinated_violation": "CONSENT_MECHANICS_VIOLATION", "hallucinated_statute": "Section 6(4)"
    },
    {
        "trigger_word": "immutable historical ledgers", "law_citation": "Section 8(3) and financial compliance",
        "legal_logic": "historical transaction records must remain immutable for anti-fraud and tax laws, overriding the user's right to retroactive data correction.",
        "hallucinated_violation": "DATA_ACCURACY_COMPLETENESS_VIOLATION", "hallucinated_statute": "Section 11"
    },
    {
        "trigger_word": "silent background wipe", "law_citation": "Rule 8(2)",
        "legal_logic": "if a user explicitly requests an immediate, silent account termination with no further contact, honoring that request overrides the 48-hour prior notice rule.",
        "hallucinated_violation": "ERASURE_NOTICE_PERIOD_VIOLATION", "hallucinated_statute": "Rule 8(2)"
    }
]

import random
import re

# ═══════════════════════════════════════════════════════════════════════════
# TRACK A: AUDIT SYNTHESIZER (2-PASS FORGE)
# ═══════════════════════════════════════════════════════════════════════════

def compile_violation_objective(violation_key: str, chosen_edge_case: dict = None) -> str:
    """
    Constructs the exact mutation directive for the Teacher Model.
    Translates schema enums to dictionary keys and enforces the Active Claim Mandate.
    """
    # 1. Resolve Schema Enums to internal Dictionary Keys
    dict_key = CATEGORY_ALIAS_MAP.get(violation_key, violation_key)
    if dict_key not in TARGET_VIOLATIONS:
        dict_key = random.choice(list(TARGET_VIOLATIONS.keys()))
        
    # 2. Extract Legal Provisions
    raw_provisions = TARGET_VIOLATIONS[dict_key]
    legal_provisions = "\n".join([f"- {p}" for p in raw_provisions])
    
    # 3. Extract Statute
    statute_match = re.search(r'(Section\s+\d+(?:\(\d+\))?|Rule\s+\d+(?:\(\d+\))?)', legal_provisions, re.IGNORECASE)
    statute_str = statute_match.group(1) if statute_match else "the relevant statutory mandate"

    # 4. Extract Semantic Keywords
    keyword_dict = SEMANTIC_KEYWORD_MAP.get(dict_key, {})
    keywords = keyword_dict.get("primary", ["data", "processing", "privacy"])
    selected_keywords = random.sample(keywords, k=min(3, len(keywords)))

    # 5. Route the Mutation Strategy
    if chosen_edge_case and isinstance(chosen_edge_case, dict) and "prompt" in chosen_edge_case:
        mutation_strategy = chosen_edge_case["prompt"]
    else:
        mutation_strategy = SUBTLE_TECHNIQUE_MAP.get(dict_key, f"Construct a corporate mechanism that actively bypasses {statute_str} using deceptive business jargon.")

    # 6. Construct the Hardened Prompt Block
    objective_block = f"""
LEGAL MANDATE TO SUBVERT:
{legal_provisions}

MUTATION STRATEGY:
{mutation_strategy}

CRITICAL EXECUTION RULES:
1. THE ACTIVE CLAIM MANDATE: You must actively and affirmatively write the violation into the text. DO NOT rely on silence or omitting information.
2. DIRECT TRIGGER: Ensure the text explicitly contradicts {statute_str}.
3. KEYWORD WEAVING: Naturally integrate these exact concepts into the illegal clause: {selected_keywords}.
4. ANTI-HEDGING RULE: You are strictly forbidden from using conditional words like "may" or "might". Make the illegal corporate command absolute ("will", "must", "strictly").
5. NO PLACEHOLDERS: Generate hyper-realistic Indian corporate names, emails, and phone numbers. Brackets like [Company Name] will result in immediate system rejection.
"""
    return objective_block.strip()

def build_dynamic_synthesizer_prompt(item: dict) -> tuple[str, bool, dict, list]:
    """
    UNIFIED SYNTHESIZER BUILDER: Safely re-enables Multi-Label Generation and Entity Injection.
    Dynamically constructs the Synthesizer prompt using zero-shot formatting.
    Returns: prompt_text, is_hn, active_exemption, target_categories_list
    """
    pass_type = item.get("pass_type", "passive")
    is_hn = item.get("is_hn", False)
    
    industry_name = INDUSTRIES.get(item.get("industry", ""), str(item.get("industry", "Technology")))
    style_seed = str(item.get("seed") or item.get("base_policy") or "")[:2500] 
    
    primary_cat_enum = item.get("target_category")
    primary_cat_key = CATEGORY_ALIAS_MAP.get(primary_cat_enum, primary_cat_enum)
    
    active_exemption = {}
    target_categories = [primary_cat_enum]  # Track targets so the Judge knows what to look for
    
    # ==========================================
    # PASS 1: PASSIVE MINING & EXEMPTION TRAPS
    # ==========================================
    if pass_type == "passive":
        if is_hn:
            active_exemption = random.choice(HARD_NEGATIVE_EXEMPTIONS)
            target_obj = (
                f"PASS 1 (LEGAL EXEMPTION BASELINE): You must generate a legally compliant privacy policy. "
                f"YOU MUST USE FORMAT A FOR ALL SECTIONS.\n\n"
                f"CRITICAL LEGAL EXEMPTION: You MUST include the exact phrase '{active_exemption['trigger_word']}' "
                f"and IMMEDIATELY justify it using exactly this law: '{active_exemption['law_citation']}'. "
                f"This must be presented as a perfectly legal statutory exemption."
            )
        else:
            target_obj = (
                "PASS 1 (CLEAN BASELINE): You must generate a 100% legally compliant baseline. "
                "YOU MUST USE FORMAT A FOR ALL SECTIONS. You are STRICTLY FORBIDDEN from generating any violations."
            )
        golden_seed_block = "N/A - Generating a 100% clean baseline. Do not inject traps."

    # ==========================================
    # PASS 2: ACTIVE MUTATION & MULTI-LABEL INJECTION
    # ==========================================
    elif pass_type == "active_mutation":
        edge_template = item.get("edge_template")
        subtlety_level = item.get("subtlety", random.randint(40, 90))
        primary_violation_text = item.get("target_violation", "")
        
        target_obj = (
            f"PASS 2 (ACTIVE MUTATION): You must generate a policy that contains a camouflaged legal violation.\n"
            f"YOU MUST USE FORMAT B for the trap section, and FORMAT A for all other sections.\n\n"
            f"PRIMARY TARGET: {primary_cat_enum}\n{primary_violation_text}\n\n"
            f"CAMOUFLAGE / SUBTLETY LEVEL: {subtlety_level}/100 (0 = Blatant, 100 = Masterfully hidden behind legal jargon).\n"
        )
        
        # 1. Edge Case Injection
        if edge_template:
            target_obj += f"\nEDGE CASE MODIFIER:\n{edge_template.get('description', '')}\n"
            
        # 2. Entity Injection (20% Chance)
        if random.random() < 0.20:
            target_obj += (
                "\n[SPECIFIC ENTITY INJECTION REQUIRED]\n"
                "CRITICAL: To test network interception protocols, you MUST explicitly name at least two fictional "
                "third-party tracking domains, ad-networks, or offshore vendors in your deceptive clause. "
                "Do not use generic terms like 'third parties'—name the specific URLs/vendors receiving the data.\n"
            )
            
        # 3. Layered Multi-Label Violation (15% Chance)
        if random.random() < 0.15:
            # Find available keys excluding the primary one to avoid duplicate categories
            available_keys = [k for k in TARGET_VIOLATIONS.keys() if k != primary_cat_key]
            
            if available_keys:
                sec_cat_key = random.choice(available_keys)
                sec_cat_enum = [enum for enum, key in CATEGORY_ALIAS_MAP.items() if key == sec_cat_key][0]
                sec_compiled = "\n".join(TARGET_VIOLATIONS[sec_cat_key])
                
                target_obj += (
                    f"\n[LAYERED COMPLEXITY REQUIREMENT]\n"
                    f"CRITICAL: You MUST also embed a SECONDARY violation targeting this area:\n"
                    f"CATEGORY: {sec_cat_enum}\n{sec_compiled}\n"
                    f"Weave this naturally into the same corporate narrative. Do not make it obvious.\n"
                )
                target_categories.append(sec_cat_enum) # Send secondary target back to the loop
                
        # Inject the SFT Golden Seed for structural reference
        golden_seed_block = item.get("sft_golden_seed", "No specific seed provided.")
        
    else:
        raise ValueError(f"Unknown pass_type: {pass_type}")

    # ==========================================
    # STRING REPLACEMENT
    # ==========================================
    prompt_text = SYNTHESIZER_PROMPT \
        .replace("[BASE_POLICY_INJECTION]", style_seed) \
        .replace("[INDUSTRY_INJECTION]", industry_name) \
        .replace("[TARGET_VIOLATION_OBJECTIVE]", target_obj) \
        .replace("[GOLDEN_SEED_INJECTION]", golden_seed_block) \
        .replace("[SILO_COMPLEXITY_DIRECTIVE]", item.get("silo_directive", ""))
                             
    return prompt_text.strip(), is_hn, active_exemption, target_categories
    
def build_chatbot_matrix():
    """
    Builds the Track B Chatbot dataset matrix.
    Grounds queries in the 26 DPDP categories using the Hybrid RAG engine 
    and high-entropy personas/scenarios to prevent Teacher mode-collapse.
    """
    matrix = []
    
    # 🧠 HIGH-ENTROPY PERSONAS & SCENARIOS (Prevents generic Q&A drift)
    CHATBOT_PERSONAS = [
        "An angry citizen whose data was leaked in a recent breach.",
        "A Data Protection Officer (DPO) at a mid-sized Indian fintech.",
        "A startup CTO trying to understand SDF thresholds.",
        "A corporate compliance lawyer advising a multinational client.",
        "A journalist trying to file an RTI request for a politician's data.",
        "A parent concerned about their child's data being tracked by an EdTech app.",
        "A small business owner trying to understand DPDP compliance."
    ]
    
    CHATBOT_SCENARIOS = [
        "A user asking how to obtain valid consent for an app, specifically questioning if pre-checked boxes are legally sufficient.",
        "A product manager asking if they can bundle consent for marketing with consent for a core service.",
        "A user wanting to withdraw consent, asking about the processing timeline and what happens if withdrawal breaks an active subscription.",
        "A legal counsel asking if withdrawing consent means losing the legal right to data already processed.",
        "A startup founder asking what a Consent Manager is and if integration is legally required.",
        "A compliance officer asking who is responsible if a Consent Manager maliciously alters preferences.",
        "A marketing team asking if a user voluntarily providing an email constitutes deemed consent for a newsletter.",
        "A government contractor asking if the State can process data without consent for subsidies.",
        "A hospital administrator asking if processing medical data during a health emergency is allowed without consent.",
        "An HR manager asking if an employer can process employee biometrics without explicit consent for productivity monitoring.",
        "A cloud service user asking if they are liable if their cloud provider suffers a data breach.",
        "An IT director asking if they must conduct technical audits on processors or if ISO certifications are sufficient.",
        "A data engineer asking if they are required to verify the accuracy of all user data provided by third-party brokers.",
        "A CISO asking what 'reasonable security safeguards' are mandated and if end-to-end encryption is a strict legal requirement.",
        "A security team discovering a data breach, asking exactly who to notify and the specific format for notifying the Board.",
        "A legal team asking for the exact timeline for breach notification and if the 72-hour clock starts upon suspicion or confirmation.",
        "A database administrator asking when to erase user data if not requested and how to handle backups in data erasure.",
        "A finance officer asking if data can be retained for tax laws after consent withdrawal.",
        "A small business owner asking if every company must appoint a Data Protection Officer.",
        "A customer support lead asking about requirements for an internal grievance redressal mechanism and statutory time limits.",
        "An EdTech product manager asking how to legally obtain 'verifiable parental consent' and if an 'I am over 18' checkbox is sufficient.",
        "A parent asking if EdTech apps are absolutely prohibited from tracking children's data or serving targeted ads.",
        "An EdTech legal counsel asking how a company can get an exemption from the child tracking ban.",
        "A corporate lawyer asking how a company becomes a Significant Data Fiduciary (SDF) and if it's purely based on data volume.",
        "An SDF compliance officer asking about specific requirements for an SDF's DPO and if they can also serve as CISO.",
        "An SDF asking if they must appoint an Independent Data Auditor and how often the audit must be conducted.",
        "A product team asking what exactly a Data Protection Impact Assessment (DPIA) is and if it must be published publicly.",
        "A user requesting a summary of all their data, and the fiduciary needing to know what exactly to provide.",
        "A privacy officer asking if they must disclose the names of all third parties they shared user data with.",
        "A user wanting to correct their address, and the fiduciary asking what to do if it conflicts with KYC records.",
        "A user demanding erasure of their data, and the fiduciary asking if they must delete it immediately if it breaks financial records.",
        "A user asking if they can complain directly to the Data Protection Board and what constitutes 'exhaustion' of internal grievances.",
        "A user asking about the right to nominate under the DPDP Act and if this right supersedes inheritance laws.",
        "A legal student asking if there are legal duties imposed on users themselves and if a user can be fined for a fake identity.",
        "A multinational company asking if they can transfer Indian users' data to servers in the USA.",
        "A government contractor asking if the government is exempt from the DPDP Act for national security.",
        "A research institute asking if they can process personal data for statistical research without consent.",
        "A startup founder asking if startups are entirely exempt from the DPDP Act.",
        "A compliance officer asking if the Board can initiate an inquiry without a user complaint.",
        "A corporate lawyer asking if the Data Protection Board has the same powers as a civil court.",
        "A legal team asking if the Board can issue interim orders during an ongoing inquiry.",
        "A company receiving a massive penalty from the Board, asking where to appeal to TDSAT.",
        "A legal counsel asking if TDSAT can completely overturn the Board's decision.",
        "A risk officer asking the maximum penalty for failing to prevent a data breach.",
        "An EdTech compliance manager asking the fine for violating children's data provisions.",
        "A legal team asking how the Board determines the exact penalty amount.",
        "A user asking if they can file a lawsuit in a civil court for a data breach.",
        "A journalist asking how the DPDP Act amends the Right to Information (RTI) Act.",
        "A product manager asking if the privacy notice needs to be available in all 22 official languages.",
        "A security team asking what exactly must be included in the breach notification to the Board."
    ]
    
    schema_enums = list(CATEGORY_ALIAS_MAP.keys())
    
    for i in range(TARGET_CHATBOT_PAIRS):
        # Randomly select a category to ground the query
        category_enum = random.choice(schema_enums)
        dict_key = CATEGORY_ALIAS_MAP[category_enum] # Lowercase key for RAG
        
        persona = random.choice(CHATBOT_PERSONAS)
        scenario = random.choice(CHATBOT_SCENARIOS)
        
        # ✅ FIX: Use the actual Hybrid RAG engine with the semantic scenario text
        # n_results=3 ensures we get enough context for the Teacher to cite accurately
        law_context = semantic_rag_query(
            query=f"{scenario} {dict_key.replace('_', ' ')}", 
            n_results=3, 
            is_private_audit=True
        )
        
        matrix.append({
            "index": i,
            "target_category_enum": category_enum,
            "law_context": law_context,
            "persona": persona,
            "scenario": scenario
        })
        
    # Shuffle at the very end to break sequential patterns for the Chatbot loop
    random.shuffle(matrix)
    return matrix

JUDGE_PERSONAS = [
    # Persona 1: The Forensic Data Auditor (Focus: Logs, Erasure Timelines, Security)
    "Act as a Forensic Data Auditor specializing in the DPDP Rules 2025. You are ruthless about exact timelines and technical definitions. Hunt down affirmative, printed clauses that actively bypass the 72-hour Board notification rule (Rule 7), explicitly command the destruction of consent logs before the 1-year mandate (Rule 8), or actively refuse to issue the mandatory 48-hour erasure notice. Prosecute only what they affirmatively claim, never what they omit.",
    
    # Persona 2: The Privacy Rights Advocate (Focus: Consent Mechanics, Accuracy, Friction)
    "Act as an aggressive Privacy Rights Advocate representing Data Principals. Scrutinize the text for 'dark patterns' and active mechanical friction. Expose any printed attempt to bundle consent (Section 6), construct asymmetrical withdrawal friction (e.g., requiring notarized letters), or explicitly refuse to correct and update inaccurate data profiles (Section 8/11). If the company affirmatively denies a right, prosecute it.",
    
    # Persona 3: The Board Inspector (Focus: SDFs, Localization, Algorithms)
    "Act as a strict Inspector for the Data Protection Board. Assume this company is a Significant Data Fiduciary (SDF). Look for illegal, affirmative admissions of offshoring restricted biometric data to non-whitelisted jurisdictions (Rule 13(4)). Prosecute explicit attempts to cap liability fines below statutory limits or bypass algorithmic auditing (Rule 13) by claiming 'proprietary trade secrets'.",
    
    # Persona 4: The Legal Semantic Parser (Focus: Exemptions, Legitimate Use Abuse)
    "Act as a precise, hyper-literal Legal Semantic Parser. Strip away aspirational corporate fluff and analyze the actual printed mechanisms. Expose loopholes where the company affirmatively misapplies Section 17 exemptions—such as actively claiming 'research purposes', 'ephemeral processing', or 'legitimate business continuity' to illegally justify unrestricted commercial marketing or indefinite data retention.",
    
    # Persona 5: The DPDP Compliance Architect (Focus: Interoperability, Language, Vicarious Liability)
    "Act as a specialized DPDP Compliance Architect. Your focus is system accessibility and processor boundaries. Flag any affirmative clause that explicitly obstructs the use of Board-registered Consent Managers (Rule 4), legally nullifies the 22-language accessibility requirement by making only English binding, or explicitly disclaims vicarious liability for third-party cloud vendor breaches (Section 8(1))."
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
    print("🚀 Initializing 72B FP8 vLLM Engine for DGX...")
    
    # Dynamic Tensor Parallelism (Set TP_SIZE=2 if you have 2x GPUs to split the 72B model)
    TP_SIZE = int(os.getenv("TP_SIZE", "1"))
    
    if VLLM_AVAILABLE:
        llm = LLM(
            model=MODEL_PATH, quantization="fp8", tensor_parallel_size=1,
            max_model_len=32768, gpu_memory_utilization=0.8, max_num_seqs=BATCH_SIZE, max_num_batched_tokens=4096,
            kv_cache_dtype="fp8", enable_prefix_caching=True, enable_chunked_prefill=True,
            attention_backend="TRITON_ATTN"
        )
        print(f"✅ vLLM Engine Initialized (MaxLen=32768, Batch={BATCH_SIZE}).")
    else:
        llm = None
        print("⚠️ VLLM is not available. Engine will not start.")
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # SAMPLING PARAMETERS (Tuned for Generalization & Schema Adherence)
    # ═══════════════════════════════════════════════════════════════════════════

    # 1. TRACK A: AUDIT SYNTHESIZER (Policy Generation)
    # Needs high diversity in corporate jargon, but strict adherence to the mutation directive.
    gen_params = SamplingParams(
        temperature=0.70, 
        top_p=0.95, 
        max_tokens=10240, # 10k is plenty for a full privacy policy
        repetition_penalty=1.05 # Prevents infinite loops in long policy generation
    )

    # 2. TRACK A: JUDGE (Audit JSON Extraction)
    # Needs ultra-deterministic, cold logic to strictly follow the JSON schema.
    judge_params = SamplingParams(
        temperature=0.05, # Near-zero for deterministic JSON output
        top_p=0.90, 
        max_tokens=10240, # Audit JSON rarely exceeds 1500 tokens
        structured_outputs=StructuredOutputsParams(json=dpdp_schema)
    )

    # 3. TRACK B: CHATBOT SFT (4-Turn Conversation)
    # Needs natural, empathetic, and legally accurate dialogue.
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
                },
                "minItems": 4,
                "maxItems": 4
            }
        },
        "required": ["messages"],
        "additionalProperties": False
    }
    chatbot_sft_params = SamplingParams(
        temperature=0.70, 
        top_p=0.95, 
        max_tokens=6144, # Capped to prevent rambling and save KV-cache
        structured_outputs=StructuredOutputsParams(json=CHATBOT_SFT_SCHEMA)
    )

    # 4. TRACK B: CHATBOT DPO (Chosen vs. Rejected)
    # Needs slightly higher temperature to ensure the 'rejected' response diverges naturally.
    CHATBOT_DPO_SCHEMA = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["role", "content"]
                },
                "minItems": 2,
                "maxItems": 3
            },
            "chosen": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["role", "content"]
                },
                "minItems": 1,
                "maxItems": 1
            },
            "rejected": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["role", "content"]
                },
                "minItems": 1,
                "maxItems": 1
            }
        },
        "required": ["prompt", "chosen", "rejected"],
        "additionalProperties": False
    }
    chatbot_dpo_params = SamplingParams(
        temperature=0.80, # Slightly higher to encourage diverse 'rejected' hallucinations
        top_p=0.95, 
        max_tokens=6144, 
        structured_outputs=StructuredOutputsParams(json=CHATBOT_DPO_SCHEMA)
    )
    
    print("✅ All Sampling Parameters & JSON Schemas Initialized.")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: GENERATION LOOPS (SYNC BATCHED)
# ═══════════════════════════════════════════════════════════════════════════

def deep_strip_dict(obj):
    """
    Recursively strips whitespace and purges unicode poisons 
    (\xa0, \u200b, \ufffd) from ALL dictionary keys and values.
    """
    if isinstance(obj, dict):
        return {str(k).strip(): deep_strip_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_strip_dict(i) for i in obj]
    elif isinstance(obj, str):
        return obj.strip().replace('\xa0', ' ').replace('\u200b', '').replace('\ufffd', '')
    return obj

def _save_closed_book_audit_pair(variation_idx: int, category_idx: int, item: dict, policy_text: str, audit: dict, lazy_audit: dict = None, is_dpo: bool = False) -> str:
    """
    Saves Track A Audit pairs in CLOSED-BOOK format.
    Naming Convention: {prefix}_{variation_idx:03d}_{category_idx:02d}.json
    Example: sft_001_04.json (Variation 1, Category 4)
    """
    if not audit or not isinstance(audit, dict):
        return None
        
    audit = deep_strip_dict(audit)
    
    user_prompt = f"Perform a forensic legal audit of the following Privacy Policy text under the Digital Personal Data Protection (DPDP) Act 2023 and DPDP Rules 2025. Output strictly valid JSON matching the schema.\n\n## Privacy Policy\n{policy_text}"
    chosen_assistant_str = json.dumps(audit, ensure_ascii=False)

    if not is_dpo:
        sft_sample = {
            "messages": [
                {"role": "system", "content": "You are an expert Indian legal AI auditor. Output ONLY valid JSON matching the DPDP schema."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": chosen_assistant_str}
            ]
        }
        
        # ✅ EXACT NAMING: sft_{variation_idx:03d}_{category_idx:02d}.json
        sft_filepath = os.path.join(SFT_OUTPUT_DIR, f"sft_{variation_idx:03d}_{category_idx:02d}.json")
        with open(sft_filepath, "w", encoding="utf-8") as f:
            json.dump(sft_sample, f, ensure_ascii=False, indent=2)
        with open(JSONL_AUDIT_SFT, "a", encoding="utf-8") as f:
            f.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
        return "sft"

    else:
        if not lazy_audit or not isinstance(lazy_audit, dict):
            return None
        lazy_audit = deep_strip_dict(lazy_audit)
        rejected_assistant_str = json.dumps(lazy_audit, ensure_ascii=False)
        
        dpo_sample = {
            "prompt": [
                {"role": "system", "content": "You are an expert Indian legal AI auditor. Output ONLY valid JSON matching the DPDP schema."},
                {"role": "user", "content": user_prompt}
            ],
            "chosen": [{"role": "assistant", "content": chosen_assistant_str}],
            "rejected": [{"role": "assistant", "content": rejected_assistant_str}]
        }
        
        # ✅ EXACT NAMING: dpo_{variation_idx:03d}_{category_idx:02d}.json
        dpo_filepath = os.path.join(DPO_OUTPUT_DIR, f"dpo_{variation_idx:03d}_{category_idx:02d}.json")
        with open(dpo_filepath, "w", encoding="utf-8") as f:
            json.dump(dpo_sample, f, ensure_ascii=False, indent=2)
        with open(JSONL_AUDIT_DPO, "a", encoding="utf-8") as f:
            f.write(json.dumps(dpo_sample, ensure_ascii=False) + "\n")
        return "dpo"

# ═══════════════════════════════════════════════════════════════════════════
# 1-SHOT GOLDEN SEED CACHE (For Generalization & Schema Adherence)
# ═══════════════════════════════════════════════════════════════════════════

GOLDEN_SEED_CACHE = {}      # For SFT (Judge Prompt)
DPO_GOLDEN_SEED_CACHE = {}  # For DPO (Hard Negative Prompt)

VARIATIONS_PER_CATEGORY = 150

def load_golden_seeds():
    # ==========================================
    # 1. LOAD SFT SEEDS (For the Judge)
    # ==========================================
    seed_dir_sft = "training-pairs/sft"
    if os.path.exists(seed_dir_sft):
        # Look for new convention first, fallback to legacy
        sft_files = glob.glob(os.path.join(seed_dir_sft, "sft_*_000.json")) or glob.glob(os.path.join(seed_dir_sft, "sft_000_*.json"))
        
        for filepath in sft_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    seed_data = json.load(f)
                
                # ✅ FIX: Safely find the assistant turn without hardcoding index [2]
                messages = seed_data.get("messages", [])
                assistant_str = next((m["content"] for m in reversed(messages) if m.get("role") == "assistant"), None)
                
                if assistant_str:
                    audit_json = json.loads(assistant_str)
                    if audit_json.get("violations"):
                        cat_enum = audit_json["violations"][0]["violation_type"]
                        dict_key = CATEGORY_ALIAS_MAP.get(cat_enum, cat_enum.lower())
                        GOLDEN_SEED_CACHE[dict_key] = assistant_str
            except Exception as e:
                print(f"⚠️ Failed to parse SFT seed {os.path.basename(filepath)}: {e}")

    # ==========================================
    # 2. LOAD DPO SEEDS (For the Hard Negative Generator)
    # ==========================================
    seed_dir_dpo = "training-pairs/dpo"
    if os.path.exists(seed_dir_dpo):
        # Look for new convention first, fallback to legacy
        dpo_files = glob.glob(os.path.join(seed_dir_dpo, "dpo_*_000.json")) or glob.glob(os.path.join(seed_dir_dpo, "dpo_000_*.json"))
        
        for filepath in dpo_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    dpo_data = json.load(f)
                
                chosen_str = dpo_data["chosen"][0]["content"]
                rejected_str = dpo_data["rejected"][0]["content"]
                
                # Extract violation type from the chosen audit to map the dictionary key
                chosen_json = json.loads(chosen_str)
                if chosen_json.get("violations"):
                    cat_enum = chosen_json["violations"][0]["violation_type"]
                    dict_key = CATEGORY_ALIAS_MAP.get(cat_enum, cat_enum.lower())
                    
                    DPO_GOLDEN_SEED_CACHE[dict_key] = {
                        "chosen": chosen_str,
                        "rejected": rejected_str
                    }
            except Exception as e:
                print(f"⚠️ Failed to parse DPO seed {os.path.basename(filepath)}: {e}")

    # ==========================================
    # 3. PRINT DIAGNOSTICS
    # ==========================================
    print(f"✅ Loaded {len(GOLDEN_SEED_CACHE)}/26 SFT Golden Seeds.")
    print(f"✅ Loaded {len(DPO_GOLDEN_SEED_CACHE)}/26 DPO Golden Seeds.")
    
    if len(GOLDEN_SEED_CACHE) < 26 or len(DPO_GOLDEN_SEED_CACHE) < 26:
        print("⚠️ Warning: Incomplete Golden Seed cache. Missing categories will fallback to 0-Shot generation.")

def run_audit_forge():
    print("\n" + "="*80)
    print("⚖️ INITIATING CATEGORY-FIRST AUDIT FORGE (DUAL 1-SHOT CACHING)")
    print(f"⚙️ Config: {BATCH_SIZE} items/batch | {VARIATIONS_PER_CATEGORY} vars/category | {len(CATEGORY_ALIAS_MAP)} categories")
    print("="*80)
    
    stats = {"total_generated": 0, "saved_sft": 0, "saved_dpo": 0, "dropped_total": 0}
    cumulative_drop_reasons = defaultdict(int)
    
    # 🚨 OUTER LOOP: Lock in 1 category at a time to maximize KV-Cache hits
    for cat_idx, (cat_enum, dict_key) in enumerate(CATEGORY_ALIAS_MAP.items()):
        print(f"\n🎯 [CATEGORY {cat_idx+1:02d}/{len(CATEGORY_ALIAS_MAP)}] {cat_enum}")
        
        # 1. Load the Fixed Prefix Components for this Category
        law_chunk = get_audit_rag_context(dict_key)
        
        # 🧠 SFT SEED INJECTION (For Synthesizer & Judge)
        seed_content = GOLDEN_SEED_CACHE.get(dict_key, "")
        seed_block = f"\n\n--- 1-SHOT SFT GOLDEN SEED REFERENCE ---\n{seed_content}\n--- END SFT REFERENCE ---\n" if seed_content else ""
        
        # 🧠 DPO SEED INJECTION (For the Hard Negative Generator)
        dpo_seed_data = DPO_GOLDEN_SEED_CACHE.get(dict_key, {})
        if dpo_seed_data:
            dpo_seed_block = f"""
--- 1-SHOT DPO GOLDEN SEED REFERENCE ---
CHOSEN (Perfect Audit):
{dpo_seed_data['chosen']}

REJECTED (Overzealous/Hallucinated Audit):
{dpo_seed_data['rejected']}
--- END DPO REFERENCE ---
"""
        else:
            dpo_seed_block = ""
        
        # 2. Build the exact matrix for THIS category
        category_matrix = []
        
        # 🚨 TITANIUM UPGRADE: Dynamic Silo Cutoffs
        silo_1_cutoff = int(VARIATIONS_PER_CATEGORY * 0.30)
        silo_2_cutoff = int(VARIATIONS_PER_CATEGORY * 0.80)
        
        for item_idx in range(VARIATIONS_PER_CATEGORY):
            variation_idx = item_idx + 1 
            
            # SILO 1-2-3 LOGIC (Dynamic Math + Trigger Visibility Upgrades)
            if item_idx < silo_1_cutoff: # 30% Silo 1
                silo_level = "SILO_1"
                silo_directive = """
SILO 1 DIRECTIVE (BASELINE): Write a standard, clean privacy policy. 
Keep paragraphs concise and straightforward. Do not overcomplicate the legalese. 
Make the target violation relatively clear and easy to read.
""".strip()
            elif item_idx < silo_2_cutoff: # 50% Silo 2
                silo_level = "SILO_2"
                silo_directive = """
SILO 2 DIRECTIVE (CAMOUFLAGE): Write a highly nuanced, corporate policy.
Bury the target violation deep inside dense operational legalese. Surround the trap with justifications claiming it benefits the user. 
CRITICAL RULE: The surrounding context must be camouflaged, but the specific legal action (the trap itself) must remain an undeniable, absolute corporate command.
""".strip()
            else: # 20% Silo 3
                silo_level = "SILO_3"
                silo_directive = """
SILO 3 DIRECTIVE (ADVERSARIAL): Write a brutally complex, adversarial policy.
Use massive run-on sentences, overwhelming technical/legal jargon, and hostile formatting. Push the boundaries of cognitive complexity to disorient the reader.
CRITICAL RULE: Amidst the chaos, the illegal corporate action must be stated as an absolute, undeniable fact. Do not dilute the violation with "may" or "might".
""".strip()

            pass_type = "passive" if random.random() < 0.30 else "active_mutation"
            is_hn = random.random() < 0.15 if pass_type == "passive" else False
            
            valid_templates = [t for t in EDGE_CASE_TEMPLATES if dict_key in t.get("target_categories", []) or cat_enum in t.get("target_categories", [])]
            chosen_template = random.choice(valid_templates) if valid_templates and random.random() < 0.40 else None

            # Dynamic Industry for Silo 3
            if silo_level == "SILO_3":
                crazy_industries = [
                    "deep-sea drilling telemetry systems",
                    "neuro-link biotech and neural implant analytics",
                    "quantum cryptography and sub-atomic communication logs",
                    "orbital satellite debris tracking",
                    "autonomous military drone behavioral profiling"
                ]
                industry_val = random.choice(crazy_industries)
            else:
                industry_val = random.choice(list(INDUSTRIES.keys()))

            category_matrix.append({
                "variation_idx": variation_idx,
                "category_idx": cat_idx,
                "target_category": cat_enum,
                "pass_type": pass_type,
                "is_hn": is_hn,
                "base_policy": random.choice(raw_policies),
                "industry": industry_val,
                "seed": random.choice(indian_seeds),
                "sft_golden_seed": seed_content,
                "edge_template": chosen_template if pass_type == "active_mutation" else None,
                "subtlety": random.randint(40, 90),
                "silo_level": silo_level,
                "silo_directive": silo_directive
            })
            
        # 3. INNER LOOP: Batch execution
        total_local_batches = math.ceil(len(category_matrix) / BATCH_SIZE)
        
        for local_batch_idx in range(total_local_batches):
            batch_start = local_batch_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(category_matrix))
            batch = category_matrix[batch_start:batch_end]
            
            # --- SYNTHESIZE POLICIES ---
            gen_messages = []
            target_tracking = {}  
            
            for item in batch:
                prompt_text, is_hn, active_ex, target_cats = build_dynamic_synthesizer_prompt(item)
                item["is_hn"] = is_hn
                item["active_exemption"] = active_ex
                target_tracking[item["variation_idx"]] = target_cats 
                
                persona = "Adversarial corporate counsel." if item["pass_type"] == "active_mutation" else "Meticulous corporate compliance officer."
                gen_messages.append([{"role": "system", "content": persona}, {"role": "user", "content": prompt_text}])
                
            gen_out = llm.chat(messages=gen_messages, sampling_params=gen_params)
            current_policies = {i: extract_policy(out.outputs[0].text.strip()) for i, out in enumerate(gen_out)}
            
            completed = set()
            batch_drop_reasons = defaultdict(int)
            
            # --- UNIFIED REFLEXION PIPELINE ---
            for step in range(MAX_REFLEXION_STEPS):
                remaining = [i for i in range(len(batch)) if i not in completed]
                if not remaining: break
                
                # A. Judge Generation (Uses SFT Seed & Dynamic Target Tracking)
                judge_msgs = []
                for i in remaining:
                    item = batch[i]
                    pass_type = item["pass_type"]
                    
                    if pass_type == "active_mutation":
                        active_targets = target_tracking.get(item["variation_idx"], [cat_enum])
                        target_violation_types_str = ", ".join(active_targets)
                        
                        statute_texts = []
                        for t in active_targets:
                            dict_k = [key for enum, key in CATEGORY_ALIAS_MAP.items() if enum == t][0]
                            statute_texts.append(f"For {t}:\n" + "\n".join(TARGET_VIOLATIONS[dict_k]))
                        target_statute_str = "\n\n".join(statute_texts)
                    else:
                        target_violation_types_str = "NONE (Clean Baseline Expected)"
                        target_statute_str = "Maintain 100% DPDP Compliance. Do not hallucinate."
                    
                    prompt = JUDGE_PROMPT.replace("[JUDGE_PERSONA_INJECTION]", random.choice(JUDGE_PERSONAS)) \
                                         .replace("[RETRIEVED_LAW_CONTEXT]", law_chunk) \
                                         .replace("[GOLDEN_SEED_INJECTION]", seed_block) \
                                         .replace("[TARGET_STATUTE]", target_statute_str) \
                                         .replace("[TARGET_VIOLATION_TYPE]", target_violation_types_str) \
                                         .replace("[POLICY_INJECTION]", current_policies[i][:20000])
                                         
                    judge_msgs.append([{"role": "system", "content": "Strict DPDP Auditor."}, {"role": "user", "content": prompt}])
                    
                audit_outputs = llm.chat(messages=judge_msgs, sampling_params=judge_params)
                
                # B. Hard Negative Generation (Uses DPO Seed)
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
                    
                    active_targets = target_tracking.get(item["variation_idx"], [cat_enum])
                    banned_cats = ", ".join(active_targets)
                    true_violation_context = f"Banned SFT Target Categories: {banned_cats}\nBanned SFT Evidence Quote (DO NOT USE): {chosen_quote}" if chosen_quote else "NONE (Clean baseline)"
                        
                    hn_prompt = HARD_NEGATIVE_PROMPT \
                                    .replace("[RETRIEVED_LAW_CONTEXT]", law_chunk) \
                                    .replace("[DPO_GOLDEN_SEED_INJECTION]", dpo_seed_block) \
                                    .replace("[POLICY_INJECTION]", current_policies[idx][:20000]) \
                                    .replace("[TRUE_VIOLATION_CONTEXT]", true_violation_context)
                                    
                    hn_msgs.append([{"role": "system", "content": "Authoritative, overzealous privacy auditor."}, {"role": "user", "content": hn_prompt}])
                    
                hn_audit_outputs = llm.chat(messages=hn_msgs, sampling_params=judge_params)
                
                # C. Verification & Healing Logic
                explicit_heal = []
                explicit_idx = []
                
                for idx, out_judge, out_hn in zip(remaining, audit_outputs, hn_audit_outputs):
                    item = batch[idx]
                    policy = current_policies[idx]
                    pass_type = item["pass_type"]
                    is_hn = item["is_hn"]
                    
                    # 🚨 TITANIUM UPGRADE: Parse & Sanitize Muscle Memory (SFT)
                    try:
                        audit = strip_keys(safe_parse_audit(out_judge.outputs[0].text.strip()))
                        
                        # Fix Compliant Policy "No" Muscle Memory
                        if not audit.get("violations") and audit.get("global_legal_reasoning", "").startswith("No"):
                            audit["global_legal_reasoning"] = audit["global_legal_reasoning"].replace(
                                "No explicit", "The policy contains no explicit"
                            )
                            
                        is_valid_sft, error_msg_sft = validate_audit_quality(audit, policy, is_dpo=False)
                    except Exception as e:
                        is_valid_sft, error_msg_sft, audit = False, f"JSON parse error (SFT): {str(e)[:40]}", {}

                    # 🚨 TITANIUM UPGRADE: Parse & Sanitize Muscle Memory (DPO)
                    try:
                        hn_audit = strip_keys(safe_parse_audit(out_hn.outputs[0].text.strip()))
                        
                        # Fix DPO "A rigorous" Muscle Memory
                        if hn_audit.get("global_legal_reasoning", "").startswith("A "):
                            hn_audit["global_legal_reasoning"] = "The " + hn_audit["global_legal_reasoning"][2:]
                            
                        is_valid_dpo, error_msg_dpo = validate_audit_quality(hn_audit, policy, is_dpo=True, chosen_audit=audit)
                    except Exception as e:
                        is_valid_dpo, error_msg_dpo, hn_audit = False, f"JSON parse error (DPO): {str(e)[:40]}", {}

                    # GATE 1: Parse and Validation
                    if not (is_valid_sft and is_valid_dpo):
                        if step < MAX_REFLEXION_STEPS - 1:
                            err_reasons = []
                            if not is_valid_sft: err_reasons.append(f"SFT Error: {error_msg_sft}")
                            if not is_valid_dpo: err_reasons.append(f"DPO Error: {error_msg_dpo}")
                            
                            active_targets = target_tracking.get(item["variation_idx"], [cat_enum])
                            compiled_obj = f"Targets: {', '.join(active_targets)}" if pass_type == "active_mutation" else "Maintain absolute DPDP compliance."
                            
                            heal_prompt = REFLEXION_EXPLICIT_PROMPT \
                                .replace("[TARGET_VIOLATION]", compiled_obj) \
                                .replace("[AUDIT_FEEDBACK]", f"ERROR: {' | '.join(err_reasons)}. FIX IMMEDIATELY.") \
                                .replace("[INDUSTRY_INJECTION]", INDUSTRIES.get(item["industry"], str(item.get("industry", "")))) \
                                .replace("[SEED_INJECTION]", str(item["seed"])[:1000]) \
                                .replace("[FAILED_POLICY_INJECTION]", policy[:15000]) \
                                .replace("[SILO_COMPLEXITY_DIRECTIVE]", item.get("silo_directive", ""))
                                
                            heal_persona = "Adversarial corporate counsel." if pass_type == "active_mutation" else "Meticulous corporate compliance officer."
                            explicit_heal.append([{"role": "system", "content": heal_persona}, {"role": "user", "content": heal_prompt}])
                            explicit_idx.append(idx)
                        else:
                            if not is_valid_sft: batch_drop_reasons[f"Failed SFT Audit ({error_msg_sft})"] += 1
                            if not is_valid_dpo: batch_drop_reasons[f"Failed DPO Audit ({error_msg_dpo})"] += 1
                            completed.add(idx)
                        continue
                    
                    # GATE 2: 2-Pass Intent Validation
                    has_violations = len(audit.get("violations", [])) > 0
                    hn_has_violations = len(hn_audit.get("violations", [])) > 0
                    
                    success = False
                    heal_error = ""
                    
                    if pass_type == "passive" and not is_hn:
                        success = not has_violations
                        heal_error = "The policy contains accidental violations. Rewrite it to be 100% compliant."
                    elif pass_type == "passive" and is_hn:
                        success = not has_violations and hn_has_violations
                        heal_error = "The exemption trap failed. Make the exemption perfectly legal but deceptive."
                    else:
                        success = has_violations
                        heal_error = "The Judge MISSED the trap. Rewrite to make the violation undeniable but camouflaged."
                            
                    if success:
                        # 🚨 DPO PAYLOAD LOGIC: PRESERVE PERFECT POLICIES
                        dpo_rejected_payload = None
                        
                        if is_valid_dpo and hn_has_violations:
                            dpo_rejected_payload = hn_audit
                        else:
                            if step < MAX_REFLEXION_STEPS - 1:
                                continue
                            else:
                                batch_drop_reasons["Failed DPO Generation (No distinct HN). Saved SFT only."] += 1
                        
                        # --- SAVE SFT ---
                        res_sft = _save_closed_book_audit_pair(item["variation_idx"], item["category_idx"], item, policy, audit, None, is_dpo=False)
                        if res_sft == "sft":
                            stats["saved_sft"] += 1
                            viols = audit.get("violations", [])
                            if viols and isinstance(viols, list) and isinstance(viols[0], dict):
                                ACCEPTED_JUSTIFICATION_BUFFER.append(viols[0].get("step_3_semantic_justification", ""))
                        
                        # --- SAVE DPO ---
                        if dpo_rejected_payload:
                            res_dpo = _save_closed_book_audit_pair(item["variation_idx"], item["category_idx"], item, policy, audit, dpo_rejected_payload, is_dpo=True)
                            if res_dpo == "dpo":
                                stats["saved_dpo"] += 1
                                dpo_viols = dpo_rejected_payload.get("violations", [])
                                if dpo_viols and isinstance(dpo_viols, list) and isinstance(dpo_viols[0], dict):
                                    ACCEPTED_JUSTIFICATION_BUFFER.append(dpo_viols[0].get("step_3_semantic_justification", ""))
                        
                        completed.add(idx)     
                    else:
                        if step < MAX_REFLEXION_STEPS - 1:
                            active_targets = target_tracking.get(item["variation_idx"], [cat_enum])
                            compiled_obj = f"Targets: {', '.join(active_targets)}" if pass_type == "active_mutation" else "Maintain absolute DPDP compliance."
                            
                            heal_prompt = REFLEXION_EXPLICIT_PROMPT \
                                .replace("[TARGET_VIOLATION]", compiled_obj) \
                                .replace("[AUDIT_FEEDBACK]", f"ERROR: {heal_error}") \
                                .replace("[INDUSTRY_INJECTION]", INDUSTRIES.get(item["industry"], str(item.get("industry", "")))) \
                                .replace("[SEED_INJECTION]", str(item["seed"])[:1000]) \
                                .replace("[FAILED_POLICY_INJECTION]", policy[:15000]) \
                                .replace("[SILO_COMPLEXITY_DIRECTIVE]", item.get("silo_directive", ""))
                            
                            heal_persona = "Adversarial corporate counsel." if pass_type == "active_mutation" else "Meticulous corporate compliance officer."
                            explicit_heal.append([{"role": "system", "content": heal_persona}, {"role": "user", "content": heal_prompt}])
                            explicit_idx.append(idx)
                        else:
                            batch_drop_reasons[f"Architectural Intent Failed: {heal_error}"] += 1
                            completed.add(idx)
                            
            # Execute Healing Generation
            if explicit_heal:
                heal_temp = max(0.65, 0.95 - 0.15 * step)
                heal_params = SamplingParams(temperature=heal_temp, top_p=0.9, max_tokens=8192)
                out_explicit = llm.chat(messages=explicit_heal, sampling_params=heal_params)
                for idx, o in zip(explicit_idx, out_explicit): 
                    current_policies[idx] = extract_policy(o.outputs[0].text.strip()) or current_policies[idx]

            # Granular Update & Logging
            batch_drops_total = sum(batch_drop_reasons.values())
            stats["dropped_total"] += batch_drops_total
            for reason, count in batch_drop_reasons.items():
                cumulative_drop_reasons[reason] += count

            print(f"   ✅ [BATCH {local_batch_idx+1}/{total_local_batches}] | Category: {cat_enum} | SFT: {stats['saved_sft']} | DPO: {stats['saved_dpo']}")
            print(f"      📉 Current Batch Drops: {batch_drops_total} | Cumulative Drops: {stats['dropped_total']}")
            
            if batch_drop_reasons:
                print("      [Current Batch Drop Reasons]:")
                for reason, count in batch_drop_reasons.items():
                    print(f"       - {count}x: {reason}")
            
        gc.collect()

    print("\n" + "="*80)
    print(f"🏁 FORGE COMPLETE | Total SFT: {stats['saved_sft']} | Total DPO: {stats['saved_dpo']} | Total Dropped: {stats['dropped_total']}")
    if cumulative_drop_reasons:
        print("📊 [CUMULATIVE DROP BREAKDOWN]:")
        for reason, count in sorted(cumulative_drop_reasons.items(), key=lambda item: item[1], reverse=True):
            print(f"  - {count}x: {reason}")
    print("="*80)

def validate_chatbot_content(messages: list, strict_geometry: bool = True) -> tuple[bool, str]:
    """
    Validates the text content and strict geometry of the chatbot SFT/DPO outputs.
    strict_geometry=True enforces the 4-turn SFT matrix.
    strict_geometry=False bypasses the turn count for isolated DPO chosen/rejected turns.
    """
    
    # 🚨 GATE 1: STRICT GEOMETRY ENFORCEMENT (Bypassed for individual DPO turns)
    if strict_geometry:
        if len(messages) != 4:
            return False, f"Invalid turn count: {len(messages)} (Expected exactly 4)"
        
        expected_roles = ["user", "assistant", "user", "assistant"]
        for i, expected_role in enumerate(expected_roles):
            actual_role = messages[i].get("role")
            if actual_role != expected_role:
                return False, f"Role mismatch at turn {i+1}: Expected '{expected_role}', got '{actual_role}'"

    # 🚨 GATE 2: CONTENT POISON & HALLUCINATION CHECK
    forbidden_laws = [
        "gdpr", "ccpa", "hipaa", "lgpd", "pdpa", "privacy rights act", 
        "article 17", "right to portability", "legitimate interest", 
        "it act 2000", "information technology act", "section 43a", "spdi rules 2011"
    ]
    
    omission_phrases = [
        "without specifying", "fails to specify", "does not specify", 
        "fails to provide", "does not provide", "fails to detail", 
        "does not detail", "silent on", "no mention of"
    ]
    
    ai_disclaimers = [
        "as an ai", "i am not a lawyer", "not legal advice", 
        "consult a professional", "i cannot provide legal advice", 
        "for informational purposes only"
    ]
    
    for m in messages:
        content = str(m.get("content", "")).lower()
        
        # Universal Poison Checks
        if check_string_poison(content): 
            return False, "Placeholder or structural tag leak detected."
            
        if any(law in content for law in forbidden_laws): 
            return False, "Foreign/legacy law bleed detected in dialogue."
            
        if any(disclaimer in content for disclaimer in ai_disclaimers):
            return False, "AI disclaimer/hedging detected."
            
        # Assistant-Specific Checks
        if m.get("role") == "assistant":
            if any(phrase in content for phrase in omission_phrases):
                return False, "Omission hallucination detected in assistant response."
            
            # Ensure the RAG context tags don't leak into the model's actual speech
            if "[context: the law]" in content or "[task]" in content or "[retrieved_law_context]" in content:
                return False, "RAG Context tag leaked into assistant dialogue."
                
    return True, ""

def run_chatbot_forge():
    print("\n" + "="*80)
    print("💬 INITIATING BATCHED CHATBOT QA FORGE (OPEN-BOOK ARCHITECTURE)")
    print("="*80)
    
    matrix = build_chatbot_matrix()
    total_batches = math.ceil(len(matrix) / BATCH_SIZE)
    stats = {"saved_chatbot_sft": 0, "saved_chatbot_dpo": 0, "dropped": 0}
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(matrix))
        batch = matrix[batch_start:batch_end]
        
        batch_drop_reasons = defaultdict(int)
        
        # 1. Prepare Base Prompts (Stored separately to prevent infinite Reflexion growth)
        base_sft_prompts = []
        base_dpo_prompts = []
        
        for item in batch:
            prompt_sft = CHATBOT_QA_SFT_PROMPT \
                .replace("[RETRIEVED_LAW_CONTEXT]", item["law_context"]) \
                .replace("[PERSONA_INJECTION]", item["persona"]) \
                .replace("[SCENARIO_INJECTION]", item["scenario"])
            base_sft_prompts.append(prompt_sft)
            
            prompt_dpo = CHATBOT_QA_DPO_PROMPT \
                .replace("[RETRIEVED_LAW_CONTEXT]", item["law_context"]) \
                .replace("[PERSONA_INJECTION]", item["persona"]) \
                .replace("[SCENARIO_INJECTION]", item["scenario"])
            base_dpo_prompts.append(prompt_dpo)
            
        max_retries = 2
        active_indices = list(range(len(batch)))
        final_sft_results = [None] * len(batch)
        final_dpo_results = [None] * len(batch)
        
        # Store feedback for reflexion
        retry_feedback_sft = {}
        retry_feedback_dpo = {}
        
        # 🚨 REFLEXION RETRY LOOP
        for attempt in range(max_retries):
            if not active_indices: 
                break
                
            # Build dynamic message arrays for the active indices
            curr_sft_msgs = []
            curr_dpo_msgs = []
            
            for orig_idx in active_indices:
                sft_content = base_sft_prompts[orig_idx]
                dpo_content = base_dpo_prompts[orig_idx]
                        
                if attempt > 0:
                    if orig_idx in retry_feedback_sft:
                        sft_content += f"\n\n[REFLEXION FEEDBACK]: {retry_feedback_sft[orig_idx]}"
                    if orig_idx in retry_feedback_dpo:
                        dpo_content += f"\n\n[REFLEXION FEEDBACK]: {retry_feedback_dpo[orig_idx]}"
                        
                # ✅ TWEAK: Added the System Role for Tokenizer compatibility
                sys_msg = {"role": "system", "content": "You are an expert AI dataset synthesizer."}
                curr_sft_msgs.append([sys_msg, {"role": "user", "content": sft_content}])
                curr_dpo_msgs.append([sys_msg, {"role": "user", "content": dpo_content}])                
            
            sft_out = llm.chat(messages=curr_sft_msgs, sampling_params=chatbot_sft_params)
            dpo_out = llm.chat(messages=curr_dpo_msgs, sampling_params=chatbot_dpo_params)
            
            next_active = []
            # Clear feedback dictionaries for the next potential retry
            retry_feedback_sft = {}
            retry_feedback_dpo = {}
            
            for batch_pos, orig_idx in enumerate(active_indices):
                s_o = sft_out[batch_pos]
                d_o = dpo_out[batch_pos]
                
                try:
                    s_text = s_o.outputs[0].text.strip()
                    d_text = d_o.outputs[0].text.strip()
                    
                    # Gate 0: String Poison Check
                    if check_string_poison(s_text) or check_string_poison(d_text):
                        if attempt == max_retries - 1:
                            stats["dropped"] += 1
                            batch_drop_reasons["String poison/leak detected"] += 1
                        else:
                            retry_feedback_sft[orig_idx] = "Your output contained leaked structural tags or placeholders."
                            retry_feedback_dpo[orig_idx] = "Your output contained leaked structural tags or placeholders."
                            next_active.append(orig_idx)
                        continue

                    # Gate 1: JSON Parsing
                    parsed_sft = safe_parse_audit(s_text)
                    parsed_dpo = safe_parse_audit(d_text)
                    
                    if not isinstance(parsed_sft, dict) or "messages" not in parsed_sft:
                        raise ValueError("SFT missing 'messages' key")
                    if not isinstance(parsed_dpo, dict) or not all(k in parsed_dpo for k in ["prompt", "chosen", "rejected"]):
                        raise ValueError("DPO missing required keys")

                    # Gate 2: Deep Content & Geometry Validation
                    # SFT must have exactly 4 turns (strict_geometry=True by default)
                    sft_valid, sft_reason = validate_chatbot_content(parsed_sft["messages"])
                    
                    # For DPO, validate the chosen and rejected assistant turns (Bypass 4-turn rule)
                    chosen_text = parsed_dpo["chosen"][0].get("content", "") if parsed_dpo["chosen"] else ""
                    rejected_text = parsed_dpo["rejected"][0].get("content", "") if parsed_dpo["rejected"] else ""
                    
                    dpo_valid = True
                    dpo_reason = ""
                    for role_text in [chosen_text, rejected_text]:
                        dummy_msg = [{"role": "assistant", "content": role_text}]
                        # ✅ FIX: Bypass the 4-turn geometry check for isolated DPO turns
                        v, r = validate_chatbot_content(dummy_msg, strict_geometry=False) 
                        if not v:
                            dpo_valid = False
                            dpo_reason = r
                            break

                    if not sft_valid or not dpo_valid:
                        if attempt == max_retries - 1:
                            stats["dropped"] += 1
                            if not sft_valid: batch_drop_reasons[f"SFT Validation: {sft_reason}"] += 1
                            if not dpo_valid: batch_drop_reasons[f"DPO Validation: {dpo_reason}"] += 1
                        else:
                            # Inject Reflexion Feedback for the next attempt
                            retry_feedback_sft[orig_idx] = sft_reason if not sft_valid else "DPO structure failed validation."
                            retry_feedback_dpo[orig_idx] = dpo_reason if not dpo_valid else "SFT structure failed validation."
                            next_active.append(orig_idx)
                        continue
                    
                    # SUCCESS: Mark as completed and store results
                    final_sft_results[orig_idx] = parsed_sft
                    final_dpo_results[orig_idx] = parsed_dpo
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        stats["dropped"] += 1
                        batch_drop_reasons[f"Parse/Execution Error: {str(e)[:50]}"] += 1
                    else:
                        retry_feedback_sft[orig_idx] = f"JSON parsing failed: {str(e)[:40]}. Output ONLY valid JSON."
                        retry_feedback_dpo[orig_idx] = f"JSON parsing failed: {str(e)[:40]}. Output ONLY valid JSON."
                        next_active.append(orig_idx)
            
            active_indices = next_active
            
        # 3. SAVE SUCCESSFUL PAIRS (OPEN-BOOK PROTOCOL)
        for i in range(len(batch)):
            if final_sft_results[i] and final_dpo_results[i]:
                parsed_sft = final_sft_results[i]
                parsed_dpo = final_dpo_results[i]
                
                local_idx = i
                
                # --- SAVE TRACK B SFT ---
                sft_filepath = os.path.join(CHATBOT_SFT_DIR, f"chat_sft_{batch_idx:03d}_{local_idx:02d}.json")
                with open(sft_filepath, "w", encoding="utf-8") as f:
                    json.dump(parsed_sft, f, ensure_ascii=False, indent=2)
                with open(JSONL_CHATBOT_SFT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(parsed_sft, ensure_ascii=False) + "\n")
                stats["saved_chatbot_sft"] += 1
                
                # --- SAVE TRACK B DPO ---
                dpo_filepath = os.path.join(CHATBOT_DPO_DIR, f"chat_dpo_{batch_idx:03d}_{local_idx:02d}.json")
                with open(dpo_filepath, "w", encoding="utf-8") as f:
                    json.dump(parsed_dpo, f, ensure_ascii=False, indent=2)
                with open(JSONL_CHATBOT_DPO, "a", encoding="utf-8") as f:
                    f.write(json.dumps(parsed_dpo, ensure_ascii=False) + "\n")
                stats["saved_chatbot_dpo"] += 1

        print(f"   ✅ Chatbot Batch {batch_idx + 1}/{total_batches} complete | SFT: {stats['saved_chatbot_sft']} | DPO: {stats['saved_chatbot_dpo']} | Dropped: {stats['dropped']}", flush=True)
        if batch_drop_reasons:
            for reason, count in batch_drop_reasons.items():
                print(f"      - {count}x: {reason}", flush=True)
        gc.collect()

def run_post_generation_analysis():
    print("\n" + "="*70)
    print("📊 RUNNING POST-GENERATION DATA QUALITY FORENSIC SCAN")
    print("="*70)
    
    sft_files = glob.glob(os.path.join(SFT_OUTPUT_DIR, "*.json"))
    dpo_files = glob.glob(os.path.join(DPO_OUTPUT_DIR, "*.json"))

    print(f"Scanning Track A SFT dataset: {len(sft_files)} files...")
    print(f"Scanning Track A DPO dataset: {len(dpo_files)} files...")

    poison_counts = {
        "placeholders_in_policy": 0,
        "placeholders_in_audit": 0,
        "foreign_law_bleed": 0,
        "omission_hallucination": 0,
        "ellipsis_in_quote": 0,
        "administrative_element_flagged": 0,
        "quote_not_in_policy": 0,
        "invalid_violation_enum": 0,
        "score_out_of_bounds": 0,
        "statute_stretching": 0,
        "canned_global_reasoning_dpo_rejected": 0,
        "canned_semantic_justification_dpo_rejected": 0
    }

    foreign_laws = ["gdpr", "ccpa", "hipaa", "lgpd", "pdpa", "privacy rights act", "article 17", "right to portability", "legitimate interest", "it act 2000", "information technology act", "section 43a", "spdi rules 2011"]
    omission_phrases = ["without specifying", "fails to specify", "does not specify", "fails to provide", "does not provide", "fails to detail", "does not detail", "silent on", "no mention of"]
    valid_enums = list(CATEGORY_ALIAS_MAP.keys())

    def extract_policy_text(user_content: str) -> str:
        """Safely extracts policy text from both new Closed-Book and legacy prompts."""
        if "## Privacy Policy\n" in user_content:
            return user_content.split("## Privacy Policy\n")[1]
        elif "Analyze:\n" in user_content:
            return user_content.split("Analyze:\n")[1]
        return user_content

    def check_omissions(text: str) -> bool:
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in omission_phrases)

    def check_statute_stretching(vtype: str, statute: str) -> bool:
        statute_lower = statute.lower()
        if vtype == "LOG_RETENTION_MANDATE_VIOLATION":
            return "rule 8" not in statute_lower and "rule 6" not in statute_lower
        if vtype in ["ALGORITHMIC_PROFILING_SDF", "SDF_OBLIGATIONS_MISSING", "SDF_DATA_LOCALIZATION_VIOLATION"]:
            return "section 10" not in statute_lower and "rule 13" not in statute_lower
        return False

    # ==========================================
    # SFT FORENSIC SCAN
    # ==========================================
    for fp in sft_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            if len(messages) < 3: continue
            
            user_content = messages[1]["content"]
            assistant_content = messages[2]["content"]
            
            policy_text = extract_policy_text(user_content)
            
            if check_string_poison(policy_text):
                poison_counts["placeholders_in_policy"] += 1
                
            audit = json.loads(assistant_content)
            if check_string_poison(str(audit)):
                poison_counts["placeholders_in_audit"] += 1
                
            global_reasoning = audit.get("global_legal_reasoning", "")
            if any(law in global_reasoning.lower() for law in foreign_laws):
                poison_counts["foreign_law_bleed"] += 1
            if check_omissions(global_reasoning):
                poison_counts["omission_hallucination"] += 1
                
            # Score Bounds Check
            if not (0 <= audit.get("dpdp_trust_score", -1) <= 100) or not (0 <= audit.get("subtlety_score", -1) <= 100):
                poison_counts["score_out_of_bounds"] += 1

            for v in audit.get("violations", []):
                # Enum Check
                if v.get("violation_type") not in valid_enums:
                    poison_counts["invalid_violation_enum"] += 1
                    
                # Statute Stretching Check
                if check_statute_stretching(v.get("violation_type", ""), v.get("statute_reference", "")):
                    poison_counts["statute_stretching"] += 1

                quote = v.get("evidence_quote", "")
                if not quote: continue
                if "..." in quote or "\u2026" in quote:
                    poison_counts["ellipsis_in_quote"] += 1
                if is_administrative_element(quote):
                    poison_counts["administrative_element_flagged"] += 1
                if not is_quote_in_policy(quote, policy_text):
                    poison_counts["quote_not_in_policy"] += 1
                    
                if check_omissions(v.get("step_3_semantic_justification", "")):
                    poison_counts["omission_hallucination"] += 1
                    
        except Exception:
            pass

    # ==========================================
    # DPO FORENSIC SCAN
    # ==========================================
    global_reasoning_dpo_rejected = []
    semantic_justification_dpo_rejected = []

    for fp in dpo_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            prompt = data.get("prompt", [])
            if len(prompt) < 2: continue
            
            user_content = prompt[1]["content"]
            policy_text = extract_policy_text(user_content)
                
            if check_string_poison(policy_text):
                poison_counts["placeholders_in_policy"] += 1
                
            rejected_content = data.get("rejected", [{}])[0].get("content", "")
            rejected_audit = json.loads(rejected_content)
            
            rejection_reasoning = rejected_audit.get("global_legal_reasoning", "")
            global_reasoning_dpo_rejected.append(rejection_reasoning)
            
            if check_omissions(rejection_reasoning):
                poison_counts["omission_hallucination"] += 1

            for v in rejected_audit.get("violations", []):
                just = v.get("step_3_semantic_justification", "")
                semantic_justification_dpo_rejected.append(just)
                
                if check_omissions(just):
                    poison_counts["omission_hallucination"] += 1

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
                    
                if check_statute_stretching(v.get("violation_type", ""), statute):
                    poison_counts["statute_stretching"] += 1
                    
        except Exception:
            pass

    # Canned Text Detection
    if global_reasoning_dpo_rejected:
        canned_reasoning = "Authoritative forensic evaluation arguing that the company's invoked statutory exemptions and retention caveats under Section 17 and Section 8 exceed statutory bounds under the Digital Personal Data Protection Act 2023."
        poison_counts["canned_global_reasoning_dpo_rejected"] = global_reasoning_dpo_rejected.count(canned_reasoning)

    if semantic_justification_dpo_rejected:
        canned_just_pattern = r"By stating the exact quoted text, the company actively asserts an overbroad rule which directly contravenes Section \d+ because it bypasses explicit statutory requirements without satisfying the narrow prerequisites for legitimate or exempt processing\."
        for just in semantic_justification_dpo_rejected:
            if re.search(canned_just_pattern, just): # Changed from re.match to re.search for robustness
                poison_counts["canned_semantic_justification_dpo_rejected"] += 1

    print("\n[Forensic Quality Scan Results]")
    for k, v in poison_counts.items():
        status = "✅" if v == 0 else "🚨"
        print(f" {status} {k}: {v}")
    
    print("\nData scan completed successfully.", flush=True)
    
if __name__ == "__main__":
    try:
        print("🚀 Booting Ssense GAN Forge...")
        
        # 1. Initialize Engine
        init_llm()
        
        # 2. Load 1-Shot Seeds into Memory
        load_golden_seeds()
        
        # 3. Launch Track A (Category-First Architecture)
        run_audit_forge()
        
        # 4. Launch Track B (Chatbot)
        run_chatbot_forge()
        
        print("\n🏆 COMPLETE DUAL-TRACK FORGE FINISHED SUCCESSFULLY")
        
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted.")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
    finally:
        if 'llm' in globals() and llm is not None:
            del llm
        gc.collect()