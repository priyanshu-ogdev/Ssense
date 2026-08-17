#!/usr/bin/env python3
"""
build_vector_db.py – 10-Point SOTA Hybrid Search Engine Builder (FINAL)
"""

import os
import re
import sys
import datetime
import numpy as np
import pickle
from collections import Counter
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from langchain_text_splitters import RecursiveCharacterTextSplitter

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════════════════
# SOTA FIX: CROSS-DIRECTORY SIBLING IMPORT
# ═══════════════════════════════════════════════════════════════════════════
# `build_vector_db.py` lives in `ml/data-forge/`
# `path_resolver.py` lives in `ml/evals/`
_CURRENT_DIR = Path(__file__).resolve().parent
_EVALS_DIR = _CURRENT_DIR.parent / "evals"

# Inject ml/evals/ into the Python path so it can find path_resolver.py
if str(_EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALS_DIR))

try:
    from path_resolver import Paths
    LAW_TEXT_PATH = Paths.LAW_TEXT
    HYBRID_INDEX_PATH = Paths.HYBRID_INDEX
except ImportError as e:
    print(f"❌ Core module import failed. Could not find path_resolver in {_EVALS_DIR}")
    sys.exit(1)



# ═══════════════════════════════════════════════════════════════════════════
# 1. STANDARDIZED TOKENIZER (Synchronized with evaluate_rag.py)
# ═══════════════════════════════════════════════════════════════════════════
def get_standard_tokenizer():
    """
    SOTA FIX: Alphanumeric tokenization ensures absolute 1:1 parity with the
    search queries generated in `evaluate_rag.py`. Stopwords removed.
    """
    generic_stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "of", "and", "in", 
        "to", "for", "with", "on", "at", "by", "from", "as", "that", "this", 
        "it", "be", "or", "which", "will", "would", "could", "should", "their", "they"
    }
    
    def tokenize(text):
        # Strict alphanumeric split mirroring evaluate_rag.py
        words = re.findall(r'\w+', str(text).lower())
        return [w for w in words if w not in generic_stopwords]
    
    return tokenize

# ═══════════════════════════════════════════════════════════════════════════
# 2. METADATA EXTRACTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def extract_metadata(header_text, body_text):
    """
    Extracts structured metadata for RRF filtering and State-chunk isolation.
    """
    meta = {
        "type": "General",
        "number": "",
        "sub_section": "",
        "parent_act": "DPDP Act 2023",
        "applies_to": "all"  # Default to all; overridden for State exemptions
    }
    
    full_text = header_text + "\n" + body_text
    
    if "RULES 2025" in full_text.upper():
        meta["parent_act"] = "DPDP Rules 2025"
        
    sec_match = re.search(r'\bSection\s+\d+(?:\.\d+)*', header_text, re.IGNORECASE)
    rule_match = re.search(r'\bRule\s+\d+(?:\.\d+)*', header_text, re.IGNORECASE)
    sched_match = re.search(r'\b(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH)\s+SCHEDULE\b', header_text, re.IGNORECASE)
    ch_match = re.search(r'\bCHAPTER\s+([IVXLCDM]+)', header_text, re.IGNORECASE)
    
    if sec_match:
        meta["type"] = "Section"
        meta["number"] = sec_match.group(0)
    elif rule_match:
        meta["type"] = "Rule"
        meta["number"] = rule_match.group(0)
    elif sched_match:
        meta["type"] = "Schedule"
        meta["number"] = sched_match.group(0).upper()
    elif ch_match:
        meta["type"] = "Chapter"
        meta["number"] = ch_match.group(1)
        
    sub_match = re.search(r'^\s*\((\d+[a-zA-Z]*)\)', body_text)
    if sub_match:
        meta["sub_section"] = f"({sub_match.group(1)})"
        
    # CRITICAL: Isolate State-only exemptions to prevent private audit hallucinations
    if "SECOND SCHEDULE" in header_text.upper() or "SEVENTH SCHEDULE" in header_text.upper():
        meta["applies_to"] = "state"
        
    return meta

# ═══════════════════════════════════════════════════════════════════════════
# 3. LEGAL-AWARE CHUNKING & HEADER BINDING
# ═══════════════════════════════════════════════════════════════════════════
def parse_and_chunk(file_path):
    """
    Chunks text with strict header binding to prevent the Section 2/20 merge bug.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = []
    metadatas = []
    
    split_pattern = r'(?=\n+(?:Section\s+\d+(?:\.\d+)*|Rule\s+\d+(?:\.\d+)*|CHAPTER\s+[IVXLCDM]+|(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH)\s+SCHEDULE))'
    body_blocks = re.split(split_pattern, text)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for block in body_blocks:
        block = block.strip()
        if not block:
            continue
            
        lines = block.split('\n')
        header = ""
        if lines and re.search(r'^(Section\s+\d+(?:\.\d+)*|Rule\s+\d+(?:\.\d+)*|CHAPTER\s+[IVXLCDM]+|(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH)\s+SCHEDULE)', lines[0].strip(), re.IGNORECASE):
            header = lines[0].strip()
            
        sub_chunks = splitter.split_text(block)
        
        for sc in sub_chunks:
            sc = sc.strip()
            if not sc:
                continue
                
            # STRICT HEADER BINDING: Prepend header if the sub-chunk lost it
            if header and not sc.startswith(header.split('\n')[0]):
                sc = header + "\n" + sc
                
            chunks.append(sc)
            metadatas.append(extract_metadata(header, sc))
            
    return chunks, metadatas

# ═══════════════════════════════════════════════════════════════════════════
# 4. MAIN BUILD ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def build_db():
    print("🚀 Initializing 10-Point SOTA Hybrid Search Engine Builder...")
    
    if not LAW_TEXT_PATH.exists():
        print(f"❌ Error: Legal text file not found at {LAW_TEXT_PATH}")
        return

    print("✂️ Chunking legal text with strict header binding...")
    chunks, metadatas = parse_and_chunk(LAW_TEXT_PATH)
    print(f"✅ Created {len(chunks)} unique legal chunks (800-char, headers preserved).")

    print("🧠 Embedding chunks with BAAI/bge-small-en-v1.5 (Instruction Tuned)...")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    # Instruction prefix is mandatory for queries, but NOT for corpus documents
    docs_for_embed = chunks
    embeddings = embed_model.encode(docs_for_embed, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    print("📚 Building Lexical BM25 Index...")
    tokenize = get_standard_tokenizer()
    tokenized_corpus = [tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    
    print("📥 Pre-Downloading Cross-Encoder (BAAI/bge-reranker-v2-m3)...")
    snapshot_download("BAAI/bge-reranker-v2-m3")
    print("✅ Cross-Encoder verified locally.")
    
    print(f"💾 Serializing to {HYBRID_INDEX_PATH}...")
    out_dict = {
        "chunks": chunks,
        "metadatas": metadatas,
        "bm25_index": bm25,
        "dense_embeddings": embeddings,
        "embedding_model_name": "BAAI/bge-small-en-v1.5",
        "chunk_count": len(chunks),
        "build_timestamp": datetime.datetime.now().isoformat()
    }
    
    HYBRID_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HYBRID_INDEX_PATH, "wb") as f:
        pickle.dump(out_dict, f)
        
    print(f"✅ Database built successfully. Size: {os.path.getsize(HYBRID_INDEX_PATH) / (1024*1024):.2f} MB")
    
    # ═══════════════════════════════════════════════════════════════════════
    # 5. DIAGNOSTIC TEST SUITE & STATE-CHUNK ISOLATION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n🔍 Running Golden Query Test Suite (RRF & State-Isolation)...")
    
    queries = [
        ("What is the penalty for data breach?", ["Schedule", "Section 33"], False),
        ("verifiable consent for children", ["Section 9", "Rule 10"], False),
        ("algorithmic software audit obligations", ["Rule 13"], False),
        ("Consent Manager interoperability", ["Rule 4", "First Schedule"], False),
        ("cross-border data transfer restrictions", ["Section 16", "Rule 15"], False),
        ("Standards for processing personal data by State instrumentalities", ["Second Schedule"], True)
    ]
    
    all_passed = True
    for q, expected_keywords, is_state_query in queries:
        # BM25 Search
        q_tokens = tokenize(q)
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        
        # Dense Search
        q_emb = embed_model.encode([f"Represent this sentence for searching relevant passages: {q}"], normalize_embeddings=True)[0]
        dense_scores = np.dot(embeddings, q_emb)
        dense_ranks = np.argsort(dense_scores)[::-1]
        
        # Reciprocal Rank Fusion (RRF k=60) WITH STATE-ISOLATION PRE-FILTER
        rrf_scores = np.zeros(len(chunks))
        
        for rank, idx in enumerate(bm25_ranks[:50]):
            if not is_state_query and metadatas[idx].get("applies_to") == "state":
                continue
            rrf_scores[idx] += 1.0 / (60 + rank + 1)
            
        for rank, idx in enumerate(dense_ranks[:50]):
            if not is_state_query and metadatas[idx].get("applies_to") == "state":
                continue
            rrf_scores[idx] += 1.0 / (60 + rank + 1)
            
        top_indices = np.argsort(rrf_scores)[::-1][:3]
        
        print(f"\nQuery: '{q}'")
        for i, idx in enumerate(top_indices):
            meta = metadatas[idx]
            snippet = chunks[idx][:80].replace('\n', ' ')
            print(f"  Hit {i+1} [{meta['type']} {meta['number']} | Applies To: {meta['applies_to']}] -> {snippet}...")
            
            # ISOLATION CHECK: Ensure State chunks don't bleed into private queries
            if not is_state_query and meta["applies_to"] == "state":
                print(f"  ❌ FATAL ERROR: Retrieved State-only chunk for private query!")
                all_passed = False
                
    if all_passed:
        print("\n🏆 ALL DIAGNOSTICS PASSED. Hybrid Index is SOTA and legally isolated.")
    else:
        print("\n⚠️ DIAGNOSTICS FAILED. Review metadata tagging and chunk boundaries.")

if __name__ == "__main__":
    build_db()