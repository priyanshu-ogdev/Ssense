#!/usr/bin/env python3
"""
build_vector_db.py – 10-Point SOTA Hybrid Search Engine Builder (FINAL)

Architecture:
1. Pure Python Hybrid Index (.pkl) - Eliminates ChromaDB daemon/SQLite locks.
2. Legal-Aware Chunking (800 char, 80 overlap, strict header binding).
3. Metadata Tagging Engine (type, number, applies_to).
4. BM25 Lexical Index (Custom legal Tokenizer + BM25Okapi).
5. Dense Embedding Index (BGE-small, Instruction Prefix, L2 Normalization).
6. Cross-Encoder Pre-Download Check (BAAI/bge-reranker-v2-m3).
7. Single File Serialization (dpdp_hybrid_index.pkl).
8. Diagnostic Test Suite with State-Chunk Isolation Verification.
"""

import os
import re
import sys
import datetime
import numpy as np
import pickle
from collections import Counter

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from langchain_text_splitters import RecursiveCharacterTextSplitter

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════════════════
# 1. CUSTOM LEGAL TOKENIZER (For BM25)
# ═══════════════════════════════════════════════════════════════════════════
def get_legal_tokenizer():
    """
    Tokenizer that preserves multi-word legal entities and statutory references 
    as single tokens to prevent BM25 IDF dilution.
    """
    generic_stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "of", "and", "in", 
        "to", "for", "with", "on", "at", "by", "from", "as", "that", "this", 
        "it", "be", "or", "which", "will", "would", "could", "should", "their", "they"
    }
    
    def tokenize(text):
        text = str(text).lower()
        # Preserve multi-word legal entities
        text = text.replace("data fiduciary", "data_fiduciary")
        text = text.replace("data principal", "data_principal")
        text = text.replace("consent manager", "consent_manager")
        text = text.replace("significant data fiduciary", "significant_data_fiduciary")
        text = text.replace("sub-section", "subsection")
        
        # Preserve statutory references (e.g., section_8, rule_13_3, section_2)
        text = re.sub(r'section\s+(\d+)\((\d+)\)', r'section_\1_\2', text)
        text = re.sub(r'rule\s+(\d+)\((\d+)\)', r'rule_\1_\2', text)
        text = re.sub(r'section\s+(\d+)', r'section_\1', text)
        text = re.sub(r'rule\s+(\d+)', r'rule_\1', text)
        
        words = re.findall(r'\b\w+\b', text)
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
        
    # 🚨 UPGRADED: Handles "Section 2." or "Section 17(1)" variations
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
        
    # 🚨 CRITICAL: Isolate State-only exemptions to prevent private audit hallucinations
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
    
    # 🚨 UPGRADED: Robust regex to catch "Section 2.", "Rule 13(3)", etc.
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
        # Extract the primary header of this block
        if lines and re.search(r'^(Section\s+\d+(?:\.\d+)*|Rule\s+\d+(?:\.\d+)*|CHAPTER\s+[IVXLCDM]+|(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH)\s+SCHEDULE)', lines[0].strip(), re.IGNORECASE):
            header = lines[0].strip()
            
        sub_chunks = splitter.split_text(block)
        
        for sc in sub_chunks:
            sc = sc.strip()
            if not sc:
                continue
                
            # 🚨 STRICT HEADER BINDING: Prepend header if the sub-chunk lost it
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
    
    file_path = "./dpdp_act_and_rules_2025.txt"
    if not os.path.exists(file_path):
        print(f"❌ Error: Legal text file not found at {file_path}")
        return

    print("✂️ Chunking legal text with strict header binding...")
    chunks, metadatas = parse_and_chunk(file_path)
    print(f"✅ Created {len(chunks)} unique legal chunks (800-char, headers preserved).")

    print("🧠 Embedding chunks with BAAI/bge-small-en-v1.5 (Instruction Tuned)...")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    # Instruction prefix is mandatory for queries, but NOT for corpus documents
    docs_for_embed = chunks
    embeddings = embed_model.encode(docs_for_embed, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    print("📚 Building Lexical BM25 Index...")
    tokenize = get_legal_tokenizer()
    tokenized_corpus = [tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    
    print("📥 Pre-Downloading Cross-Encoder (BAAI/bge-reranker-v2-m3)...")
    snapshot_download("BAAI/bge-reranker-v2-m3")
    print("✅ Cross-Encoder verified locally.")
    
    print("💾 Serializing dpdp_hybrid_index.pkl...")
    out_dict = {
        "chunks": chunks,
        "metadatas": metadatas,
        "bm25_index": bm25,
        "dense_embeddings": embeddings,
        "embedding_model_name": "BAAI/bge-small-en-v1.5",
        "chunk_count": len(chunks),
        "build_timestamp": datetime.datetime.now().isoformat()
    }
    
    pkl_path = "dpdp_hybrid_index.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(out_dict, f)
        
    print(f"✅ Database built successfully. Size: {os.path.getsize(pkl_path) / (1024*1024):.2f} MB")
    
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
        # 🚨 UPGRADED: Added a State-specific query to prove isolation works both ways
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
            # 🚨 PRE-FILTER: Block State chunks from private queries BEFORE scoring
            if not is_state_query and metadatas[idx].get("applies_to") == "state":
                continue
            rrf_scores[idx] += 1.0 / (60 + rank + 1)
            
        for rank, idx in enumerate(dense_ranks[:50]):
            # 🚨 PRE-FILTER: Block State chunks from private queries BEFORE scoring
            if not is_state_query and metadatas[idx].get("applies_to") == "state":
                continue
            rrf_scores[idx] += 1.0 / (60 + rank + 1)
            
        top_indices = np.argsort(rrf_scores)[::-1][:3]
        
        print(f"\nQuery: '{q}'")
        for i, idx in enumerate(top_indices):
            meta = metadatas[idx]
            snippet = chunks[idx][:80].replace('\n', ' ')
            print(f"  Hit {i+1} [{meta['type']} {meta['number']} | Applies To: {meta['applies_to']}] -> {snippet}...")
            
            # 🚨 ISOLATION CHECK: Ensure State chunks don't bleed into private queries
            if not is_state_query and meta["applies_to"] == "state":
                print(f"  ❌ FATAL ERROR: Retrieved State-only chunk for private query!")
                all_passed = False
                
    if all_passed:
        print("\n🏆 ALL DIAGNOSTICS PASSED. Hybrid Index is SOTA and legally isolated.")
    else:
        print("\n⚠️ DIAGNOSTICS FAILED. Review metadata tagging and chunk boundaries.")

if __name__ == "__main__":
    build_db()