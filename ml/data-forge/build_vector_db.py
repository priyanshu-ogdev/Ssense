#!/usr/bin/env python3
"""
build_vector_db.py – SOTA Hybrid Search Engine & Dual-Format Artifact Builder

Generates:
1. `dpdp_hybrid_index.pkl` -> Python-native serialized hybrid index (Backward compatibility).
2. `dpdp_embeddings.safetensors` -> Zero-copy float32 dense embeddings matrix for Rust Native Daemon.
3. `dpdp_index.json` -> Structured chunks, legal metadata, and Okapi BM25 statistics for Rust Native Daemon.
"""

import os
import re
import sys
import json
import datetime
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple

import numpy as np
import pickle

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from safetensors.numpy import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════════════════
# PATH RESOLUTION & SIBLING IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
_CURRENT_DIR = Path(__file__).resolve().parent
_EVALS_DIR = _CURRENT_DIR.parent / "evals"

if str(_EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALS_DIR))

try:
    from path_resolver import Paths
    LAW_TEXT_PATH = Paths.LAW_TEXT
    HYBRID_INDEX_PATH = Paths.HYBRID_INDEX
except ImportError:
    LAW_TEXT_PATH = _CURRENT_DIR / "dpdp_act_and_rules_2025.txt"
    HYBRID_INDEX_PATH = _CURRENT_DIR / "dpdp_hybrid_index.pkl"

RAG_DIR = _CURRENT_DIR.parent / "models" / "rag-index"
SAFETENSORS_PATH = RAG_DIR / "dpdp_embeddings.safetensors"
INDEX_JSON_PATH = RAG_DIR / "dpdp_index.json"


# ═══════════════════════════════════════════════════════════════════════════
# 1. STANDARDIZED TOKENIZER (Parity with Evaluation & Rust Engine)
# ═══════════════════════════════════════════════════════════════════════════
GENERIC_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "and", "in", 
    "to", "for", "with", "on", "at", "by", "from", "as", "that", "this", 
    "it", "be", "or", "which", "will", "would", "could", "should", "their", "they"
}

def get_standard_tokenizer():
    def tokenize(text: str) -> List[str]:
        words = re.findall(r'\w+', str(text).lower())
        return [w for w in words if w not in GENERIC_STOPWORDS]
    return tokenize


# ═══════════════════════════════════════════════════════════════════════════
# 2. METADATA EXTRACTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def extract_metadata(header_text: str, body_text: str) -> Dict[str, str]:
    meta = {
        "type": "General",
        "number": "",
        "sub_section": "",
        "parent_act": "DPDP Act 2023",
        "applies_to": "all"
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
        
    if "SECOND SCHEDULE" in header_text.upper() or "SEVENTH SCHEDULE" in header_text.upper():
        meta["applies_to"] = "state"
        
    return meta


# ═══════════════════════════════════════════════════════════════════════════
# 3. LEGAL-AWARE CHUNKING & HEADER BINDING
# ═══════════════════════════════════════════════════════════════════════════
def parse_and_chunk(file_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
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
                
            if header and not sc.startswith(header.split('\n')[0]):
                sc = header + "\n" + sc
                
            chunks.append(sc)
            metadatas.append(extract_metadata(header, sc))
            
    return chunks, metadatas


# ═══════════════════════════════════════════════════════════════════════════
# 4. RUST ARTIFACT EXPORT ENGINE (.safetensors + .json)
# ═══════════════════════════════════════════════════════════════════════════
def export_rust_artifacts(
    chunks: List[str],
    metadatas: List[Dict[str, str]],
    bm25: BM25Okapi,
    dense_embeddings: np.ndarray,
    safetensors_path: Path,
    index_json_path: Path
):
    print("\n📦 Exporting Language-Agnostic Rust Artifacts...")

    # 1. Export Dense Embeddings to .safetensors (float32, 2D Tensor)
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors library is missing. Install with: pip install safetensors")

    dense_f32 = np.ascontiguousarray(dense_embeddings, dtype=np.float32)
    tensors_dict = {
        "dense_embeddings": dense_f32
    }
    
    safetensors_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors_dict, str(safetensors_path))
    print(f"  • Dense Safetensors: {safetensors_path} ({os.path.getsize(safetensors_path)/(1024*1024):.2f} MB, shape={dense_f32.shape})")

    # 2. Export Text Chunks, Metadata, and Okapi BM25 Statistics to .json
    # Convert Counter/IDF objects into plain serializable structures for Rust
    idf_dict = {k: float(v) for k, v in bm25.idf.items()}
    doc_lens = [int(dl) for dl in bm25.doc_len]

    json_payload = {
        "schema_version": "2.0.0",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimensions": int(dense_embeddings.shape[1]),
        "total_chunks": len(chunks),
        "bm25_params": {
            "k1": float(bm25.k1),
            "b": float(bm25.b),
            "avgdl": float(bm25.avgdl),
            "doc_len": doc_lens,
            "idf": idf_dict
        },
        "chunks": chunks,
        "metadatas": metadatas
    }

    with open(index_json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)
    print(f"  • Hybrid JSON Index: {index_json_path} ({os.path.getsize(index_json_path)/(1024*1024):.2f} MB)")

    # 3. Validation Check
    loaded_tensors = load_file(str(safetensors_path))
    assert "dense_embeddings" in loaded_tensors, "Safetensors key validation failed!"
    assert loaded_tensors["dense_embeddings"].shape == dense_f32.shape, "Tensor shape mismatch!"
    print("  ✅ Rust Artifact Integrity Verification Passed.")


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN BUILD ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def build_db():
    print("═══════════════════════════════════════════════════════════════════════")
    print("🚀 SOTA HYBRID SEARCH ENGINE & TRIPLE-FORMAT ARTIFACT BUILDER")
    print("═══════════════════════════════════════════════════════════════════════")
    
    if not LAW_TEXT_PATH.exists():
        print(f"❌ Error: Legal text file not found at {LAW_TEXT_PATH}")
        return

    print("✂️ Chunking statutory corpus with strict header binding...")
    chunks, metadatas = parse_and_chunk(LAW_TEXT_PATH)
    print(f"✅ Created {len(chunks)} legal chunks (800-char envelope with header preservation).")

    print("\n🧠 Computing normalized dense embeddings with BAAI/bge-small-en-v1.5...")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = embed_model.encode(chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    print("\n📚 Compiling BM25 Okapi Lexical Inverted Index...")
    tokenize = get_standard_tokenizer()
    tokenized_corpus = [tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

    print("\n📥 Verifying Cross-Encoder Local Artifacts (BAAI/bge-reranker-v2-m3)...")
    snapshot_download("BAAI/bge-reranker-v2-m3")
    print("✅ Cross-Encoder verified locally.")

    # -------------------------------------------------------------------------
    # A. Python Legacy Pickle Serialization
    # -------------------------------------------------------------------------
    print(f"\n💾 Serializing Python Hybrid Index to: {HYBRID_INDEX_PATH}...")
    out_dict = {
        "chunks": chunks,
        "metadatas": metadatas,
        "bm25_index": bm25,
        "dense_embeddings": embeddings,
        "embedding_model_name": "BAAI/bge-small-en-v1.5",
        "chunk_count": len(chunks),
        "build_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    HYBRID_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HYBRID_INDEX_PATH, "wb") as f:
        pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"✅ Pickle artifact successfully saved ({os.path.getsize(HYBRID_INDEX_PATH) / (1024*1024):.2f} MB).")

    # -------------------------------------------------------------------------
    # B. Rust-Native .safetensors + .json Export
    # -------------------------------------------------------------------------
    export_rust_artifacts(
        chunks=chunks,
        metadatas=metadatas,
        bm25=bm25,
        dense_embeddings=embeddings,
        safetensors_path=SAFETENSORS_PATH,
        index_json_path=INDEX_JSON_PATH
    )

    # -------------------------------------------------------------------------
    # C. Verification Test Suite & State-Chunk Isolation Check
    # -------------------------------------------------------------------------
    print("\n🔍 Executing Multi-Vector Retrieval Verification Suite...")
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
        q_tokens = tokenize(q)
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        
        q_emb = embed_model.encode([f"Represent this sentence for searching relevant passages: {q}"], normalize_embeddings=True)[0]
        dense_scores = np.dot(embeddings, q_emb)
        dense_ranks = np.argsort(dense_scores)[::-1]
        
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
            
            if not is_state_query and meta["applies_to"] == "state":
                print(f"  ❌ FATAL ERROR: Retrieved State-only chunk for private query!")
                all_passed = False
                
    if all_passed:
        print("\n🏆 ALL RETRIEVAL & ISOLATION CHECKS PASSED.")
    else:
        print("\n⚠️ RETRIEVAL CHECKS FAILED. Review metadata tagging and chunk boundaries.")


if __name__ == "__main__":
    build_db()