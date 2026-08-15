#!/usr/bin/env bash
# ==============================================================================
# 01_prepare_data.sh – Stage 1: Synthetic Data Forge, Model Verification & Unsloth Preparation
# ==============================================================================
# ⚠️ SPECIALIZED PIPELINE NOTICE:
# This script is exclusively engineered for preparing data and downloading checkpoints for the Indian Digital Personal
# Data Protection (DPDP) Act 2023 & Rules 2025 specialized legal models on NVIDIA DGX Spark infrastructure (128 GB VRAM).
# It checks/downloads our Teacher Model (`Qwen2-72B-Instruct-FP8`) and Student Model (`Qwen/Qwen3.5-9B`), synthesizes
# statutory training data, and formats Unsloth SFT + DPO datasets. It is specialized for our DPDP law pipeline only.
#
# Prerequisites for DGX Spark Execution:
#   - NVIDIA DGX / GPU environment (128 GB VRAM unified or split)
#   - Python 3.10+ / 3.12 with CUDA 12.x support
#   - Packages installed (`pip install -r requirements.txt`): unsloth, trl, transformers, vllm, datasets, torch, huggingface_hub, chromadb, langchain-text-splitters, rank_bm25
#
# Usage:
#   bash scripts/01_prepare_data.sh [--skip-gan]
#
# Options:
#   --skip-gan    Skip running gan_forge.py (useful if raw GAN policy files exist)
#   -h, --help    Show help menu and exit
# ==============================================================================

set -euo pipefail

# Locate repository root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve Python binary
PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

SKIP_GAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-gan)
            SKIP_GAN=true
            shift
            ;;
        -h|--help)
            echo "Usage: bash scripts/01_prepare_data.sh [options]"
            echo "Options:"
            echo "  --skip-gan    Skip running gan_forge.py and proceed directly to tree building and dataset formatting"
            echo "  -h, --help    Show this help message and exit"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for available options."
            exit 1
            ;;
    esac
done

# Setup logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export LOG_DIR="${REPO_ROOT}/logs"
export PYTHONUNBUFFERED=1
mkdir -p "${LOG_DIR}"

SCRIPT_NAME=$(basename "$0" .sh)
MASTER_LOG="${LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log"
exec > >(tee -i "${MASTER_LOG}")
exec 2>&1

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 INITIATING STAGE 1: DGX SPARK MODEL VERIFICATION & DATA PREPARATION"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "   • Repository Root: ${REPO_ROOT}"
echo "   • Python Binary:   $(command -v "${PYTHON_CMD}")"
echo "   • Logs Directory:  ${LOG_DIR}"
echo "   • Skip GAN Forge:  ${SKIP_GAN}"
echo "   • Target Domain:   Digital Personal Data Protection (DPDP) Act 2023 & Rules 2025"
echo "════════════════════════════════════════════════════════════════════════════════"

# Terminate lingering vLLM API server processes from aborted jobs if any
echo -e "\n🧹 Cleaning up any orphaned vLLM API endpoints before starting..."
pkill -f vllm.entrypoints.openai.api_server 2>/dev/null || true

# ════════════════════════════════════════════════════════════════════════════════
# [PHASE 0.1]: PURGE CACHES (PRESERVING GENERATED DATA)
# ════════════════════════════════════════════════════════════════════════════════
echo -e "\n🧹 [PHASE 0.1]: Purging Caches..."

# 1. Purge Python Bytecode Cache (Kills the "Ghost in the Machine")
find "${REPO_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "   ✅ Purged all __pycache__ directories."

# 2. Purge Vector DB Cache (Forces fresh embedding generation if rerun)
rm -f "${REPO_ROOT}/ml/data-forge/dpdp_hybrid_index.pkl" 2>/dev/null || true
echo "   ✅ Purged Hybrid RAG Index cache."

# 3. Purge Aggregated JSONL data (so `prepare_unsloth_data.py` rebuilds them fresh from existing JSONs)
rm -f "${REPO_ROOT}/ml/slm-training/data"/*.jsonl 2>/dev/null || true
echo "   ✅ Purged aggregated JSONL training files (Will be rebuilt in Phase 3)."

# 4. [DISABLED] Purge Chatbot generations
# rm -f "${REPO_ROOT}/ml/data-forge/training-pairs/chatbot-sft"/*.json 2>/dev/null || true
# rm -f "${REPO_ROOT}/ml/data-forge/training-pairs/chatbot-dpo"/*.json 2>/dev/null || true
echo "   🛡️ [DISABLED] Chatbot QA purge skipped. Preserving existing data."

# 5. [DISABLED] SURGICAL PURGE: SFT & DPO generations
# if [ -d "${REPO_ROOT}/ml/data-forge/training-pairs/sft" ]; then
#     find "${REPO_ROOT}/ml/data-forge/training-pairs/sft" -type f -name "*.json" \
#         ! -name "sft_000_*.json" -delete
# fi
echo "   🛡️ [DISABLED] Standard SFT generation purge skipped. Preserving existing data."

# if [ -d "${REPO_ROOT}/ml/data-forge/training-pairs/dpo" ]; then
#     find "${REPO_ROOT}/ml/data-forge/training-pairs/dpo" -type f -name "*.json" \
#         ! -name "dpo_000_*.json" -delete
# fi
echo "   🛡️ [DISABLED] Standard DPO generation purge skipped. Preserving existing data."

# ════════════════════════════════════════════════════════════════════════════════
# [PHASE 0.5/4]: MODEL VERIFICATION & DOWNLOAD (LOCAL CHECK & HUGGINGFACE FALLBACK)
# ════════════════════════════════════════════════════════════════════════════════
mkdir -p "${REPO_ROOT}/ml/models"
echo -e "\n▶️  [PHASE 0.5/4]: Verifying Teacher ('Qwen2-72B') & Student ('Qwen 3.5 9B') Models in ml/models/..."

# 1. Check Student Model: Qwen/Qwen3.5-9B
if [ -d "${REPO_ROOT}/ml/models/Qwen3.5-9B" ] && [ -n "$(ls -A "${REPO_ROOT}/ml/models/Qwen3.5-9B" 2>/dev/null)" ]; then
    echo "   ✅ Student model ('Qwen3.5-9B') found locally at ml/models/Qwen3.5-9B. Skipping download."
elif [ -d "${REPO_ROOT}/ml/models/Qwen2.5-9B-Instruct" ] && [ -n "$(ls -A "${REPO_ROOT}/ml/models/Qwen2.5-9B-Instruct" 2>/dev/null)" ]; then
    echo "   ✅ Student model found locally at ml/models/Qwen2.5-9B-Instruct. Skipping download."
else
    echo "   📥 Student model not found locally in ml/models/. Downloading Qwen/Qwen3.5-9B via hf..."
    if command -v hf &> /dev/null; then
        hf download Qwen/Qwen3.5-9B --local-dir "${REPO_ROOT}/ml/models/Qwen3.5-9B"
    else
        "${PYTHON_CMD}" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3.5-9B', local_dir='${REPO_ROOT}/ml/models/Qwen3.5-9B')"
    fi
    echo "   ✅ Downloaded Qwen/Qwen3.5-9B successfully."
fi

# 2. Check Teacher Model: Qwen2-72B-Instruct-FP8
if [ -d "${REPO_ROOT}/ml/models/Qwen2-72B-Instruct-FP8" ] && [ -n "$(ls -A "${REPO_ROOT}/ml/models/Qwen2-72B-Instruct-FP8" 2>/dev/null)" ]; then
    echo "   ✅ Teacher model ('Qwen2-72B-Instruct-FP8') found locally at ml/models/Qwen2-72B-Instruct-FP8. Skipping download."
elif [ -d "${REPO_ROOT}/ml/models/Qwen2.5-72B-Instruct-FP8" ] && [ -n "$(ls -A "${REPO_ROOT}/ml/models/Qwen2.5-72B-Instruct-FP8" 2>/dev/null)" ]; then
    echo "   ✅ Teacher model found locally at ml/models/Qwen2.5-72B-Instruct-FP8. Skipping download."
else
    echo "   📥 Teacher model not found locally in ml/models/. Downloading Qwen/Qwen2-72B-Instruct-FP8 via hf..."
    if command -v hf &> /dev/null; then
        hf download Qwen/Qwen2-72B-Instruct-FP8 --local-dir "${REPO_ROOT}/ml/models/Qwen2-72B-Instruct-FP8"
    else
        "${PYTHON_CMD}" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2-72B-Instruct-FP8', local_dir='${REPO_ROOT}/ml/models/Qwen2-72B-Instruct-FP8')"
    fi
    echo "   ✅ Downloaded teacher model successfully."
fi

# 3. Check RAG Models: BAAI/bge-small-en-v1.5 and BAAI/bge-reranker-v2-m3
if [ -d "${REPO_ROOT}/ml/models/bge-small-en-v1.5" ] && [ -n "$(ls -A "${REPO_ROOT}/ml/models/bge-small-en-v1.5" 2>/dev/null)" ]; then
    echo "   ✅ RAG Embedding model ('bge-small-en-v1.5') found locally in ml/models/. Skipping download."
else
    echo "   📥 RAG Embedding model not found locally in ml/models/. Downloading BAAI/bge-small-en-v1.5..."
    if command -v hf &> /dev/null; then
        hf download BAAI/bge-small-en-v1.5 --local-dir "${REPO_ROOT}/ml/models/bge-small-en-v1.5"
    else
        "${PYTHON_CMD}" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-small-en-v1.5', local_dir='${REPO_ROOT}/ml/models/bge-small-en-v1.5')"
    fi
    echo "   ✅ Downloaded RAG embedding model successfully."
fi

if [ -d "${REPO_ROOT}/ml/models/bge-reranker-v2-m3" ] && [ -n "$(ls -A "${REPO_ROOT}/ml/models/bge-reranker-v2-m3" 2>/dev/null)" ]; then
    echo "   ✅ RAG Reranker model ('bge-reranker-v2-m3') found locally in ml/models/. Skipping download."
else
    echo "   📥 RAG Reranker model not found locally in ml/models/. Downloading BAAI/bge-reranker-v2-m3..."
    if command -v hf &> /dev/null; then
        hf download BAAI/bge-reranker-v2-m3 --local-dir "${REPO_ROOT}/ml/models/bge-reranker-v2-m3"
    else
        "${PYTHON_CMD}" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-reranker-v2-m3', local_dir='${REPO_ROOT}/ml/models/bge-reranker-v2-m3')"
    fi
    echo "   ✅ Downloaded RAG reranker model successfully."
fi

# ════════════════════════════════════════════════════════════════════════════════
# [PHASE 0.7/4]: RAG DEPENDENCY RESOLUTION
# ════════════════════════════════════════════════════════════════════════════════
echo -e "\n▶️  [PHASE 0.7/4]: Verifying Data Forge & Vector DB Dependencies..."
"${PYTHON_CMD}" -m pip install -q chromadb langchain-text-splitters rank_bm25 sentence-transformers langchain
echo "   ✅ Dependencies synchronized."

cd "${REPO_ROOT}/ml/data-forge"

# ════════════════════════════════════════════════════════════════════════════════
# [PHASE 1/4]: SYNTHETIC DATA FORGE
# ════════════════════════════════════════════════════════════════════════════════
if [ "${SKIP_GAN}" = true ]; then
    echo -e "\n⏭️  [PHASE 1/4]: Skipping Synthetic GAN Forge (--skip-gan enabled)."
else
    echo -e "\n▶️  [PHASE 1/4(a)]: Building Vector Database (ChromaDB)..."
    "${PYTHON_CMD}" -u build_vector_db.py

    echo -e "\n▶️  [PHASE 1/4(b)]: Forging Synthetic Data (GAN Forge via vLLM)..."
    "${PYTHON_CMD}" -u gan_forge.py
fi

# ════════════════════════════════════════════════════════════════════════════════
# [PHASE 2/4]: DETERMINISTIC RUST DECISION TREE
# ════════════════════════════════════════════════════════════════════════════════
echo -e "\n▶️  [PHASE 2/4]: Building Deterministic Rust Decision Tree..."
"${PYTHON_CMD}" -u build_dpdp_tree.py

# ════════════════════════════════════════════════════════════════════════════════
# [PHASE 3/4]: UNSLOTH DATASET ALIGNMENT & FORMATTING
# ════════════════════════════════════════════════════════════════════════════════
echo -e "\n▶️  [PHASE 3/4]: Aligning & Formatting Data Schemas for Unsloth..."
"${PYTHON_CMD}" -u prepare_unsloth_data.py

echo -e "\n════════════════════════════════════════════════════════════════════════════════"
echo "✅ STAGE 1 COMPLETE: Models verified and Unsloth SFT & DPO training datasets synthesized."
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📁 Execution logs saved to: ${LOG_DIR}"
echo "👉 Next step: Run 'bash scripts/02_train_models.sh' to fine-tune both DPDP SLMs on DGX Spark."