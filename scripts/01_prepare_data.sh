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
#   - Packages installed (`pip install -r requirements.txt`): unsloth, trl, transformers, vllm, datasets, torch, huggingface_hub
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
export LOG_DIR="${REPO_ROOT}/logs/prepare_data_${TIMESTAMP}"
export PYTHONUNBUFFERED=1
mkdir -p "${LOG_DIR}"

MASTER_LOG="${LOG_DIR}/00_prepare_data_master.log"
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
# [PHASE 0/4]: MODEL VERIFICATION & DOWNLOAD (LOCAL CHECK & HUGGINGFACE FALLBACK)
# ════════════════════════════════════════════════════════════════════════════════
mkdir -p "${REPO_ROOT}/ml/models"
echo -e "\n▶️  [PHASE 0/4]: Verifying Teacher (`Qwen2-72B`) & Student (`Qwen 3.5 9B`) Models in ml/models/..."

# 1. Check Student Model: Qwen/Qwen3.5-9B (or Qwen2.5-9B-Instruct if already downloaded)
if [ -d "${REPO_ROOT}/ml/models/Qwen3.5-9B" ] && [ -n "$(ls -A "${REPO_ROOT}/ml/models/Qwen3.5-9B" 2>/dev/null)" ]; then
    echo "   ✅ Student model (`Qwen3.5-9B`) found locally at ml/models/Qwen3.5-9B. Skipping download."
elif [ -d "${REPO_ROOT}/ml/models/Qwen2.5-9B-Instruct" ] && [ -n "$(ls -A "${REPO_ROOT}/ml/models/Qwen2.5-9B-Instruct" 2>/dev/null)" ]; then
    echo "   ✅ Student model found locally at ml/models/Qwen2.5-9B-Instruct. Skipping download."
else
    echo "   📥 Student model not found locally in ml/models/. Downloading Qwen/Qwen3.5-9B via huggingface-cli..."
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download Qwen/Qwen3.5-9B --local-dir "${REPO_ROOT}/ml/models/Qwen3.5-9B"
    else
        "${PYTHON_CMD}" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3.5-9B', local_dir='${REPO_ROOT}/ml/models/Qwen3.5-9B')"
    fi
    echo "   ✅ Downloaded Qwen/Qwen3.5-9B successfully."
fi

# 2. Check Teacher Model: Qwen2-72B-Instruct-FP8 (or Qwen2.5-72B-Instruct-FP8)
if [ -d "${REPO_ROOT}/ml/models/Qwen2-72B-Instruct-FP8" ] && [ -n "$(ls -A "${REPO_ROOT}/ml/models/Qwen2-72B-Instruct-FP8" 2>/dev/null)" ]; then
    echo "   ✅ Teacher model (`Qwen2-72B-Instruct-FP8`) found locally at ml/models/Qwen2-72B-Instruct-FP8. Skipping download."
elif [ -d "${REPO_ROOT}/ml/models/Qwen2.5-72B-Instruct-FP8" ] && [ -n "$(ls -A "${REPO_ROOT}/ml/models/Qwen2.5-72B-Instruct-FP8" 2>/dev/null)" ]; then
    echo "   ✅ Teacher model found locally at ml/models/Qwen2.5-72B-Instruct-FP8. Skipping download."
else
    echo "   📥 Teacher model not found locally in ml/models/. Downloading Qwen/Qwen2-72B-Instruct-FP8 via huggingface-cli..."
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download Qwen/Qwen2-72B-Instruct-FP8 --local-dir "${REPO_ROOT}/ml/models/Qwen2-72B-Instruct-FP8"
    else
        "${PYTHON_CMD}" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2-72B-Instruct-FP8', local_dir='${REPO_ROOT}/ml/models/Qwen2-72B-Instruct-FP8')"
    fi
    echo "   ✅ Downloaded teacher model successfully."
fi

cd "${REPO_ROOT}/ml/data-forge"

# ════════════════════════════════════════════════════════════════════════════════
# [PHASE 1/4]: SYNTHETIC DATA FORGE
# ════════════════════════════════════════════════════════════════════════════════
if [ "${SKIP_GAN}" = true ]; then
    echo -e "\n⏭️  [PHASE 1/4]: Skipping Synthetic GAN Forge (--skip-gan enabled)."
else
    echo -e "\n▶️  [PHASE 1/4]: Forging Synthetic Data (GAN Forge via vLLM)..."
    "${PYTHON_CMD}" -u gan_forge.py 2>&1 | tee "${LOG_DIR}/01_gan_forge.log"
fi

# ════════════════════════════════════════════════════════════════════════════════
# [PHASE 2/4]: DETERMINISTIC RUST DECISION TREE
# ════════════════════════════════════════════════════════════════════════════════
echo -e "\n▶️  [PHASE 2/4]: Building Deterministic Rust Decision Tree..."
"${PYTHON_CMD}" -u build_dpdp_tree.py 2>&1 | tee "${LOG_DIR}/02_build_tree.log"

# ════════════════════════════════════════════════════════════════════════════════
# [PHASE 3/4]: UNSLOTH DATASET ALIGNMENT & FORMATTING
# ════════════════════════════════════════════════════════════════════════════════
echo -e "\n▶️  [PHASE 3/4]: Aligning & Formatting Data Schemas for Unsloth..."
"${PYTHON_CMD}" -u prepare_unsloth_data.py 2>&1 | tee "${LOG_DIR}/03_prepare_data.log"

echo -e "\n════════════════════════════════════════════════════════════════════════════════"
echo "✅ STAGE 1 COMPLETE: Models verified and Unsloth SFT & DPO training datasets synthesized."
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📁 Execution logs saved to: ${LOG_DIR}"
echo "👉 Next step: Run 'bash scripts/02_train_models.sh' to fine-tune both DPDP SLMs on DGX Spark."
