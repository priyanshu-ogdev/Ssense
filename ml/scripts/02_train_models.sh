#!/usr/bin/env bash
# ==============================================================================
# 02_train_models.sh – Stage 2: VRAM Airlock & Unsloth SLM Fine-Tuning (DGX Spark)
# ==============================================================================
# ⚠️ SPECIALIZED PIPELINE NOTICE:
# This script is exclusively engineered for training the specialized Indian Digital Personal Data Protection (DPDP) Act
# 2023 & Rules 2025 legal models (Forensic Auditor & Conversational Chatbot) on NVIDIA DGX Spark infrastructure (128 GB VRAM).
# It executes our 32-bit FP32 `adamw_torch` + `rsLoRA` + `SimPO` pipeline on `Qwen2.5-7B-Instruct`. Not a generic wrapper.
#
# Prerequisites for DGX Spark Execution:
#   - NVIDIA DGX / GPU environment (`multiprocessing.set_start_method("spawn")` enabled)
#   - Python 3.10+ / 3.12 with CUDA 12.x/13.x support
#   - Packages installed (`pip install -r ml/requirements.txt`): unsloth, trl, transformers, vllm, datasets, torch
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ML_DIR}/.." && pwd)"
TRAINING_DIR="${ML_DIR}/slm-training"

PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

MODEL_TARGET="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TARGET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash ml/scripts/02_train_models.sh [options]"
            echo "Options:"
            echo "  --model <audit|chatbot|all>   Specify which model(s) to train (default: all)"
            echo "  -h, --help                    Show this help message and exit"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for available options."
            exit 1
            ;;
    esac
done

if [[ "${MODEL_TARGET}" != "audit" && "${MODEL_TARGET}" != "chatbot" && "${MODEL_TARGET}" != "all" ]]; then
    echo "❌ Invalid --model target: '${MODEL_TARGET}'. Must be 'audit', 'chatbot', or 'all'."
    exit 1
fi

if [[ ! -d "${TRAINING_DIR}" ]]; then
    echo "❌ FATAL: Training directory not found at ${TRAINING_DIR}"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export LOG_DIR="${REPO_ROOT}/logs"
export PYTHONUNBUFFERED=1
mkdir -p "${LOG_DIR}"

SCRIPT_NAME=$(basename "$0" .sh)
MASTER_LOG="${LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log"
exec > >(tee -i "${MASTER_LOG}")
exec 2>&1

# ==============================================================================
# FUNCTION: VRAM AIRLOCK SANITIZATION
# ==============================================================================
vram_airlock() {
    echo -e "\n════════════════════════════════════════════════════════════════════════════════"
    echo "🧹 VRAM AIRLOCK: PURGING RESIDUAL GPU & POSIX SHARED MEMORY"
    echo "════════════════════════════════════════════════════════════════════════════════"

    echo "   [1/5] Terminating Ray background daemons..."
    ray stop -f 2>/dev/null || true

    echo "   [2/5] Hunting zombie vLLM and orphaned inference processes..."
    pkill -9 -f vllm 2>/dev/null || true
    pkill -9 -f vllm.entrypoints.openai.api_server 2>/dev/null || true
    pkill -9 -f ray:: 2>/dev/null || true

    echo "   [3/5] Neutralizing orphaned PyTorch multiprocessing spawned workers..."
    pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true

    echo "   [4/5] Flushing POSIX shared memory allocations..."
    rm -rf /dev/shm/* 2>/dev/null || true

    echo "   [5/5] Checking OS file cache drop permissions..."
    if command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
        sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null || true
    else
        echo "         (Skipping OS drop_caches: non-root/non-sudo context)"
    fi

    echo "⏳ Waiting 5 seconds for Nvidia GPU memory state stabilization..."
    sleep 5
    echo "✅ VRAM Airlock complete. Environment strictly isolated."
}

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 INITIATING STAGE 2: VRAM AIRLOCK & SLM TRAINING PIPELINE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "   • Repository Root: ${REPO_ROOT}"
echo "   • ML Root:         ${ML_DIR}"
echo "   • Python Binary:   $(command -v "${PYTHON_CMD}")"
echo "   • Execution Dir:   ${TRAINING_DIR}"
echo "   • Logs Directory:  ${LOG_DIR}"
echo "   • Training Target: ${MODEL_TARGET}"
echo "════════════════════════════════════════════════════════════════════════════════"

# Execute Initial Pre-Training Airlock
vram_airlock

# ------------------------------------------------------------------------------
# SLM FINE-TUNING EXECUTION
# ------------------------------------------------------------------------------
cd "${TRAINING_DIR}"

if [[ "${MODEL_TARGET}" == "audit" || "${MODEL_TARGET}" == "all" ]]; then
    echo -e "\n▶️  [TRAINING 1/2]: Executing Forensic Auditor Fine-Tuning (train_audit.py)..."
    "${PYTHON_CMD}" -u train_audit.py
fi

if [[ "${MODEL_TARGET}" == "all" ]]; then
    # 🚨 CRITICAL: Re-engage the VRAM Airlock between models to guarantee the 
    # Chatbot starts with 100% clean memory and zero PyTorch fragmentation.
    vram_airlock
fi

if [[ "${MODEL_TARGET}" == "chatbot" || "${MODEL_TARGET}" == "all" ]]; then
    echo -e "\n▶️  [TRAINING 2/2]: Executing Conversational Chatbot Fine-Tuning (train_chatbot.py)..."
    "${PYTHON_CMD}" -u train_chatbot.py
fi

echo -e "\n════════════════════════════════════════════════════════════════════════════════"
echo "✅ STAGE 2 COMPLETE: Selected SLM models successfully fine-tuned and exported."
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📁 Execution logs saved to: ${LOG_DIR}"
echo "👉 Next step: Run 'bash ml/scripts/03_evaluate_models.sh' to execute functional & adversarial certification."
