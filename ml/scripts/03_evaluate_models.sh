#!/usr/bin/env bash
# ==============================================================================
# 03_evaluate_models.sh – Stage 3: Master Certification Orchestrator (DGX Spark)
# ==============================================================================
# SOTA Upgrades Implemented:
# 1. Single CI/CD Entrypoint: Absorbs legacy verify.sh, evaluate.sh, and verify_edge.sh.
# 2. Native Edge Routing: Pass `--edge` to automatically target GGUF binaries via llama.cpp.
# 3. Direct Python Invocation: Bypasses redundant bash wrappers for flawless exit-code trapping.
# 4. Hardware Env Overrides: Forces TOKENIZERS_PARALLELISM=false for safe VRAM airlocking.
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ML_DIR}/.." && pwd)"

# ─── DGX & PYTORCH ENVIRONMENT HARDENING ──────────────────────────────────────
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# ─── DEFAULT CONFIGURATION ────────────────────────────────────────────────────
BACKEND="unsloth"
AUDIT_MODEL="../models/audit-model-final"
CHATBOT_MODEL="../models/chatbot-model-final"
AUDIT_LORA="audit"
CHATBOT_LORA="chatbot"
VLLM_URL="http://localhost:8000/v1/completions"
EDGE_MODE=false
EXTRA_ARGS=()

# ─── ARGUMENT PARSING ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend) BACKEND="$2"; shift 2 ;;
        --audit-model) AUDIT_MODEL="$2"; shift 2 ;;
        --chatbot-model) CHATBOT_MODEL="$2"; shift 2 ;;
        --audit-lora) AUDIT_LORA="$2"; shift 2 ;;
        --chatbot-lora) CHATBOT_LORA="$2"; shift 2 ;;
        --vllm-url) VLLM_URL="$2"; shift 2 ;;
        --skip-run) EXTRA_ARGS+=("--skip-run"); shift ;;
        --edge) EDGE_MODE=true; shift ;;
        -h|--help)
            echo "Usage: bash ml/scripts/03_evaluate_models.sh [options]"
            echo "Options:"
            echo "  --backend <unsloth|vllm|llamacpp>    Backend engine (default: unsloth)"
            echo "  --edge                               Auto-configures llama.cpp to test quantized GGUF Edge models"
            echo "  --audit-model <path>                 Path to Auditor model"
            echo "  --chatbot-model <path>               Path to Chatbot model"
            echo "  --skip-run                           Skip inference, aggregate existing JSON reports only"
            exit 0
            ;;
        *) 
            echo "Unknown argument: $1"
            exit 1 
            ;;
    esac
done

# ─── EDGE/GGUF AUTO-CONFIGURATION ─────────────────────────────────────────────
if [[ "$EDGE_MODE" == true ]]; then
    BACKEND="llamacpp"
    AUDIT_MODEL="../models/audit-model-final-gguf"
    CHATBOT_MODEL="../models/chatbot-model-final-gguf"
    EXTRA_ARGS+=("--base-model-label" "GGUF-Q4_K_M-Edge")
fi

# ─── DGX LOGGING SETUP ────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/03_evaluate_models_${TIMESTAMP}.log"

# Setup dual-logging to both terminal and log file safely
exec > >(tee -i "${MASTER_LOG}")
exec 2>&1

echo "════════════════════════════════════════════════════════════════════════════════"
if [[ "$EDGE_MODE" == true ]]; then
    echo "🔧 INITIATING STAGE 3: GGUF EDGE-DEPLOYMENT CERTIFICATION HARNESS"
else
    echo "🏆 INITIATING STAGE 3: FUNCTIONAL & ADVERSARIAL MODEL CERTIFICATION"
fi
echo "════════════════════════════════════════════════════════════════════════════════"
echo "   • Inference Backend: ${BACKEND}"
echo "   • Auditor Model:     ${AUDIT_MODEL}"
echo "   • Chatbot Model:     ${CHATBOT_MODEL}"
echo "   • Master Log:        ${MASTER_LOG}"
echo "════════════════════════════════════════════════════════════════════════════════"

# ─── EXECUTE PYTHON ORCHESTRATOR ──────────────────────────────────────────────
cd "${ML_DIR}/evals"

EXIT_CODE=0
python3 verify.py \
    --backend "${BACKEND}" \
    --audit-model-path "${AUDIT_MODEL}" \
    --chatbot-model-path "${CHATBOT_MODEL}" \
    --audit-lora-name "${AUDIT_LORA}" \
    --chatbot-lora-name "${CHATBOT_LORA}" \
    --vllm-url "${VLLM_URL}" \
    "${EXTRA_ARGS[@]}" || EXIT_CODE=$?

echo -e "\n════════════════════════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ STAGE 3 COMPLETE: Models certified across all 18 functional thresholds!"
else
    echo "❌ STAGE 3 COMPLETE: Certification checks encountered threshold infractions."
    echo "   (Review the scorecard. Threshold failures indicate model capability limits,"
    echo "    not pipeline bugs. Adjust SimPO training data to improve.)"
fi
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📁 Execution logs saved to: ${MASTER_LOG}"

exit ${EXIT_CODE}