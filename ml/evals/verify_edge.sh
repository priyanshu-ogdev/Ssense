#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# verify_edge.sh – GGUF Edge-Deployment Certification Harness for DPDP SLMs
#
# Runs the certification suite specifically against the quantized Q4_K_M GGUF
# model via the llama.cpp backend. This is a SEPARATE certification from the
# full-precision (BF16) certification run.
#
# WHY: Quantization from BF16 → Q4_K_M compresses weights by ~4x. While
# language fluidity usually survives, NIAH context recall, RAG contextual
# precision, and exact numerical extraction (penalties/days) degrade heavily
# under 4-bit quantization. The edge deployment model needs its own
# certification, not an inherited one.
#
# THRESHOLDS: Relaxed from full-precision targets:
#   - RAG Recall@3: 85% (vs 95% full-precision)
#   - Schema Compliance: 95% (vs 98%)
#   - Weighted F1: 0.80 (vs 0.88)
#
# Usage:
#   bash ml/evals/verify_edge.sh --audit-model ../models/audit-model-final-gguf
#   bash ml/evals/verify_edge.sh --audit-model ../models/audit-model-final-gguf/model.Q4_K_M.gguf
# ═══════════════════════════════════════════════════════════════════════════

set -uo pipefail

export TOKENIZERS_PARALLELISM="false"

AUDIT_MODEL="../models/audit-model-final-gguf"
CHATBOT_MODEL="../models/chatbot-model-final-gguf"
SKIP_RUN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --audit-model)
      AUDIT_MODEL="$2"
      shift 2
      ;;
    --chatbot-model)
      CHATBOT_MODEL="$2"
      shift 2
      ;;
    --skip-run)
      SKIP_RUN=true
      shift
      ;;
    -h|--help)
      echo "Usage: verify_edge.sh [options]"
      echo ""
      echo "GGUF Edge-Deployment Certification Harness"
      echo "Uses llama.cpp backend with relaxed thresholds for quantized models."
      echo ""
      echo "Options:"
      echo "  --audit-model <path>     Path to GGUF auditor model (file or directory)"
      echo "  --chatbot-model <path>   Path to GGUF chatbot model (file or directory)"
      echo "  --skip-run               Skip eval execution, aggregate existing reports only"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🔧 STARTING GGUF EDGE-DEPLOYMENT CERTIFICATION HARNESS (llama.cpp)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "   • Backend:        llamacpp (Q4_K_M quantized)"
echo "   • Auditor Model:  ${AUDIT_MODEL}"
echo "   • Chatbot Model:  ${CHATBOT_MODEL}"
echo "   • Thresholds:     RELAXED (edge-deployment tolerances)"
echo "════════════════════════════════════════════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
  PYTHON_CMD="python3"
fi

ARGS=("--backend" "llamacpp" "--audit-model-path" "${AUDIT_MODEL}" "--chatbot-model-path" "${CHATBOT_MODEL}" "--base-model-label" "GGUF-Q4_K_M-Edge")
if [ "$SKIP_RUN" = true ]; then
  ARGS+=("--skip-run")
fi

EXIT_CODE=0
"${PYTHON_CMD}" "${SCRIPT_DIR}/verify.py" "${ARGS[@]}" || EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
  echo "✅ GGUF EDGE CERTIFICATION PASSED! Quantized model meets relaxed thresholds."
else
  echo "❌ GGUF EDGE CERTIFICATION FAILED! Quantization degradation exceeds acceptable limits."
  echo "   Review scorecard — consider re-quantizing with Q5_K_M or Q6_K for better accuracy."
fi

exit ${EXIT_CODE}
