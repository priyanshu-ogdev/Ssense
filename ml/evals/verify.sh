#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# verify.sh – Master Automated Evaluation & Certification Harness for DPDP SLMs
#
# Runs all 5 Functional & Adversarial Evaluation suites:
# 1. Pillar 1 & 5: Schema Compliance & Efficiency (run_grammar_evals.py)
# 2. Pillar 2, 3, & 4: Violation F1, Trust Score MAE, & Hallucination (run_accuracy_evals.py)
# 3. Conversational Chatbot Benchmarks (run_chatbot_evals.py)
# 4. Red-Team Statutory Hallucination Suite (run_hallucination_benchmark.py)
# 5. Adversarial Security Suite: NIAH, Prompt Injection, Sycophancy, JSON Fuzzing (run_security_evals.py)
#
# Usage:
#   bash ml/evals/verify.sh --backend unsloth --audit-model ../models/audit-model-final --chatbot-model ../models/chatbot-model-final
#   bash ml/evals/verify.sh --backend vllm --vllm-url http://localhost:8000/v1/completions
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Mandatory multiprocessing/tokenizer deadlock defense
export TOKENIZERS_PARALLELISM="false"

BACKEND="unsloth"
AUDIT_MODEL="../models/audit-model-final"
CHATBOT_MODEL="../models/chatbot-model-final"
AUDIT_LORA="audit"
CHATBOT_LORA="chatbot"
VLLM_URL="http://localhost:8000/v1/completions"
SKIP_RUN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --audit-model)
      AUDIT_MODEL="$2"
      shift 2
      ;;
    --chatbot-model)
      CHATBOT_MODEL="$2"
      shift 2
      ;;
    --audit-lora)
      AUDIT_LORA="$2"
      shift 2
      ;;
    --chatbot-lora)
      CHATBOT_LORA="$2"
      shift 2
      ;;
    --vllm-url)
      VLLM_URL="$2"
      shift 2
      ;;
    --skip-run)
      SKIP_RUN=true
      shift
      ;;
    -h|--help)
      echo "Usage: verify.sh [options]"
      echo "Options:"
      echo "  --backend <unsloth|vllm|llamacpp>    Backend inference engine (default: unsloth)"
      echo "  --audit-model <path>                 Path to Forensic Auditor model"
      echo "  --chatbot-model <path>               Path to Conversational Chatbot model"
      echo "  --audit-lora <name>                  LoRA routing name for vLLM auditor endpoint"
      echo "  --chatbot-lora <name>                LoRA routing name for vLLM chatbot endpoint"
      echo "  --vllm-url <url>                     vLLM endpoint URL"
      echo "  --skip-run                           Skip evaluation execution and aggregate reports only"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 STARTING MASTER DPDP SLM EVALUATION & CERTIFICATION HARNESS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "   • Backend:        ${BACKEND}"
echo "   • Auditor Model:  ${AUDIT_MODEL} (vLLM LoRA: ${AUDIT_LORA})"
echo "   • Chatbot Model:  ${CHATBOT_MODEL} (vLLM LoRA: ${CHATBOT_LORA})"
echo "════════════════════════════════════════════════════════════════════════════════"

# Locate verify.py directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
  PYTHON_CMD="python3"
fi

ARGS=("--backend" "${BACKEND}" "--audit-model-path" "${AUDIT_MODEL}" "--chatbot-model-path" "${CHATBOT_MODEL}" "--audit-lora-name" "${AUDIT_LORA}" "--chatbot-lora-name" "${CHATBOT_LORA}" "--vllm-url" "${VLLM_URL}")
if [ "$SKIP_RUN" = true ]; then
  ARGS+=("--skip-run")
fi

"${PYTHON_CMD}" "${SCRIPT_DIR}/verify.py" "${ARGS[@]}"
EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
  echo "✅ MASTER CERTIFICATION PASSED! ALL FUNCTIONAL & ADVERSARIAL THRESHOLDS MET."
else
  echo "❌ MASTER CERTIFICATION FAILED! PLEASE REVIEW SCORECARD FOR THRESHOLD INFRACTIONS."
fi

exit ${EXIT_CODE}
