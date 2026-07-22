#!/usr/bin/env bash
# ==============================================================================
# 03_evaluate_models.sh – Stage 3: Functional & Adversarial Model Certification (DGX Spark)
# ==============================================================================
# ⚠️ SPECIALIZED PIPELINE NOTICE:
# This script is exclusively engineered for certifying the Indian Digital Personal Data Protection (DPDP) Act 2023 &
# Rules 2025 specialized legal models (`Qwen/Qwen3.5-9B`) on NVIDIA DGX Spark infrastructure. It verifies all 13
# certification thresholds across accuracy, grammar, TTR fluidity, and adversarial security. Not a generic wrapper.
#
# Prerequisites for DGX Spark Execution:
#   - NVIDIA DGX / GPU environment (`TOKENIZERS_PARALLELISM=false`)
#   - Python 3.10+ / 3.12 with CUDA 12.x support
#   - Packages installed (`pip install -r requirements.txt`): unsloth, trl, transformers, vllm, datasets, torch, jsonschema
# ==============================================================================
# This script executes the master evaluation suite across all 5 evaluation modules:
#   1. Pillar 1 & 5: Schema Compliance & Inference Efficiency (run_grammar_evals.py)
#   2. Pillar 2-4: Violation F1, Trust MAE, & Zero Hallucination (run_accuracy_evals.py)
#   3. Chatbot Evals: Statutory Accuracy, TTR Vocabulary Fluidity, & Schema Bleed (run_chatbot_evals.py)
#   4. Red-Team Hallucination Benchmark: Statutory Trap Resistance (run_hallucination_benchmark.py)
#   5. Adversarial Vulnerability Suite: NIAH 20k context recall, Prompt Injection refusal,
#      Anti-Sycophancy false premise correction, and JSON Fuzzing resilience (run_security_evals.py)
#
# Usage:
#   bash scripts/03_evaluate_models.sh [options]
#
# Options:
#   --backend <unsloth|vllm|llamacpp>    Backend inference engine (default: unsloth)
#   --audit-model <path>                 Path to Forensic Auditor model
#   --chatbot-model <path>               Path to Conversational Chatbot model
#   --audit-lora <name>                  LoRA routing name for vLLM auditor endpoint
#   --chatbot-lora <name>                LoRA routing name for vLLM chatbot endpoint
#   --vllm-url <url>                     vLLM endpoint URL
#   --skip-run                           Skip inference runs and only aggregate existing reports
#   -h, --help                           Show help menu and exit
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: bash scripts/03_evaluate_models.sh [options]"
            echo "Options:"
            echo "  --backend <unsloth|vllm|llamacpp>    Backend inference engine (default: unsloth)"
            echo "  --audit-model <path>                 Path to Forensic Auditor model (default: ../models/audit-model-final)"
            echo "  --chatbot-model <path>               Path to Conversational Chatbot model (default: ../models/chatbot-model-final)"
            echo "  --audit-lora <name>                  LoRA routing name for vLLM auditor endpoint (default: audit)"
            echo "  --chatbot-lora <name>                LoRA routing name for vLLM chatbot endpoint (default: chatbot)"
            echo "  --vllm-url <url>                     vLLM endpoint URL (default: http://localhost:8000/v1/completions)"
            echo "  --skip-run                           Skip inference evaluation runs and only aggregate existing reports"
            echo "  -h, --help                           Show this help message and exit"
            exit 0
            ;;
        *)
            # Forward all arguments directly to verify.sh
            break
            ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export LOG_DIR="${REPO_ROOT}/logs/evaluate_models_${TIMESTAMP}"
export PYTHONUNBUFFERED=1
mkdir -p "${LOG_DIR}"

MASTER_LOG="${LOG_DIR}/00_evaluate_models_master.log"
exec > >(tee -i "${MASTER_LOG}")
exec 2>&1

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🏆 INITIATING STAGE 3: FUNCTIONAL & ADVERSARIAL MODEL CERTIFICATION"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "   • Repository Root: ${REPO_ROOT}"
echo "   • Logs Directory:  ${LOG_DIR}"
echo "   • Forwarded Args:  $*"
echo "════════════════════════════════════════════════════════════════════════════════"

cd "${REPO_ROOT}/ml/evals"

# Execute verify.sh inside ml/evals with forwarded arguments
bash verify.sh "$@"
EXIT_CODE=$?

echo -e "\n════════════════════════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ STAGE 3 COMPLETE: Models certified across all functional and adversarial thresholds!"
else
    echo "❌ STAGE 3 COMPLETE: Certification checks encountered threshold infractions or errors."
fi
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📁 Execution logs saved to: ${LOG_DIR}"
echo "📄 Scorecard summary saved to: ${REPO_ROOT}/ml/evals/reports/final_model_certification_report.md"

exit ${EXIT_CODE}
