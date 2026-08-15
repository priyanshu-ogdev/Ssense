#!/usr/bin/env bash
# ==============================================================================
# evaluate.sh – Direct Alias for Stage 3: Functional & Adversarial Certification
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Delegates directly to 03_evaluate_models.sh
bash "${SCRIPT_DIR}/03_evaluate_models.sh" "$@"
