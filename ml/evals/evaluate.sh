#!/usr/bin/env bash
# ==============================================================================
# evaluate.sh – Direct Alias for Master Model Evaluation & Certification
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Delegates directly to the master verify.sh harness
bash "${SCRIPT_DIR}/verify.sh" "$@"
