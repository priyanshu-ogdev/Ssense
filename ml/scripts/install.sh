#!/usr/bin/env bash
# ==============================================================================
# install.sh – DGX Spark (GB10 / aarch64 / CUDA 13.0) Provisioning Engine
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ML_DIR}/.." && pwd)"

echo "════════════════════════════════════════════════════════════════════════"
echo "🚀 INITIATING GB10 HARDWARE-SPECIFIC ML PROVISIONING"
echo "════════════════════════════════════════════════════════════════════════"
echo "   • ML Root:         ${ML_DIR}"
echo "   • Repository Root: ${REPO_ROOT}"

# 1. CRITICAL COMPILER FLAGS FOR BLACKWELL (sm_121) ON CUDA 13.0
export TORCH_CUDA_ARCH_LIST="12.0"
export FLASH_ATTENTION_FORCE_BUILD="TRUE"
export XFORMERS_MORE_DETAILS=1

echo "✅ Environment variables configured: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# ==============================================================================
# PHASE 1: Build Essentials & Base Requirements
# ==============================================================================
echo -e "\n📦 [PHASE 1] Upgrading pip, installing build essentials & requirements..."
python3 -m pip install --upgrade pip wheel setuptools packaging ninja cmake

# Robustly installs requirements.txt from ml directory
python3 -m pip install -r "${ML_DIR}/requirements.txt"

# ==============================================================================
# PHASE 2: Xformers (Source Build for aarch64 + CUDA 13.0)
# ==============================================================================
echo -e "\n🔥 [PHASE 2] Compiling Xformers from source (8 Threads)..."
export MAX_JOBS=8
python3 -m pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.30#egg=xformers --no-build-isolation

# ==============================================================================
# PHASE 3: Flash-Attention (Source Build for aarch64 + CUDA 13.0)
# ==============================================================================
echo -e "\n⚡ [PHASE 3] Compiling Flash-Attention (RAM THROTTLED to 4 Threads)..."
export MAX_JOBS=4
python3 -m pip install -v -U git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3#egg=flash-attn --no-build-isolation

# ==============================================================================
# PHASE 4: Llama-CPP-Python (GGUF Export Backend)
# ==============================================================================
echo -e "\n🦙 [PHASE 4] Compiling Llama-CPP-Python with CUDA 13.0 backend..."
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120" \
    python3 -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# ==============================================================================
# PHASE 5: Unsloth, Unsloth Zoo & BitsAndBytes (IRONCLAD NO-DEPS)
# ==============================================================================
echo -e "\n🦥 [PHASE 5] Installing Unsloth + Unsloth Zoo & BitsAndBytes..."

# Official aarch64 wheel with strict no-deps
python3 -m pip install "bitsandbytes>=0.50.0" --no-deps

python3 -c "import bitsandbytes" || echo "⚠️ bitsandbytes native library notice -- proceeding with adamw_torch optimizer"

# Hard-pinned to Unsloth's exact caps
python3 -m pip install --upgrade --force-reinstall --no-cache-dir --no-deps \
    "unsloth==2026.8.15" "unsloth_zoo==2026.8.10"

echo -e "\n════════════════════════════════════════════════════════════════════════"
echo "✅ SUCCESS: GB10 aarch64 Provisioning Complete!"
echo "════════════════════════════════════════════════════════════════════════"
