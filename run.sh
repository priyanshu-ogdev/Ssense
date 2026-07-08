#!/bin/bash

# 🚨 STRICT EXECUTION MODES
set -e               # Exit immediately if a command exits with a non-zero status
set -o pipefail      # Catch errors inside piped commands (e.g., script.py | tee)

# 📂 LOGGING SETUP
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$HOME/ssense/logs/run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/00_master_pipeline.log"
exec > >(tee -i "$MASTER_LOG")
exec 2>&1

# 🛑 GLOBAL ERROR TRAP
trap 'echo -e "\n❌ FATAL: Pipeline crashed at line $LINENO. Inspect logs in $LOG_DIR for the traceback." | tee -a "$MASTER_LOG"' ERR

echo "======================================================="
echo "🚀 INITIATING SSENSE END-TO-END DPDP ALIGNMENT PIPELINE"
echo "======================================================="
echo "📁 Logs mapped to: $LOG_DIR"

# Load conda environment
echo -e "\n⚙️ Bootstrapping Conda Environment 'ssense'..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ssense

# ---------------------------------------------------------
# STAGE 1: DATA FORGE & PREPARATION (vLLM 72B)
# ---------------------------------------------------------
cd ~/ssense/ml/data-forge

echo -e "\n▶️ [PHASE 1/5]: Forging Synthetic Data (GAN Forge)..."
python gan_forge.py 2>&1 | tee "$LOG_DIR/01_gan_forge.log"

echo -e "\n▶️ [PHASE 2/5]: Building Deterministic Rust Tree..."
python build_dpdp_tree.py 2>&1 | tee "$LOG_DIR/02_build_tree.log"

echo -e "\n▶️ [PHASE 3/5]: Aligning Data Schemas for Unsloth..."
python prepare_unsloth_data.py 2>&1 | tee "$LOG_DIR/03_prepare_data.log"

# ---------------------------------------------------------
# STAGE 2: THE VRAM AIRLOCK (MEMORY SANITIZATION)
# ---------------------------------------------------------
echo -e "\n======================================================="
echo "🧹 VRAM AIRLOCK: PURGING 72B WEIGHTS FROM UNIFIED MEMORY"
echo "======================================================="

# 1. Terminate Ray distributed daemons
echo "   [1/4] Terminating Ray background daemons..."
ray stop -f 2>/dev/null || true

# 2. Murder orphaned Python/vLLM processes holding CUDA context
echo "   [2/4] Hunting zombie vLLM processes..."
pkill -9 -f vllm || true
pkill -9 -f ray:: || true

# 3. Clear PyTorch IPC shared memory blocks
echo "   [3/4] Flushing POSIX shared memory leaks..."
rm -rf /dev/shm/* 2>/dev/null || true

# 4. Drop Linux page caches (Warning: Requires sudo privileges)
echo "   [4/4] Dropping OS file caches..."
sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

echo "⏳ Waiting 15 seconds for Nvidia driver state to reset..."
sleep 15
echo "✅ VRAM Purged. Environment ready for Unsloth."

# ---------------------------------------------------------
# STAGE 3: SLM ALIGNMENT (Unsloth 9B)
# ---------------------------------------------------------
cd ~/ssense/ml/slm-training

echo -e "\n======================================================="
echo "🏋️ INITIATING MODEL TRAINING"
echo "======================================================="

echo -e "\n▶️ [PHASE 4/5]: Executing Supervised Fine-Tuning (SFT)..."
python train_sft.py 2>&1 | tee "$LOG_DIR/04_train_sft.log"

echo -e "\n▶️ [PHASE 5/5]: Executing Direct Preference Optimization (DPO)..."
python train_dpo.py 2>&1 | tee "$LOG_DIR/05_train_dpo.log"

echo -e "\n======================================================="
echo "✅ PIPELINE COMPLETE: GGUF Models compiled and exported."
echo "======================================================="
echo "📁 Final execution logs saved to: $LOG_DIR"