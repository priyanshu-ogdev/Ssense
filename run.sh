#!/bin/bash

# 🚨 Exit immediately if any script in the pipeline fails
set -e

# Setup logging
LOG_FILE="pipeline_execution.log"
exec > >(tee -i $LOG_FILE)
exec 2>&1

echo "======================================================="
echo "🚀 INITIATING SSENSE END-TO-END DPDP ALIGNMENT PIPELINE"
echo "======================================================="

# Load conda environment natively (adjust path if your conda init is different)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ssense

echo ""
echo "▶️ [PHASE 1/5]: Forging Synthetic Data..."
cd ~/ssense/ml/data-forge
python gan_forge.py

echo ""
echo "▶️ [PHASE 2/5]: Building Deterministic Rust Tree..."
python build_tree.py

echo ""
echo "▶️ [PHASE 3/5]: Aligning Data Schemas for Unsloth..."
python prepare_unsloth_data.py

echo ""
echo "▶️ [PHASE 4/5]: Executing Supervised Fine-Tuning (SFT)..."
cd ~/ssense/ml/slm-training
python train_sft.py

echo ""
echo "▶️ [PHASE 5/5]: Executing Direct Preference Optimization (DPO)..."
python train_dpo.py

echo ""
echo "======================================================="
echo "✅ PIPELINE COMPLETE: GGUF Models ready for deployment."
echo "======================================================="