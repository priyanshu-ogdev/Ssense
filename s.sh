#!/bin/bash
# Ssense ML Pipeline - Bulletproof DGX Installation

echo "🚀 Upgrading build tools..."
pip install --upgrade pip setuptools wheel build

echo "🔥 Installing PyTorch and xformers with explicit CUDA 12.4 support..."
# We install torch FIRST so vllm's metadata generation can see it
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124

echo "📦 Installing the rest of the stack (Bypassing Build Isolation)..."
# --no-build-isolation is the magic flag. It forces vllm to use the globally installed CUDA torch.
pip install -r requirements.txt --no-build-isolation

echo "🛠️ Recompiling llama-cpp-python with CUDA support for GPU-accelerated GGUF evaluation..."
# Without this, llama-cpp will run purely on CPU, making 9B evaluation painfully slow
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.3.6 --no-cache-dir --force-reinstall

echo "🌐 Installing Playwright browser binaries..."
playwright install chromium

echo "✅ Ssense ML Environment is fully operational."