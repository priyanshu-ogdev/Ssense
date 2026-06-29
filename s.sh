# 1. Install PyTorch with explicit CUDA 12.1 support
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install vLLM (Will now use the pre-built wheel because Torch matches)
pip install vllm==0.6.4.post1

# 3. Install Unsloth (Using their dedicated wheel to prevent xformers conflicts)
pip install unsloth==2024.11.6

# 4. Install Flash-Attention (No build isolation prevents ninja compiler errors)
pip install flash-attn==2.7.2.post1 --no-build-isolation