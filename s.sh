# 1. Create & activate environment
conda create -n ssense_ml python=3.10 -y
conda activate ssense_ml

# 2. Install numpy + PyTorch (CUDA 12.1)
pip install numpy==1.26.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Now vLLM will build correctly
pip install vllm==0.6.4.post1

# 4. Rest of the stack
pip install unsloth==2024.11.6 trl==0.12.2 datasets==3.2.0 transformers==4.46.3 accelerate==1.2.1 peft==0.14.0 bitsandbytes==0.45.0
pip install flash-attn==2.7.2.post1 --no-build-isolation


