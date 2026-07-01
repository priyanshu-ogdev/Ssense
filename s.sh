# Ensure the CLI is installed
pip install -U "huggingface_hub[cli]"

# Download the vLLM-optimized FP8 model directly into your models directory
huggingface-cli download neuralmagic/Qwen2-72B-Instruct-FP8 \
  --local-dir ./ml/models/Qwen2-72B-Instruct-FP8 \
  --local-dir-use-symlinks False