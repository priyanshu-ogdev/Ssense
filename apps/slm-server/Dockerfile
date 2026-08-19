# ═══════════════════════════════════════════════════════════════════════════
# SOTA SLM SERVER DOCKERFILE (GPU / vLLM 0.27.1 / DGX Optimized)
# ═══════════════════════════════════════════════════════════════════════════

# 1. Use the official vLLM Base Image (Contains CUDA, PyTorch, Flash-Attn)
FROM vllm/vllm-openai:v0.27.1

# 2. Environment Variables for SOTA GPU Stability & Memory Control
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VLLM_WORKER_MULTIPROCESS_METHOD=spawn \
    RAY_DISABLE_MEMORY_MONITOR=1 \
    HF_HOME=/app/.cache/huggingface \
    MODELS_DIR=/app/models

WORKDIR /app

# 3. Install lightweight system dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Application Dependencies (PyTorch & vLLM are already in the base)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the microservice source code
COPY . /app

# 6. Expose API Port (Aligned with Nginx Gateway)
EXPOSE 8000

# 7. Strict Healthcheck (Start-period is 90s to allow the 15GB model to load into VRAM)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 8. SOTA LAUNCH COMMAND: MUST use 1 worker for vLLM PagedAttention to function
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]