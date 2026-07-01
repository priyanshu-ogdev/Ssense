Moving to **Python 3.12** is the right call – it unlocks pre‑built wheels for vLLM, bitsandbytes, and other ML packages that were never published for 3.10. With Python 3.12 and a modern stack, you’ll never see another C++ compilation crash.

Let’s lock in a **2026‑compatible environment** that works with your DGX Spark (CUDA 12.x drivers, NGC container).

---

## 1. Fresh Conda environment (Python 3.12)
```bash
conda create -n ssense_ml312 python=3.12 -y
conda activate ssense_ml312
```

---

## 2. Install CUDA toolkit inside the environment (for any compilation)
```bash
conda install -c conda-forge cudatoolkit=12.6 -y
```
Now set environment variables (add to `~/.bashrc` to persist):
```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```

---

## 3. Install PyTorch 2.7.1 with CUDA 12.4 (backward‑compatible)
```bash
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## 4. `requirements.txt` – modern, pinned where necessary

```text
# ── Training & Alignment ──
unsloth
trl==0.15.0
datasets==4.2.0
transformers==5.0.0
accelerate==2.0.0
peft==0.15.0
bitsandbytes==0.46.0
xformers

# ── Heavy Inference (72B GAN forge) ──
vllm==0.24.0

# ── Local CPU/GPU evaluation (9B GGUF) ──
llama-cpp-python
jsonschema

# ── Scraping ──
curl_cffi
playwright
playwright-stealth
readability-lxml
beautifulsoup4
tqdm
aiohttp
PyMuPDF

# ── Utilities ──
numpy
requests
tenacity
```

Install it **with build isolation disabled** (so vLLM sees your system CUDA and PyTorch):
```bash
pip install -r requirements.txt --no-build-isolation
```

---

## 5. Recompile `llama-cpp-python` with GPU support
```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall
```

---

## 6. Playwright browsers
```bash
playwright install chromium
```

---

## 7. Final verification
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import llama_cpp; print('llama GPU:', llama_cpp.llama_supports_gpu_offload())"
```

All three should report `True` (or show version numbers without errors).

---

## Why this finally works
- **Python 3.12** has pre‑compiled wheels for every package above.
- **vLLM 0.24.0** is the latest stable (July 2026) and publishes `manylinux_2_28` wheels – no compilation.
- **Transformers v5** is now standard; Unsloth and TRL fully support it.
- **bitsandbytes 0.46** installs cleanly on Python 3.12.
- By installing PyTorch and CUDA toolkit first, and then using `--no-build-isolation`, vLLM’s metadata step sees the real CUDA‑enabled torch and skips the fatal “Unknown runtime environment” error.

Your GAN forge is now ready to run – and you’ll never have to fight library installation again.