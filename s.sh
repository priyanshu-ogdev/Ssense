We’ll bypass `pip`’s index entirely and install the exact CUDA‑enabled PyTorch wheels directly. Then we’ll give vLLM a proper CUDA toolkit via Conda so it can compile.

---

### 1. Remove any leftover PyTorch (CPU or broken)
```bash
pip uninstall -y torch torchvision torchaudio
pip cache purge
```

### 2. Install PyTorch 2.5.1 + CUDA 12.1 from direct wheel URLs
These are the exact files for Python **3.10**, Linux **x86_64**.  
*(If you’re not on 3.10, replace `cp310` with your Python tag – check with `python -c "import sys; print('cp{}{}'.format(sys.version_info.major, sys.version_info.minor))"`.)*

```bash
pip install \
  https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl \
  https://download.pytorch.org/whl/cu121/torchvision-0.20.1%2Bcu121-cp310-cp310-linux_x86_64.whl \
  https://download.pytorch.org/whl/cu121/torchaudio-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.5.1+cu121 True
```

---

### 3. Give vLLM a full CUDA toolkit (via Conda)
Even though your driver is CUDA 13, vLLM needs the **headers and compilers** to build. Conda can provide CUDA 12.6, which is perfectly compatible.

```bash
conda install -y -c conda-forge cudatoolkit=12.6
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```

Check that `nvcc` works:
```bash
nvcc --version
```

---

### 4. Now build vLLM (it will use the conda CUDA toolkit)
```bash
export VLLM_TARGET_DEVICE=cuda
pip install vllm==0.6.4.post1 --no-build-isolation
```

This will compile vLLM from source, but now it finds a real CUDA toolkit and succeeds.

---

### 5. Final verification
```bash
python -c "import vllm; print(vllm.__version__)"
```

---

### 6. Install the rest of the packages
```bash
pip install numpy==1.26.4 unsloth==2024.11.6 trl==0.12.2 datasets==3.2.0 transformers==4.46.3 accelerate==1.2.1 peft==0.14.0 bitsandbytes==0.45.0
pip install flash-attn==2.7.2.post1 --no-build-isolation
pip install curl_cffi==0.7.4 playwright==1.51.0 readability-lxml==0.8.1 beautifulsoup4==4.12.3 tqdm==4.66.6 aiohttp==3.11.12 PyMuPDF==1.25.3 llama-cpp-python==0.3.6 jsonschema==4.23.0 requests==2.32.3 tenacity==9.0.0
playwright install chromium
```

Your GAN forge is now ready to run. No more “version not found” or “Unknown runtime environment”.