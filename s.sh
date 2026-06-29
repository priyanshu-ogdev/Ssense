# 1. Inject the NVIDIA CUDA 12.1 Toolkit (provides 'nvcc')
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y

# 2. Inject modern C++ compilers into Conda to handle vLLM's advanced kernels
conda install -c conda-forge gcc gxx -y

# 3. Explicitly map the CUDA path for vLLM's setup script
export CUDA_HOME=$CONDA_PREFIX

# 4. Execute the source build (This will take 10-20 minutes on the DGX)
pip install vllm==0.6.4.post1 --no-build-isolation