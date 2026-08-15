import sys

def run_diagnostics():
    print("=" * 65)
    print("🔍 QWEN3.5 FAST-PATH KERNELS DIAGNOSTIC SUITE")
    print("=" * 65)

    # 1. PyTorch & CUDA Check
    try:
        import torch
        print(f"✅ PyTorch Version : {torch.__version__}")
        print(f"✅ CUDA Available   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   └── GPU Device    : {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch Check FAILED: {e}")
        return

    print("-" * 65)

    # 2. Flash Linear Attention (fla) Check
    try:
        import fla
        from fla.modules import FusedRMSNormGated
        fla_version = getattr(fla, "__version__", "Installed (No __version__ attribute)")
        print(f"✅ flash-linear-attention (fla) : {fla_version}")
        print("   └── Module 'fla.modules.FusedRMSNormGated': LOADED SUCCESSFULLY")
    except ImportError as e:
        print(f"❌ flash-linear-attention (fla) : FAILED")
        print(f"   └── Error Detail: {e}")

    print("-" * 65)

    # 3. Causal-Conv1D Check
    try:
        import causal_conv1d
        from causal_conv1d import causal_conv1d_fn
        conv_version = getattr(causal_conv1d, "__version__", "Installed")
        print(f"✅ causal-conv1d                : {conv_version}")
        print("   └── Function 'causal_conv1d_fn': LOADED SUCCESSFULLY")
    except ImportError as e:
        print(f"❌ causal-conv1d                : FAILED")
        print(f"   └── Error Detail: {e}")

    print("=" * 65)

if __name__ == "__main__":
    run_diagnostics()