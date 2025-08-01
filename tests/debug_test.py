import torch
import ctypes
from pathlib import Path
import platform
import os

print("==== GPU + cuDNN Diagnostic ====\n")

# 1. Torch version + CUDA version
print(f"Torch version       : {torch.__version__}")
print(f"CUDA version (Torch): {torch.version.cuda}")
print(f"Python version      : {platform.python_version()}")
print(f"Platform            : {platform.system()} {platform.release()}")
print(f"Device Count        : {torch.cuda.device_count()}")

# 2. GPU availability
if not torch.cuda.is_available():
    print("❌ CUDA not available — torch.cuda.is_available() is False")
    exit(1)

print(f"GPU Name            : {torch.cuda.get_device_name(0)}")
print(f"GPU Memory (Total)  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 3. Try allocating GPU tensor
try:
    x = torch.rand((8192, 8192), device="cuda")
    print(f"✅ Tensor allocated on GPU: {x.shape}")
except Exception as e:
    print(f"❌ Tensor allocation failed: {e}")
    exit(1)

# 4. Locate and try loading cuDNN DLL
torch_lib_path = Path(torch.__file__).parent / "lib"
cudnn_dll_path = torch_lib_path / "cudnn_ops64_9.dll"

print(f"\nChecking cuDNN DLL at: {cudnn_dll_path}")
if not cudnn_dll_path.exists():
    print("❌ cudnn_ops64_9.dll not found!")
    exit(1)

try:
    ctypes.CDLL(str(cudnn_dll_path))
    print("✅ cuDNN DLL loaded successfully")
except Exception as e:
    print(f"❌ Failed to load cuDNN DLL: {e}")
    exit(1)

# 5. Print env PATH entries for CUDA leftovers
print("\nCUDA-related PATH entries:")
cuda_paths = [p for p in os.environ["PATH"].split(";") if "cuda" in p.lower()]
if not cuda_paths:
    print("✅ No CUDA clutter in PATH")
else:
    for path in cuda_paths:
        print(f"⚠️ {path}")

print("\n==== All checks passed! Ready for real pipeline ====")
