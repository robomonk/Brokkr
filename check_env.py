import torch
import bitsandbytes as bnb

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")

try:
    import flash_attn
    print("Flash Attention: INSTALLED ✅")
except ImportError:
    print("Flash Attention: NOT INSTALLED ❌")