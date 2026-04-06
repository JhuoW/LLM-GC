#!/bin/bash
set -e

# Verify GPU detection
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
assert torch.cuda.device_count() == 3, f'Expected 3 GPUs, got {torch.cuda.device_count()}'
print(f'   PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'   GPU 0: {torch.cuda.get_device_name(0)}')
print(f'   Total GPUs: {torch.cuda.device_count()}')
"

# LlamaFactory(datasets<=4.0.0)
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics,deepspeed]"
cd ..

# HuggingFace ecosystem
pip install "transformers>=4.51,<5.0" \
    accelerate \
    peft \
    trl \
    bitsandbytes \
    "huggingface_hub<1.0.0"

pip install scipy sentencepiece protobuf

# Flash-Attention 2 
# FA3/FA4 do NOT work on SM120, FA2 works.
TORCH_CUDA_ARCH_LIST="12.0" pip install flash-attn==2.7.4.post1 --no-build-isolation || \
    echo "flash-attn build failed — will use PyTorch SDPA instead (still efficient)"

# vLLM (must use cu128 index for SM120 kernel support)
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

# FlashInfer (vLLM on SM120)
pip install flashinfer-python flashinfer-cubin
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128


pip uninstall s3fs -y 2>/dev/null || true


echo ""
echo "=== Dependency Check ==="
pip check 2>&1 | grep -E "llamafactory|datasets|fsspec|torch" || echo " No conflicts detected"

echo ""
echo "=== Installed Versions ==="
python -c "
import torch, transformers, datasets, accelerate, peft, trl
print(f'torch:          {torch.__version__}')
print(f'CUDA:           {torch.version.cuda}')
print(f'transformers:   {transformers.__version__}')
print(f'datasets:       {datasets.__version__}')
print(f'accelerate:     {accelerate.__version__}')
print(f'peft:           {peft.__version__}')
print(f'trl:            {trl.__version__}')
try:
    import deepspeed; print(f'deepspeed:      {deepspeed.__version__}')
except: pass
try:
    import flash_attn; print(f'flash-attn:     {flash_attn.__version__}')
except: print('flash-attn:     not installed (using PyTorch SDPA)')
try:
    import vllm; print(f'vllm:           {vllm.__version__}')
except: pass
try:
    import flashinfer; print(f'flashinfer:     installed')
except: pass
"
# torch:          2.10.0+cu128
# CUDA:           12.8
# transformers:   4.57.3
# datasets:       4.0.0
# accelerate:     1.11.0
# peft:           0.18.1
# trl:            0.24.0
# deepspeed:      0.18.2
# flash-attn:     2.7.4.post1
# vllm:           0.18.0