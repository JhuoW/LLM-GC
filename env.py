import torch
import torch, transformers, datasets, accelerate, peft, trl

print(torch.cuda.is_available())           # True
print(torch.cuda.device_count())           # 3
print(torch.cuda.get_device_name(0))       # NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
print(torch.version.cuda)                  # 12.8
print(torch.__version__)                  # 2.10.0+cu128

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