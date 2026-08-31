#!/bin/bash
# One-time environment setup on an Ibex login node. Run from the repo root:
#   bash slurm/setup_env.sh
#
# TODO (Ibex-specific, fill in after `module avail cuda` / `module avail python`):
#   - CUDA_MODULE: an available CUDA module close to what the chosen torch wheel expects.
#   - PYTHON_MODULE: an available Python 3.8-3.10 module (AGENTS.md's stated PyTorch-1.12.1
#     compatibility range). If Ibex's system python3 is already in that range, this can be
#     dropped and the plain `python3 -m venv` below used instead.
set -euo pipefail

CUDA_MODULE="cuda/11.6"        # TODO: verify with `module avail cuda` on Ibex
PYTHON_MODULE="python/3.9"     # TODO: verify with `module avail python` on Ibex

module load "$CUDA_MODULE"
module load "$PYTHON_MODULE" || true   # harmless if the system default python is already 3.8-3.10

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Pinned to match AGENTS.md's stated compatibility (PyTorch 1.12.1). cu116 matches CUDA_MODULE
# above -- if that module isn't available, pick the closest cuXXX wheel to whatever CUDA module
# Ibex does offer, from https://pytorch.org/get-started/previous-versions/#v1121shell:
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt

python3 - <<'EOF'
import torch
print(f"torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"float64 tensor on cuda: {torch.zeros(1, dtype=torch.float64, device='cuda')}")
EOF

echo "Setup done. If cuda available printed False, the CUDA_MODULE/torch wheel pairing above"
echo "needs adjusting -- check 'nvidia-smi' and 'module avail cuda' for what's actually offered."
