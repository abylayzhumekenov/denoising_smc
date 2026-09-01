#!/bin/bash
# One-time environment setup on an Ibex login node. Run from the repo root:
#   bash slurm/setup_env.sh
#
# Confirmed against real Ibex `module avail cuda` / `module avail python` output (Aug 2026):
#   CUDA modules available: 11.7.1, 11.8, 12.1, 12.2, 12.4.1
#   Python modules available: 3.11.0, 3.12.1  -- ONLY these two, no 3.8/3.9/3.10 at all.
#
# DEVIATION FROM AGENTS.md: AGENTS.md states PyTorch-1.12.1 compatibility, which needs Python
# 3.8-3.10 -- that range does not exist as a module on Ibex, so this is not "pick the closest
# wheel to the CUDA module" (as slurm/README.md originally suggested trying first), it's a hard
# incompatibility. There is no Python 3.8-3.10 to fall back to here, so we move to a modern torch
# 2.x build instead of the pin. Risk assessment: this repo's one private-API dependency
# (training/networks.py's AttentionOp using torch._softmax_backward_data) was already validated
# to work under ordinary single backward() calls (which is all GEM/generate_burgers_gem.py uses --
# see smc/scripts_2/hutchinson_findings.md); the double-backward route that originally motivated checking
# this is a separate, deprioritized track. Verify with `python -m smc.scripts_2.check_gem_tds_real_model`
# immediately after this script finishes, before trusting any full run.
#
# torch==2.7.1 confirmed (via PyPI classifiers) to support Python 3.9-3.13, so both available
# Ibex Python modules work; picking 3.11.0 as the more conservative/widely-used of the two.
set -euo pipefail

CUDA_MODULE="cuda/11.8"        # matches the cu118 wheel below
PYTHON_MODULE="python/3.11.0"  # the more conservative of Ibex's two available Python modules

module load "$CUDA_MODULE"
module load "$PYTHON_MODULE"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Modern torch 2.x, NOT the AGENTS.md-pinned 1.12.1 -- see the comment block above for why.
# cu118 matches CUDA_MODULE above.
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu118

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
