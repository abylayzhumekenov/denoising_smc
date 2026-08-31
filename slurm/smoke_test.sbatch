#!/bin/bash
# Smoke test: confirms CUDA + the environment work, then runs the closed-form correctness check
# (smc/check_gem_tds_real_model.py) and a tiny 4-particle/100-step generate_burgers_gem run.
# Submit BEFORE any full-scale job:
#   sbatch slurm/smoke_test.sbatch
#
# Partition/GRES confirmed from real `sinfo -o "%P %G %l"` output (Aug 2026): `debug` is the
# fast-turnaround GPU partition (gpu:p6000:2 or gpu:v100:2 per node, 2h wall-time limit) -- ideal
# for a quick check like this one. `batch` has NO gpu (was the original placeholder -- wrong).
#SBATCH --job-name=smc_smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
mkdir -p slurm/logs
source .venv/bin/activate

echo "=== device check ==="
python3 -c "
import torch
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (unexpected on a GPU node)')
"

echo "=== correctness self-check (smc/check_gem_tds_real_model.py) ==="
python -m smc.check_gem_tds_real_model --config configs/burgers.yaml

echo "=== tiny smoke run (4 particles, 100 steps) ==="
python -m scripts.generate_burgers_gem --config configs/burgers.yaml \
  --n-particles 4 --num-steps 100 --out slurm/logs/smoke_burgers_gem.npz

echo "=== done -- check timing above to size --time for the full run ==="
