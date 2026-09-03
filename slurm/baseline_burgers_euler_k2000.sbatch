#!/bin/bash
# Same as baseline_burgers_k2000.sbatch but with the 2nd-order Heun correction switched off
# (configs/burgers_euler.yaml sets generate.integrator: euler), so this does plain 1st-order
# Euler ODE integration -- 1 network call/step, same as GEM's SDE step count. Same K=2000,
# same zeta_obs/zeta_pde, same seed as the Heun baseline and the GEM K=2000 run, so all three
# numbers are directly comparable.
#
# Goal: isolate whether the ~30x Heun-vs-GEM error gap is caused by the *2nd-order correction*
# (in which case dropping it here should move the baseline's error a lot, maybe toward GEM's)
# or whether it's something else entirely (in which case Euler should still track close to
# Heun's baseline, since both remain deterministic/noise-free).
#
#   sbatch slurm/baseline_burgers_euler_k2000.sbatch
#SBATCH --job-name=baseline_euler_k2000
#SBATCH --chdir=/home/alvarem/denoising_smc
#SBATCH --partition=debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miguelangel.alvarezballesteros@kaust.edu.sa

set -euo pipefail
mkdir -p slurm/logs

# Must match setup_env.sh exactly -- see smoke_test.sbatch's comment on this line for why this is
# required (Slurm jobs don't inherit login-node modules).
module load cuda/11.8
module load python/3.11.0

source .venv/bin/activate

echo "=== device check ==="
python3 -c "
import torch
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (unexpected on a GPU node)')
"

echo "=== 1st-order Euler baseline, configs/burgers_euler.yaml (iterations: 2000, batch_size: 1) ==="
python3 generate_pde.py --config configs/burgers_euler.yaml

echo "=== done ==="
