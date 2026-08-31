#!/bin/bash
# Full-scale GEM+SMC run on Burgers. Only submit after slurm/smoke_test.sbatch has passed and
# given you a real per-step timing to size --time from -- don't guess it upfront.
#   sbatch slurm/run_burgers_gem.sbatch
#
# Partition/GRES confirmed from real `sinfo -o "%P %G %l"` output (Aug 2026). `gpu24` is a 1-day
# time-limited slice of the main GPU pool (v100/a100/etc available there too) -- a reasonable
# starting guess for the default N=8/K=2000 config below. If the smoke test's 100-step timing
# extrapolates past ~20h for the full 2000 steps, switch to `--partition=gpu` instead (same GPU
# types, 14-day limit, just a longer queue) rather than raising --time past gpu24's 1-day cap.
#SBATCH --job-name=burgers_gem
#SBATCH --partition=gpu24
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00              # TODO: retune from the smoke test's 100-step timing
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
mkdir -p slurm/logs
source .venv/bin/activate

# Defaults match configs/burgers.yaml (batch_size=1, iterations=2000) -- override via
# `sbatch --export=N_PARTICLES=8,NUM_STEPS=1000 slurm/run_burgers_gem.sbatch` for an
# intermediate-scale run without editing this file.
N_PARTICLES="${N_PARTICLES:-8}"
NUM_STEPS="${NUM_STEPS:-2000}"

python -m scripts.generate_burgers_gem --config configs/burgers.yaml \
  --n-particles "$N_PARTICLES" --num-steps "$NUM_STEPS" \
  --out "slurm/logs/burgers_gem_N${N_PARTICLES}_K${NUM_STEPS}_${SLURM_JOB_ID}.npz"
