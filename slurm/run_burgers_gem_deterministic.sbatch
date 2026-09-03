#!/bin/bash
# Deterministic-drift GEM control: same sigma**2-based score and same delta-SCALED guidance as
# the real GEM+SMC run, but with the SDE's injected Brownian noise switched off (--no-noise, see
# smc/scripts_2/proposals/gem.py's inject_noise flag) AND resampling fully disabled
# (--resample-threshold 0, same trick as the earlier no-SMC control: ESS >= 1 always, so
# `ess < 0*N` is never true).
#
# This is the next cut after confirming (baseline_burgers_euler_k2000.sbatch) that dropping the
# Heun 2nd-order correction changes nothing (0.0521 vs Heun's 0.0521) -- so integrator order is
# ruled out. What's left is: the injected noise, or the delta-scaled guidance convention, or an
# interaction of both. This run isolates the guidance-scaling piece alone:
#   - lands back near Heun/Euler's ~0.052  -> the injected noise is what's causing the blowup
#   - still ~1-2+ like the noisy GEM runs  -> the delta-scaled guidance itself is miscalibrated,
#     independent of any stochasticity (points at rescaling zeta_obs/zeta_pde for GEM's convention)
#
# Same K=2000, N=8 as the existing GEM runs for direct comparability.
#   sbatch slurm/run_burgers_gem_deterministic.sbatch
#SBATCH --job-name=burgers_gem_det
#SBATCH --chdir=/home/alvarem/denoising_smc
#SBATCH --partition=debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miguelangel.alvarezballesteros@kaust.edu.sa

set -euo pipefail
mkdir -p slurm/logs

module load cuda/11.8
module load python/3.11.0

source .venv/bin/activate

N_PARTICLES="${N_PARTICLES:-8}"
NUM_STEPS="${NUM_STEPS:-2000}"

python -m scripts.generate_burgers_gem --config configs/burgers.yaml \
  --n-particles "$N_PARTICLES" --num-steps "$NUM_STEPS" \
  --no-noise --resample-threshold 0 \
  --out "slurm/logs/burgers_gem_deterministic_N${N_PARTICLES}_K${NUM_STEPS}_${SLURM_JOB_ID}.npz"
