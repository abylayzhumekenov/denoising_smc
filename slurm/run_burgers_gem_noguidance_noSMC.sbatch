#!/bin/bash
# No-guidance, no-SMC GEM: pure unconditional guided-Euler-Maruyama with the guidance term
# zeroed out entirely (--no-guidance) -- just the sigma**2-based score and the SDE's injected
# noise, no observation/PDE conditioning at all. Resampling also disabled (--resample-threshold
# 0), though with zero guidance the Girsanov increment should be ~trivial regardless.
#
# This is a reference floor, not a fix: it tells us how bad an outright UNCONDITIONAL sample
# is relative to ground truth (should be large -- an unconditional draw has no reason to match
# this particular Burgers realization) and, as a sanity check, whether the unconditional SDE
# machinery itself (schedule, score, noise) is behaving sanely on its own before any guidance
# is layered back on.
#
#   sbatch slurm/run_burgers_gem_noguidance_noSMC.sbatch
#SBATCH --job-name=burgers_gem_noguide
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
  --no-guidance --resample-threshold 0 \
  --out "slurm/logs/burgers_gem_noguidance_noSMC_N${N_PARTICLES}_K${NUM_STEPS}_${SLURM_JOB_ID}.npz"
