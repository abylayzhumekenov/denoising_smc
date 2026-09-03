#!/bin/bash
# Flat-guidance, no-SMC control: keeps GEM's SDE noise and sigma**2 score exactly as-is, but adds
# guidance the way the Heun baseline does -- flat, unscaled by delta (--flat-guidance).
# Resampling disabled (--resample-threshold 0); read off per-particle raw errors the same way as
# the earlier no-SMC control, since the Girsanov weight formula assumes the delta-scaled
# convention and isn't meaningful here.
#
# x_next = x_cur + delta*score + guidance_grad + z   (guidance term mirrors Heun's post-hoc,
# fixed-size nudge; only the unconditional drift+noise part is delta-scaled)
#
# Distinguishes: does reusing the baseline's flat-guidance convention (with the SAME
# zeta_obs/zeta_pde it was tuned for) recover ~Heun-level accuracy inside the noisy SDE?
#   - yes -> confirms the delta-scaling of guidance (not the noise) is the dominant cause
#   - no  -> guidance scaling isn't the (whole) story; noise or something else still implicated
#
#   sbatch slurm/run_burgers_gem_flatguidance_noSMC.sbatch
#SBATCH --job-name=burgers_gem_flatg
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
  --flat-guidance --resample-threshold 0 \
  --out "slurm/logs/burgers_gem_flatguidance_noSMC_N${N_PARTICLES}_K${NUM_STEPS}_${SLURM_JOB_ID}.npz"
