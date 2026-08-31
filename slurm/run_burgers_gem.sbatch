#!/bin/bash
# Full-scale GEM+SMC run on Burgers. Only submit after slurm/smoke_test.sbatch has passed and
# given you a real per-step timing to size --time from -- don't guess it upfront.
#   sbatch slurm/run_burgers_gem.sbatch
#
# Partition/GRES/time retuned from the actual smoke-test timing (Aug 2026): 100 steps at N=4
# particles took ~33s on a V100 (vs ~20m46s for the same 100 steps on CPU -- ~38x speedup).
# Extrapolating roughly linearly in step count to the full K=2000 puts this at ~11 minutes for
# N=4, plausibly under an hour even allowing generously for N=8's extra per-step cost -- nowhere
# near the previous 20h guess. `gpu4` (4h cap) comfortably covers that with margin and should
# queue faster than the longer-cap partitions (gpu24/gpu) since fewer people reserve short slots.
# If a real run ever needs longer (e.g. a much larger N/K sweep), bump both --partition and
# --time together rather than raising --time past whatever partition's cap you're on.
#SBATCH --job-name=burgers_gem
#SBATCH --chdir=/home/alvarem/denoising_smc     # see smoke_test.sbatch's comment on this line
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00              # ~3-5x the extrapolated worst case, still well under gpu4's 4h cap
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --mail-type=ALL                        # TODO: BEGIN/END/FAIL/ALL, whichever you want
#SBATCH --mail-user=miguelangel.alvarezballesteros@kaust.edu.sa

set -euo pipefail
mkdir -p slurm/logs

# Must match setup_env.sh exactly -- see smoke_test.sbatch's comment on this same line for why
# this is required (Slurm jobs don't inherit login-node modules).
module load cuda/11.8
module load python/3.11.0

source .venv/bin/activate

# Defaults match configs/burgers.yaml (batch_size=1, iterations=2000) -- override via
# `sbatch --export=N_PARTICLES=8,NUM_STEPS=1000 slurm/run_burgers_gem.sbatch` for an
# intermediate-scale run without editing this file.
N_PARTICLES="${N_PARTICLES:-8}"
NUM_STEPS="${NUM_STEPS:-2000}"

python -m scripts.generate_burgers_gem --config configs/burgers.yaml \
  --n-particles "$N_PARTICLES" --num-steps "$NUM_STEPS" \
  --out "slurm/logs/burgers_gem_N${N_PARTICLES}_K${NUM_STEPS}_${SLURM_JOB_ID}.npz"
