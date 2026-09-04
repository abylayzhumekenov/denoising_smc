#!/bin/bash
# Full-scale GEM+SMC run on Burgers. Only submit after slurm/smoke_test.sbatch has passed and
# given you a real per-step timing to size --time from -- don't guess it upfront.
#   sbatch slurm/run_burgers_gem.sbatch
#
# Partition/GRES/time retuned from the actual smoke-test timing (Aug 2026): 100 steps at N=4
# particles took ~33s on a V100 (vs ~20m46s for the same 100 steps on CPU -- ~38x speedup).
# Extrapolating roughly linearly in step count to the full K=2000 puts this at ~11 minutes for
# N=4, plausibly under an hour even allowing generously for N=8's extra per-step cost.
#
# `gpu4` was tried first but rejected with "not permitted" -- account pi-temponrf isn't on that
# partition's allow-list (Ibex partitions can have separate per-account/QOS access controls).
# `debug` is the one partition confirmed to actually work for this account (the smoke test ran
# there), and the estimated runtime fits comfortably inside its 2h cap, so using it here too as
# the reliable default. debug is nominally meant for quick interactive checks rather than
# production runs, but at this run's short expected duration it's a reasonable stopgap -- if you
# start running longer N/K sweeps later, run
#   sacctmgr show assoc user=alvarem format=Account,Partition,QOS%20 -p
# to see which of gpu/gpu24/gpu72/gpu_wide your account is actually authorized for, rather than
# guessing partition names again.
#SBATCH --job-name=burgers_gem
#SBATCH --chdir=/home/alvarem/denoising_smc     # see smoke_test.sbatch's comment on this line
#SBATCH --partition=debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00              # under debug's 2h cap, ~5-8x the extrapolated worst case
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

# Larger N_PARTICLES batches (e.g. N=16) push the per-step denoiser forward+backward close to
# the V100's 32G capacity; PyTorch's caching allocator can then fail a small allocation even
# though it's holding several times that much in fragmented, reserved-but-unallocated blocks
# (seen directly: a 51275999-style OOM reporting ~2GB "reserved but unallocated" right next to a
# failed 256MB request). expandable_segments lets the allocator grow an existing segment instead
# of demanding a fresh contiguous block, which resolves exactly this fragmentation pattern --
# see https://pytorch.org/docs/stable/notes/cuda.html#environment-variables. Harmless/no-op at
# the N=8 sizes that were already fitting comfortably.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Defaults match configs/burgers.yaml (batch_size=1, iterations=2000) -- override via
# `sbatch --export=N_PARTICLES=8,NUM_STEPS=1000 slurm/run_burgers_gem.sbatch` for an
# intermediate-scale run without editing this file.
N_PARTICLES="${N_PARTICLES:-8}"
NUM_STEPS="${NUM_STEPS:-2000}"

python -m scripts.generate_burgers_gem --config configs/burgers.yaml \
  --n-particles "$N_PARTICLES" --num-steps "$NUM_STEPS" \
  --out "slurm/logs/burgers_gem_N${N_PARTICLES}_K${NUM_STEPS}_${SLURM_JOB_ID}.npz"
