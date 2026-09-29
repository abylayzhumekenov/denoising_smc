#!/bin/bash
# exp2c (Stage 1: 2D beta x ratio likelihood sweep): one job per BETA, looping the
# PDE/obs ratio r = omega/beta in {0, 1e-3, 3e-3, 1e-2}. lambda_girs=1,
# rho_temp=rho_temp_init=1, K=2000, N=8. The ODE baseline and the final aggregation
# run only in the BETA=1000 job (so the three submitted jobs do not race on the table).
#   BETA=1000 sbatch slurm/logs2/smc_exp2c.sbatch      (or: bash slurm/logs2/submit_exp2.sh)
#SBATCH --job-name=exp2c
#SBATCH --chdir=/home/alvarem/denoising_smc
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs2/%x_%j.out
#SBATCH --error=slurm/logs2/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miguelangel.alvarezballesteros@kaust.edu.sa

set -uo pipefail
mkdir -p slurm/logs2

module load cuda/11.8
module load python/3.11.0

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

CFG=configs/smc/burgers.yaml
N_PARTICLES="${N_PARTICLES:-8}"
NUM_STEPS="${NUM_STEPS:-2000}"
BETA="${BETA:-1000}"

# r = omega/beta, with a filesystem-safe tag per ratio.
RATIOS=(0 1e-3 3e-3 1e-2)
TAGS=(r0 r1em3 r3em3 r1em2)

if [ "$BETA" = "1000" ]; then
  echo "=== $(date -Is) ode ==="
  python3 generate_pde.py --config configs/burgers.yaml
fi

for i in 0 1 2 3; do
  r="${RATIOS[$i]}"
  tag="${TAGS[$i]}"
  w=$(python3 -c "print($BETA * $r)")
  echo "=== $(date -Is) exp2_b${BETA}_${tag} (omega=${w}) ==="
  python3 generate_pde_smc.py --config $CFG --n-particles "$N_PARTICLES" --num-steps "$NUM_STEPS" \
    --obs-weight "$BETA" --pde-weight "$w" --rho-temp 1 --rho-temp-init 1 --lambda-girs 1 \
    --run-id "exp2_b${BETA}_${tag}"
done

if [ "$BETA" = "1000" ]; then
  echo "=== $(date -Is) aggregate ==="
  python3 slurm/logs2/aggregate_exp2.py
fi

echo "=== $(date -Is) done ==="
