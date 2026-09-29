#!/bin/bash
# exp2a (Stage 1: beta calibration, part 1): ODE baseline + SMC beta in {1, 10},
# omega = 2e-4 * beta (the cheap balance calibration), lambda_girs=1, rho_temp=rho_temp_init=1,
# K=2000, N=8. Goal: locate the overall guidance scale (beta) at large K.
#   sbatch slurm/logs2/smc_exp2a.sbatch      (or: bash slurm/logs2/submit_exp2.sh)
#SBATCH --job-name=exp2a
#SBATCH --chdir=/home/alvarem/denoising_smc
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:50:00
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

echo "=== $(date -Is) ode ==="
python3 generate_pde.py --config configs/burgers.yaml

# beta in {1, 10}, omega = 2e-4 * beta
for b in 1 10; do
  w=$(python3 -c "print($b * 2e-4)")
  echo "=== $(date -Is) exp2_b${b}_l1 ==="
  python3 generate_pde_smc.py --config $CFG --n-particles "$N_PARTICLES" --num-steps "$NUM_STEPS" \
    --obs-weight "$b" --pde-weight "$w" --rho-temp 1 --rho-temp-init 1 --lambda-girs 1 \
    --run-id "exp2_b${b}_l1"
done

echo "=== $(date -Is) aggregate ==="
python3 slurm/logs2/aggregate_exp2.py

echo "=== $(date -Is) done ==="
