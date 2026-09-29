#!/bin/bash
# exp2b (Stage 1: beta x ratio likelihood sweep, beta=3000): SMC at
# r = omega/beta in {0, 1e-3, 3e-3, 1e-2} (omega = 0, 3, 9, 30), lambda_girs=1,
# rho_temp=rho_temp_init=1, K=2000, N=8.
#   sbatch slurm/logs2/smc_exp2b.sbatch      (or: bash slurm/logs2/submit_exp2.sh)
#SBATCH --job-name=exp2b
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

echo "=== $(date -Is) exp2_b3000_r0 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 --obs-weight 3000 --pde-weight 0 --rho-temp 1 --rho-temp-init 1 --lambda-girs 1 --run-id exp2_b3000_r0

echo "=== $(date -Is) exp2_b3000_r1em3 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 --obs-weight 3000 --pde-weight 3 --rho-temp 1 --rho-temp-init 1 --lambda-girs 1 --run-id exp2_b3000_r1em3

echo "=== $(date -Is) exp2_b3000_r3em3 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 --obs-weight 3000 --pde-weight 9 --rho-temp 1 --rho-temp-init 1 --lambda-girs 1 --run-id exp2_b3000_r3em3

echo "=== $(date -Is) exp2_b3000_r1em2 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 --obs-weight 3000 --pde-weight 30 --rho-temp 1 --rho-temp-init 1 --lambda-girs 1 --run-id exp2_b3000_r1em2

echo "=== $(date -Is) aggregate ==="
python3 slurm/logs2/aggregate_exp2.py

echo "=== $(date -Is) done ==="
