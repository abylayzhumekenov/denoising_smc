#!/bin/bash
# exp3c (Stage 2: method suite, exp1-style) — SMC *Girs* (rho_temp=1.0, lambda_girs=1) at
# (beta, omega) in {(10000, 0), (10000, 10)} x rho_temp_init in {0,1}, K=2000, N=8.
# (Duplicates exp2_b10000_r0 / _r1em3 up to rho_temp_init; kept for a symmetric exp1-style table.)
#   sbatch slurm/logs3/smc_exp3c.sbatch      (or: bash slurm/logs3/submit_exp3.sh)
#SBATCH --job-name=exp3c
#SBATCH --chdir=/home/alvarem/denoising_smc
#SBATCH --partition=debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs3/%x_%j.out
#SBATCH --error=slurm/logs3/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miguelangel.alvarezballesteros@kaust.edu.sa

set -uo pipefail
mkdir -p slurm/logs3

module load cuda/11.8
module load python/3.11.0

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

CFG=configs/smc/burgers.yaml

echo "=== $(date -Is) exp3_girs_b1e4_r0_ri0 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 \
  --obs-weight 10000 --pde-weight 0 --rho-temp 1.0 --rho-temp-init 0 --lambda-girs 1 \
  --run-id exp3_girs_b1e4_r0_ri0

echo "=== $(date -Is) exp3_girs_b1e4_r0_ri1 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 \
  --obs-weight 10000 --pde-weight 0 --rho-temp 1.0 --rho-temp-init 1 --lambda-girs 1 \
  --run-id exp3_girs_b1e4_r0_ri1

echo "=== $(date -Is) exp3_girs_b1e4_r1em3_ri0 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 \
  --obs-weight 10000 --pde-weight 10 --rho-temp 1.0 --rho-temp-init 0 --lambda-girs 1 \
  --run-id exp3_girs_b1e4_r1em3_ri0

echo "=== $(date -Is) exp3_girs_b1e4_r1em3_ri1 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 \
  --obs-weight 10000 --pde-weight 10 --rho-temp 1.0 --rho-temp-init 1 --lambda-girs 1 \
  --run-id exp3_girs_b1e4_r1em3_ri1

echo "=== $(date -Is) aggregate ==="
python3 slurm/logs3/aggregate_exp3.py

echo "=== $(date -Is) done ==="
