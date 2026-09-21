#!/bin/bash
# exp1a: ODE baseline + SMC *unweighted* (multiple-try) variants. Burgers, K=2000, N=8.
#   sbatch slurm/logs1/smc_exp1a.sbatch      (or: bash slurm/logs1/submit_exp1.sh)
#SBATCH --job-name=exp1a
#SBATCH --chdir=/home/alvarem/denoising_smc
#SBATCH --partition=debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:50:00
#SBATCH --output=slurm/logs1/%x_%j.out
#SBATCH --error=slurm/logs1/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miguelangel.alvarezballesteros@kaust.edu.sa

set -uo pipefail
mkdir -p slurm/logs1

module load cuda/11.8
module load python/3.11.0
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

CFG=configs/smc/burgers.yaml

echo "=== $(date -Is) ode ==="
python3 generate_pde.py --config configs/burgers.yaml

echo "=== $(date -Is) smc_unw_b1w1_ri0 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 --obs-weight 1 --pde-weight 1 --rho-temp 0.0 --rho-temp-init 0 --lambda-girs 1 --run-id smc_unw_b1w1_ri0

echo "=== $(date -Is) smc_unw_b1w1_ri1 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 --obs-weight 1 --pde-weight 1 --rho-temp 0.0 --rho-temp-init 1 --lambda-girs 1 --run-id smc_unw_b1w1_ri1

echo "=== $(date -Is) smc_unw_b10w10_ri0 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 --obs-weight 10 --pde-weight 10 --rho-temp 0.0 --rho-temp-init 0 --lambda-girs 1 --run-id smc_unw_b10w10_ri0

echo "=== $(date -Is) smc_unw_b10w10_ri1 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 --obs-weight 10 --pde-weight 10 --rho-temp 0.0 --rho-temp-init 1 --lambda-girs 1 --run-id smc_unw_b10w10_ri1

echo "=== $(date -Is) aggregate ==="
python3 slurm/logs1/aggregate_exp1.py

echo "=== $(date -Is) done ==="
