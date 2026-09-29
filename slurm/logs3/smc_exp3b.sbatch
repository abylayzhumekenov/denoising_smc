#!/bin/bash
# exp3b (Stage 2: method suite, part 2): controls at (BETA, OMEGA) plus the rho_temp sweep.
#   sbatch slurm/logs3/smc_exp3b.sbatch      (or: bash slurm/logs3/submit_exp3.sh)
# Controls: no-guidance (beta=omega=0), observation-only (omega=0), PDE-only (beta=0).
# Tempering: rho_temp in {0,1} (lambda=1, rho_temp_init=1).
# Set BETA/OMEGA below (literal) from slurm/logs2/exp2_table.md before submitting.
#SBATCH --job-name=exp3b
#SBATCH --chdir=/home/alvarem/denoising_smc
#SBATCH --partition=debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:50:00
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

echo "=== $(date -Is) exp3_noguidance ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 \
  --obs-weight 0 --pde-weight 0 --rho-temp 1 --rho-temp-init 1 --lambda-girs 1 \
  --run-id exp3_noguidance

echo "=== $(date -Is) exp3_obsonly ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 \
  --obs-weight 100 --pde-weight 0 --rho-temp 1 --rho-temp-init 1 --lambda-girs 1 \
  --run-id exp3_obsonly

echo "=== $(date -Is) exp3_pdeonly ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 \
  --obs-weight 0 --pde-weight 0.02 --rho-temp 1 --rho-temp-init 1 --lambda-girs 1 \
  --run-id exp3_pdeonly

for rt in 0 1; do
  echo "=== $(date -Is) exp3_rt${rt} ==="
  python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 \
    --obs-weight 100 --pde-weight 0.02 --rho-temp "$rt" --rho-temp-init 1 --lambda-girs 1 \
    --run-id "exp3_rt${rt}"
done

echo "=== $(date -Is) aggregate ==="
python3 slurm/logs3/aggregate_exp3.py

echo "=== $(date -Is) done ==="
