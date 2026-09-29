#!/bin/bash
# exp2b (Stage 1: beta calibration, part 2): SMC beta in {100, 1000}, omega = 2e-4 * beta,
# lambda_girs=1, rho_temp=rho_temp_init=1, K=2000, N=8.
#   sbatch slurm/logs2/smc_exp2b.sbatch      (or: bash slurm/logs2/submit_exp2.sh)
#SBATCH --job-name=exp2b
#SBATCH --chdir=/home/alvarem/denoising_smc
#SBATCH --partition=debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
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

# beta in {100, 1000}, omega = 2e-4 * beta
for b in 100 1000; do
  w=$(python3 -c "print($b * 2e-4)")
  echo "=== $(date -Is) exp2_b${b}_l1 ==="
  python3 generate_pde_smc.py --config $CFG --n-particles "$N_PARTICLES" --num-steps "$NUM_STEPS" \
    --obs-weight "$b" --pde-weight "$w" --rho-temp 1 --rho-temp-init 1 --lambda-girs 1 \
    --run-id "exp2_b${b}_l1"
done

echo "=== $(date -Is) aggregate ==="
python3 slurm/logs2/aggregate_exp2.py

echo "=== $(date -Is) done ==="
