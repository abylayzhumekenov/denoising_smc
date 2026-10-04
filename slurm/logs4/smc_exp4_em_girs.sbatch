#!/bin/bash
# exp4 SMC: proposal=em, weighting=girs [lambda_girs=1 (full Girsanov)]; K in {125 250 500 1000 2000 4000}. beta=1e4, omega=0, N=8,
# seed 0, offset 0, sensors 5. ~80 min/job.
# Run: sbatch slurm/logs4/smc_exp4_em_girs.sbatch
#SBATCH --job-name=exp4_em_girs
#SBATCH --chdir=/home/alvarem/denoising_smc
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:50:00
#SBATCH --output=slurm/logs4/%x_%j.out
#SBATCH --error=slurm/logs4/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miguelangel.alvarezballesteros@kaust.edu.sa

set -uo pipefail
mkdir -p slurm/logs4

module load cuda/11.8
module load python/3.11.0

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

CFG=configs/smc/burgers.yaml

for K in 125 250 500 1000 2000 4000; do
  echo "=== $(date -Is) exp4_em_girs_K${K} ==="
  python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps $K \
    --obs-weight 10000 --pde-weight 0 --rho-temp 1.0 --rho-temp-init 1 --lambda-girs 1 \
    --proposal em --run-id exp4_em_girs_K${K}
done

echo "=== $(date -Is) aggregate ==="
python3 slurm/logs4/aggregate_exp4.py

echo "=== $(date -Is) done ==="
