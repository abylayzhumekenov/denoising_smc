#!/bin/bash
# exp3a (Stage 2: method suite, part 1): at the Stage-1 chosen (BETA, OMEGA), the
# lambda_girs x rho_temp_init grid, K=2000, N=8.
#   BETA=100 OMEGA=0.02 sbatch slurm/logs3/smc_exp3a.sbatch
#   (or: bash slurm/logs3/submit_exp3.sh)
# Set BETA/OMEGA from the Stage-1 (exp2) table before submitting.
#SBATCH --job-name=exp3a
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
N_PARTICLES="${N_PARTICLES:-8}"
NUM_STEPS="${NUM_STEPS:-2000}"
BETA="${BETA:-100}"
OMEGA="${OMEGA:-0.02}"

# lambda_girs in {0,1} x rho_temp_init in {0,1}
for l in 0 1; do
  for ri in 0 1; do
    echo "=== $(date -Is) exp3_l${l}_ri${ri} ==="
    python3 generate_pde_smc.py --config $CFG --n-particles "$N_PARTICLES" --num-steps "$NUM_STEPS" \
      --obs-weight "$BETA" --pde-weight "$OMEGA" --rho-temp 1 --rho-temp-init "$ri" \
      --lambda-girs "$l" --run-id "exp3_l${l}_ri${ri}"
  done
done

echo "=== $(date -Is) aggregate ==="
python3 slurm/logs3/aggregate_exp3.py

echo "=== $(date -Is) done ==="
