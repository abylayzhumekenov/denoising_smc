#!/bin/bash
# exp4 ODE K-sweep (baseline reference): Heun PF-ODE + flat two-phase guidance, seed 0, offset 0.
# zeta_obs/zeta_pde held at the released 320/100 for every K (beta-choice (a); dosage ~ K).
# K in {125 250 500 1000 2000 4000}. Run: sbatch slurm/logs4/ode_exp4.sbatch
#SBATCH --job-name=exp4_ode
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

for K in 125 250 500 1000 2000 4000; do
  echo "=== $(date -Is) exp4_ode_K${K} ==="
  python3 generate_pde.py --config slurm/logs4/ode_configs/burgers_k${K}.yaml
done

echo "=== $(date -Is) aggregate ==="
python3 slurm/logs4/aggregate_exp4.py

echo "=== $(date -Is) done ==="
