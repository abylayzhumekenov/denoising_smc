#!/bin/bash
# exp3a (Stage 2: method suite, exp1-style) — ODE baseline + SMC *unweighted*
# (rho_temp=0, lambda_girs=1, but rho_temp=0 makes the weights uniform / lambda moot), at the
# exp2-best scale: (beta, omega) in {(10000, 0), (10000, 10)} x rho_temp_init in {0,1},
# K=2000, N=8.
#   sbatch slurm/logs3/smc_exp3a.sbatch      (or: bash slurm/logs3/submit_exp3.sh)
#SBATCH --job-name=exp3a
#SBATCH --chdir=/home/alvarem/denoising_smc
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

echo "=== $(date -Is) ode ==="
python3 generate_pde.py --config configs/burgers.yaml

echo "=== $(date -Is) exp3_unw_b1e4_r0_ri0 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 \
  --obs-weight 10000 --pde-weight 0 --rho-temp 0.0 --rho-temp-init 0 --lambda-girs 1 \
  --run-id exp3_unw_b1e4_r0_ri0

echo "=== $(date -Is) exp3_unw_b1e4_r0_ri1 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 \
  --obs-weight 10000 --pde-weight 0 --rho-temp 0.0 --rho-temp-init 1 --lambda-girs 1 \
  --run-id exp3_unw_b1e4_r0_ri1

echo "=== $(date -Is) exp3_unw_b1e4_r1em3_ri0 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 \
  --obs-weight 10000 --pde-weight 10 --rho-temp 0.0 --rho-temp-init 0 --lambda-girs 1 \
  --run-id exp3_unw_b1e4_r1em3_ri0

echo "=== $(date -Is) exp3_unw_b1e4_r1em3_ri1 ==="
python3 generate_pde_smc.py --config $CFG --n-particles 8 --num-steps 2000 \
  --obs-weight 10000 --pde-weight 10 --rho-temp 0.0 --rho-temp-init 1 --lambda-girs 1 \
  --run-id exp3_unw_b1e4_r1em3_ri1

echo "=== $(date -Is) aggregate ==="
python3 slurm/logs3/aggregate_exp3.py

echo "=== $(date -Is) done ==="
