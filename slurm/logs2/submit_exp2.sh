#!/bin/bash
# Submit the Stage-1 (exp2) beta x ratio likelihood sweep (beta in {1000, 3000, 10000}).
# Run from anywhere:
#   bash slurm/logs2/submit_exp2.sh
set -uo pipefail

cd "$(dirname "$0")/../.."   # repo root

sbatch slurm/logs2/smc_exp2a.sbatch
sbatch slurm/logs2/smc_exp2b.sbatch
sbatch slurm/logs2/smc_exp2c.sbatch
