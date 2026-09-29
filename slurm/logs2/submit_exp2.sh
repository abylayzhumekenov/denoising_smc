#!/bin/bash
# Submit the Stage-1 (exp2) 2D beta x ratio likelihood sweep: three parallel jobs, one per BETA.
# (The historical 1-D beta scan lives in smc_exp2a.sbatch / smc_exp2b.sbatch.) Run from anywhere:
#   bash slurm/logs2/submit_exp2.sh
set -uo pipefail

cd "$(dirname "$0")/../.."   # repo root

BETA=1000  sbatch slurm/logs2/smc_exp2c.sbatch
BETA=3000  sbatch slurm/logs2/smc_exp2c.sbatch
BETA=10000 sbatch slurm/logs2/smc_exp2c.sbatch
