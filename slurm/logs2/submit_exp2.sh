#!/bin/bash
# Submit the two Stage-1 (exp2) beta-calibration jobs. Run from anywhere:
#   bash slurm/logs2/submit_exp2.sh
set -uo pipefail

cd "$(dirname "$0")/../.."   # repo root

sbatch slurm/logs2/smc_exp2a.sbatch
sbatch slurm/logs2/smc_exp2b.sbatch
