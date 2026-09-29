#!/bin/bash
# Submit the two Stage-2 (exp3) method-suite jobs. Set BETA/OMEGA from the exp2 table first:
#   BETA=100 OMEGA=0.02 bash slurm/logs3/submit_exp3.sh
set -uo pipefail

cd "$(dirname "$0")/../.."   # repo root

export BETA="${BETA:-100}"
export OMEGA="${OMEGA:-0.02}"

sbatch slurm/logs3/smc_exp3a.sbatch
sbatch slurm/logs3/smc_exp3b.sbatch
