#!/bin/bash
# Submit the three Stage-2 (exp3) method-suite jobs (unweighted / PBS / Girs). Run from anywhere:
#   bash slurm/logs3/submit_exp3.sh
set -uo pipefail

cd "$(dirname "$0")/../.."   # repo root

sbatch slurm/logs3/smc_exp3a.sbatch
sbatch slurm/logs3/smc_exp3b.sbatch
sbatch slurm/logs3/smc_exp3c.sbatch
