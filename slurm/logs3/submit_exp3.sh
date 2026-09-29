#!/bin/bash
# Submit the Stage-2 (exp3) method-suite jobs. Set BETA/OMEGA (literal) in the sbatch scripts
# from slurm/logs2/exp2_table.md first. Run from anywhere:
#   bash slurm/logs3/submit_exp3.sh
set -uo pipefail

cd "$(dirname "$0")/../.."   # repo root

sbatch slurm/logs3/smc_exp3a.sbatch
sbatch slurm/logs3/smc_exp3b.sbatch
