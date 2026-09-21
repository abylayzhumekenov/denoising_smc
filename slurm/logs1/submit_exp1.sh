#!/bin/bash
# Submit the three exp1 jobs (ODE+unweighted, PBS, Girsanov). Run from anywhere:
#   bash slurm/logs1/submit_exp1.sh
set -uo pipefail

cd "$(dirname "$0")/../.."   # repo root

sbatch slurm/logs1/smc_exp1a.sbatch
sbatch slurm/logs1/smc_exp1b.sbatch
sbatch slurm/logs1/smc_exp1c.sbatch
