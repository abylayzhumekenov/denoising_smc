#!/bin/bash
# Submit the exp4 campaign: 1 ODE K-sweep + 3 EM + 6 Heun (K-split) = 10 jobs.
# All limits 01:50:00; explicit CLI flags only (no env vars).
# Run from anywhere:  bash slurm/logs4/submit_exp4.sh
set -uo pipefail

cd "$(dirname "$0")/../.."   # repo root

sbatch slurm/logs4/ode_exp4.sbatch
sbatch slurm/logs4/smc_exp4_em_pbs.sbatch
sbatch slurm/logs4/smc_exp4_em_girs.sbatch
sbatch slurm/logs4/smc_exp4_em_unw.sbatch
sbatch slurm/logs4/smc_exp4_heun_pbs_a.sbatch
sbatch slurm/logs4/smc_exp4_heun_pbs_b.sbatch
sbatch slurm/logs4/smc_exp4_heun_girs_a.sbatch
sbatch slurm/logs4/smc_exp4_heun_girs_b.sbatch
sbatch slurm/logs4/smc_exp4_heun_unw_a.sbatch
sbatch slurm/logs4/smc_exp4_heun_unw_b.sbatch
