# exp2 — Stage 1: find the guidance scale `β`

Goal: pick the **overall guidance scale** `β` at full scale (K=2000, N=8). The obs-vs-PDE
*balance* is fixed to the cheap local calibration, `ω = 2e-4 · β`.

## Submit (from the repo root on Ibex)
```bash
cd ~/denoising_smc        # your Ibex checkout
git pull
bash slurm/logs2/submit_exp2.sh
```
Requires the venv from `slurm/setup_env.sh`, plus the usual `pretrained-models/` and `data/`.

## What it runs
- `smc_exp2a.sbatch`: ODE baseline + SMC at `β ∈ {1, 10}`
- `smc_exp2b.sbatch`: SMC at `β ∈ {100, 1000}`

All SMC runs: `ω = 2e-4·β`, `λ=1`, `ρ_temp = ρ_temp_init = 1`, K=2000, N=8.
Run ids are `exp2_b<b>_l1`; the ODE baseline gets an auto-generated run id.

## Where results go / what to read
- `results/smc/burgers/exp2_b<b>_l1/`, `results/ode/burgers/<auto>/` (`result.npz`, `config.yaml`, `metrics.json`)
- `slurm/logs2/exp2_table.md` — regenerated at the end of each job (and any time via
  `python3 slurm/logs2/aggregate_exp2.py`).

Pick `β*` = the run with the best `rel_err` and non-collapsed ESS, then use `ω* = 2e-4·β*`
in Stage 2 (`slurm/logs3/`).
