# exp3 — Stage 2: pick the weighting / tempering at the chosen `(β, ω)`

Goal: given `β*, ω*` from the Stage-1 table (`slurm/logs2/exp2_table.md`), decide
pbs vs corrected Girsanov, whether the PDE term helps, and the tempering / initial-weight settings.

## Submit (from the repo root on Ibex)
```bash
cd ~/denoising_smc
git pull
BETA=<beta*> OMEGA=<omega*> bash slurm/logs3/submit_exp3.sh
```
`BETA`/`OMEGA` default to `100` / `0.02` if not set — set them from `exp2_table.md`.

## What it runs (all K=2000, N=8)
- `smc_exp3a.sbatch`: `λ ∈ {0,1} × ρ_temp_init ∈ {0,1}` at `(β, ω)` — pbs vs Girsanov, initial weight on/off.
- `smc_exp3b.sbatch`: controls (`β=ω=0` no-guidance floor; `ω=0` obs-only; `β=0` pde-only) and
  `ρ_temp ∈ {0,1}`.

Run ids: `exp3_l<λ>_ri<ρ>`, `exp3_noguidance`, `exp3_obsonly`, `exp3_pdeonly`, `exp3_rt<ρ>`.

## Where results go / what to read
- `results/smc/burgers/exp3_*/`
- `slurm/logs3/exp3_table.md` — regenerated at the end of each job (and any time via
  `python3 slurm/logs3/aggregate_exp3.py`).

## Note
Both `exp2_table.md` and `exp3_table.md` include the ODE reference row (it has no run-id prefix).
Aggregator filters are env-overridable (`EXP2_K/EXP2_N/EXP2_PREFIX/EXP2_OUT`, same for `EXP3_*`).
