# Vision and Decision Log — DiffusionPDE + Denoising SMC

_Status snapshot: 2026-09-15. Repo: `denoising_smc` — the official DiffusionPDE reference
implementation (Huang et al., NeurIPS 2024) plus a denoising / Sequential Monte Carlo (SMC)
research extension._

This document is the single source of truth for the project's intent, structure, and the
decisions taken along the way. It is agent-neutral: any LLM or human working here should read it
before making structural changes. `AGENTS.md` is only a pointer to this file.

---

## 1. Purpose and end goal

The research goal is to make **guided diffusion sampling for PDE inverse problems statistically
sound** by treating the guided proposal as an importance-sampling distribution on path space and
correcting it with a Girsanov / Doob/Feynman–Kac weight. Concretely:

- Develop and validate the Girsanov-corrected SMC weight (`docs/note_1.pdf`) and the equivalent
  potential/Doob representation (`docs/note_2.pdf`).
- Validate first on a closed-form Gaussian-mixture toy model, then on a real pretrained PDE
  diffusion model (Burgers as the first real case).
- Compare against the existing deterministic guided probability-flow-ODE baseline of Huang et al.
- Produce paper-facing results: correctness of the weight, effective sample size / variance
  behaviour, and reconstruction/uncertainty quality.

End state: a clean, config-driven SMC implementation under `smc/`, runnable through a root
dispatcher, with reproducible structured outputs under `results/`, and a frozen archive of the
earlier exploratory code.

---

## 2. Two methods, two targets (why we do not compare SMC to the ODE on one axis)

The repository contains two genuinely different objects:

- **SMC (this work).** Targets the posterior `p(x0 | y) ∝ p(y | x0) p(x0)`. The guided proposal is
  a reverse-time SDE whose drift is augmented by the twist gradient; the path-space weight makes the
  particle population consistent for the posterior (up to discretization and a terminal correction).
  The twist affects only weight variance, never validity.

- **Huang et al. guided ODE baseline (`scripts/generate_*.py`).** A deterministic probability-flow
  ODE (Heun) with a post-hoc, constant-coefficient subtraction of the loss gradient, plus a two-phase
  guidance schedule. It defines no probabilistic target; read as a guided PF-ODE, its implicit target
  is a schedule-dependent, ever-colder reconstruction that concentrates on loss minimizers, not the
  Bayesian posterior.

`docs/note_4.pdf` formalises this: the baseline's flat guidance is equivalent to a PF-ODE with a
schedule-tailored twist whose terminal inverse temperature diverges, and relative-L2-to-ground-truth
is a reconstruction metric rather than a posterior metric. **Do not present SMC and the ODE baseline
as two discretizations of the same target.**

---

## 3. Structural principles

These principles generate the specific decisions in §5.

- **P1 — One shared project package.** All cross-cutting *project-owned* code lives in `common/`
  (results I/O now; config loading, seeding, logging later). The ODE baseline and the SMC
  implementation both import `common`; neither imports the other.
- **P2 — Vendored code stays vendored.** `dnnlib/`, `torch_utils/`, `training/`, and `train.py`
  remain at their upstream EDM paths. They are never merged, renamed, or relocated.
- **P3 — Baseline independent of SMC.** `scripts/` must not import from `smc/`.
- **P4 — `smc_archive/` is frozen.** A historical record; do not edit, extend, or import from it.
- **P5 — Location-first ignore + one output convention.** `.gitignore` is location-first with a fixed
  tooling/binary pattern set; all runs write to `results/{ode,smc}/<pde>/<run_id>/`.

---

## 4. Repository map

| Path | Role |
|---|---|
| `dnnlib/`, `torch_utils/`, `training/`, `train.py` | **Vendored EDM training stack — do not restructure.** Device auto-detection added to `torch_utils.misc`; otherwise upstream. |
| `generate_pde.py`, `scripts/generate_*.py`, `configs/*.yaml` | **Upstream ODE baseline only.** May use `common` (shared infra) but must not import from `smc/`. |
| `scripts/generate_*.py` | Six PDE monoliths (Burgers, Darcy, Poisson, Helmholtz, NS bounded/non-bounded). |
| `common/` | **(to be created)** Shared project-owned infrastructure (results I/O, config, seeding, logging). Imported by both baseline and SMC. |
| `smc/` | **New SMC implementation** (currently empty; reserved with `.gitkeep`). |
| `smc_archive/scripts_1` | Frozen: toy (closed-form) SMC validation code. |
| `smc_archive/scripts_2` | Frozen: real-model SMC (Burgers), GEM proposal, weightings, checks, old runner. |
| `smc_archive/scripts_3` | Frozen: non-SMC newcomers (`sample_prior.py`, `generate_darcy_local.py`). |
| `configs/smc/` | **(to be created)** SMC run configs, symmetric with `configs/`. |
| `generate_pde_smc.py` | **(to be created)** SMC dispatcher, symmetric with `generate_pde.py`. |
| `results/{ode,smc}/<pde>/<run_id>/` | Structured run outputs (resolved config, metrics, logs). |
| `docs/note_1.pdf`, `docs/note_2.pdf`, `docs/note_4/` | Theory and comparison notes. |
| `literature/` | Third-party papers and survey. |

**Vendored-code constraint.** The pretrained `.pkl` checkpoints reference `torch_utils.persistence`
(the `_reconstruct_persistent_obj` unpickle anchor) and `training.dataset.ImageFolderDataset` by
exact module path. Relocating or renaming those modules breaks `pickle.load(...)['ema']`, i.e.
loading the pretrained models. This is why P2 forbids moving the vendored stack (and why any
`vendor/`-style reorg would require shims — not worth it).

**Hard rules**

- Shared cross-cutting code goes in `common/`; the baseline and SMC implementations do not import
  each other.
- The ODE baseline stays upstream-faithful except for the intentional device auto-detection and the
  sanctioned output-directory change (D13).
- Vendored EDM modules (`dnnlib/`, `torch_utils/`, `training/`, `train.py`) are never moved/renamed.
- `smc_archive/` is frozen: do not edit, extend, or import from it. It is a historical record.
- New SMC code goes in `smc/`; new SMC configs in `configs/smc/`; runs write to `results/`.

---

## 5. Decision log

Each entry: decision — rationale — consequences. (Dated as adopted.)

- **D1 (2026-09-15) — Keep device auto-detection.** All upstream scripts read `device: 'auto'` and
  use `auto_device()` from `torch_utils.misc` (CUDA → CPU; MPS opt-in for float32-only callers).
  *Rationale:* runnability on CPU/GPU without editing configs. *Consequence:* a small, additive,
  backwards-compatible cross-cutting change in upstream files; it is the only intended non-cosmetic
  difference from upstream.

- **D2 (2026-09-15) — Baseline independent of SMC.** `scripts/generate_*.py` must not import from
  `smc/`; `scripts/generate_burgers.py` keeps its own `random_sensor`/`get_burger_loss`, and the SMC
  stack keeps a batched duplicate. *Rationale:* keeps the upstream baseline reproducible and
  comparable; avoids coupling. *Consequence:* deliberate duplication between baseline and SMC.

- **D3 (2026-09-15) — Freeze legacy SMC in `smc_archive/`; new implementation in `smc/`.**
  *Rationale:* the exploratory code (toy + first real model) is messy and will be reimplemented;
  preserve it as a record without letting it constrain the new design. *Consequence:* no new work
  touches `smc_archive/`; `smc/` is reserved for the clean implementation.

- **D4 (2026-09-15) — Monolith per PDE, no adapter/protocol abstraction.** New SMC uses one
  self-contained module per PDE (like the ODE baseline), sharing small algorithm helpers at most.
  The multi-PDE adapter/protocol design was considered and rejected for now. *Rationale:* we will
  not touch most PDEs; an abstraction is premature. *Consequence:* adding a PDE later means a new
  monolith, not a new adapter.

- **D5 (2026-09-15) — Root dispatcher + separate configs.** `generate_pde_smc.py` mirrors
  `generate_pde.py`; SMC configs live in `configs/smc/`. *Rationale:* symmetry with the baseline
  workflow and reproducibility. *Consequence:* `scripts/` stays upstream-only; the SMC dispatcher is
  the only SMC entry point outside `smc/`.

- **D6 (2026-09-15) — SMC config schema.** Copy shared `data`/`test`/`generate` fields, plus an
  `smc:` block. Guidance/likelihood parameters are named as **likelihood quantities**, not the ODE's
  `zeta_*`; the tempering key must not collide with the ODE schedule exponent `rho`. Single-run
  configs now; sweeps deferred. *Rationale:* the ODE `zeta_*` are flat-guidance tuning constants,
  not log-likelihood parameters. *Consequence:* field names are provisional; the author will finalize
  them after a working SMC exists. A sweep spec, if added later, is one file expanded at runtime —
  never one config per sweep cell.

- **D7 (2026-09-15) — Structured results.** One run → one directory
  `results/{ode,smc}/<pde>/<run_id>/{result.*, config.yaml, metrics.json, run.log}`, with
  `run_id = YYYYMMDD-HHMMSS_<short config hash>` (overridable). *Rationale:* reproducibility and
  sweep aggregation; ODE and SMC should be consistent. *Consequence:* implemented through the shared
  `common/` writer (D11); the baseline side is D13.

- **D8 (2026-09-15) — Validation is a separate tier.** Toy validations and correctness checks
  (e.g. GEM↔TDS kernel-ratio identity) are not reachable from the run dispatcher. *Rationale:* keep
  the run path simple; checks are diagnostics. *Consequence:* a `smc/checks/` (or equivalent) tier
  will be created with the new implementation.

- **D9 (2026-09-15) — `.gitignore` policy: location-first.** Ignore a fixed set of tooling/OS junk
  and never-source binary extensions (match anywhere); ignore output locations via anchored rules;
  keep empty output dirs present using self-ignoring `.gitignore` placeholders (`*` / `!.gitignore`)
  in `data/`, `pretrained-models/`, `results/`, `slurm/logs/`. No global media bans and no
  per-file exception block. *Rationale:* the rule count stays constant as the repo grows and tracked
  figures/PDFs need no `!` exceptions. *Consequence:* a generated file outside a named output dir is
  not auto-ignored; name its dir or add one anchored rule.

- **D10 (2026-09-15) — Documentation policy.** `README.md` remains upstream-oriented; `AGENTS.md`
  is a one-line pointer to this file; stale operational docs (e.g. `slurm/README.md`) are deferred
  and may be deleted rather than updated. *Rationale:* avoid maintaining drifting docs. *Consequence:*
  this file carries all project context.

- **D11 (2026-09-15) — Single shared project package `common/`.** All cross-cutting project-owned
  code lives in `common/` (run-id/run-dir creation, `config.yaml`/`metrics.json` writing; later
  config loading, seeding, logging). *Rationale:* avoids duplicating run/results plumbing across the
  baseline and SMC and gives shared code one home without coupling the two to each other.
  *Consequence:* both baseline and SMC depend on `common`; neither imports the other. Shared project
  code must not be placed in the vendored `dnnlib/`/`torch_utils/`.

- **D12 (2026-09-15) — Vendored EDM stack stays at upstream paths.** `dnnlib/`, `torch_utils/`,
  `training/`, and `train.py` are never merged, renamed, or relocated. *Rationale:* the pretrained
  `.pkl` checkpoints reference `torch_utils.persistence` (`_reconstruct_persistent_obj`) and
  `training.dataset.ImageFolderDataset` by exact module path, so moving them breaks model loading;
  plus ~43 import sites and upstream diffability. *Consequence:* no `vendor/`-style reorg; vendored
  modules may be annotated as vendored but not moved.

- **D13 (2026-09-15) — Baseline output directory (decided; implementation pending).** The ODE
  baseline writes to `results/ode/<pde>/<run_id>/{result.npy|result.mat, config.yaml, metrics.json}`
  (metric keys per PDE: Burgers `relative_error`; Darcy `error_rate_a` + `relative_error_u`; others
  `relative_error_a` + `relative_error_u`). Output root and `run_id` are read from the config with
  `.get` defaults (`generate.out_dir` → `results/ode`; `generate.run_id` → auto), so no config edits
  are required, and the old hardcoded CWD filenames are dropped. *Rationale:* consistency with D7.
  *Consequence:* a deliberate, sanctioned deviation from upstream in the six baseline scripts
  (alongside device auto-detection), implemented through the `common/` writer (D11).

---

## 6. Findings that inform the design

- **Target mismatch (note_4).** The ODE baseline's effective target is a schedule-dependent cold
  reconstruction, not the posterior; its guidance omits the PF-ODE coefficient and (in the flat
  placement) the step factor. Relative-L2 error is a reconstruction metric.
- **Flat guidance breaks the weight.** The closed-form Girsanov increment
  `C_k = -bᵀz - ½δ‖b‖²` is the exact Gaussian kernel ratio only when guidance enters the drift
  (step-scaled). A flat state update yields a different ratio, so flat guidance + Girsanov is
  out-of-theory.
- **The PDE residual is not a likelihood.** The current twist uses unsquared residual norms, so the
  "target" is a tempered surrogate unless a genuine likelihood and terminal correction are specified.
- **Millard et al. (2026, `literature/arXiv-2601.23262v2`).** Using the same pretrained models,
  they compare methods by reconstruction (relative L2), retune three likelihood weights and a
  tempering exponent, and use the SOSaG (jitter + 2nd-order) proposal with pseudo-bootstrap (pBS)
  weighting. Their own table shows TDS/Girsanov is the worst SMC variant while pBS is best; higher
  tempering improves reconstruction. This is protocol/tuning alignment, not a shared target.
- **Doob / V_τ + Hutchinson deprioritized.** The Hessian-trace term is small and noisy and costs
  `O(probes)` per particle step; Girsanov (Hessian-free) is the active route.

---

## 7. Roadmap and open items

Ordered roughly:

1. **Create `common/`** with the shared results writer (run-id, run dir, `config.yaml`/`metrics.json`).
2. **Baseline output directory (D13)** — route the six ODE scripts' outputs into
   `results/ode/<pde>/<run_id>/` via `common/`.
3. **Implement the new Burgers SMC in `smc/`** — monolith + small helpers, wired to
   `generate_pde_smc.py` and `configs/smc/burgers.yaml`, writing to `results/smc/`.
   **Milestone:** runs on CPU at a small K and writes a structured run directory.
4. **Finalize the `smc:` config fields** (likelihood parameter names, tempering key, run-id/out
   options) — author to set after a working SMC.
5. **Decide and add new checks** under the validation tier (start with the GEM↔TDS identity).
6. **Sweep design** (one spec expanded at runtime) — deferred.
7. **Modeling/algorithm follow-ups:** proper likelihood (e.g. squared-residual Gaussian form),
   flat-guidance kernel ratio or removal, terminal correction, SOSaG proposal, tempering sweeps.
8. **New SMC slurm script** (`slurm/run_smc*.sbatch`) parameterized by config.
9. **Stale docs:** delete/replace `slurm/README.md`; keep `README.md` upstream-oriented.

Status snapshot (git): `00ee0e1` archive restructure · `d3a00bc` note_4 · `b29ca3e` note_4 `.bbl` ·
`1ab2d8e` vision doc.

---

## 8. Conventions and entry points

**Entry points**

| Command | Purpose |
|---|---|
| `torchrun --standalone --nproc_per_node=N train.py --outdir=DIR --data=PATH --cond=0 --arch=ddpmpp --batch=60 --batch-gpu=20 --duration=20 --ema=0.05` | Train the diffusion model (EDM-style) |
| `python3 generate_pde.py --config configs/<pde>.yaml` | Solve a PDE with the upstream guided ODE baseline |
| `python3 generate_pde_smc.py --config configs/smc/<pde>.yaml` | **(to be built)** Solve a PDE with the new SMC sampler |
| `python3 merge_data.py` | Merge raw `.mat` → scaled `.npy` for training |

**Conventions / gotchas**

- All PDE guidance uses `torch.float64`.
- Two-phase guidance in the baseline: observation gradients only for the first ~80% of steps, then a
  10×-reduced observation weight plus the PDE-residual term.
- Data live in `(-1, 1)`; inverse-transform before PDE/observation losses. Per-PDE scale factors
  (Darcy: `a=(a+1.5)/0.2`, `u=(u+0.9)/115`; Burgers: `x*1.415`).
- Pretrained models are `.pkl` pickles loaded via `pickle.load(f)['ema']` (~208 MB; git-ignored).
- EDM ODE schedule: Heun 2nd order,
  `sigma_t = (sigma_max^(1/rho) + t/(N-1) (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho`.
- `--cond=0` in training means the model is unconditional.
- Sign convention: `d/dτ = −d/d(sigma_t)`.
- No tests, formatter, or linter are configured.
- Device: configs use `device: 'auto'`; helpers use `auto_device()` from `torch_utils.misc`.
  Training requires GPU / `torchrun`; inference runs on CPU.
