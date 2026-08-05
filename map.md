# Repository Map: denoising_smc

## 1. What the Original Repo Does

**DiffusionPDE: Generative PDE-Solving Under Partial Observation** (NeurIPS 2024)

This repository implements the paper *"DiffusionPDE: Generative PDE-Solving Under Partial Observation"* by Jiahe Huang, Guandao Yang, Zichen Wang, and Jeong Joon Park (University of Michigan / Stanford University).

The core idea: Use **denoising diffusion probabilistic models** to solve **Partial Differential Equations (PDEs)** given sparse/partial observations. It covers:

- **Forward problems**: Given sparse observations of the PDE coefficient (e.g., permeability field), recover the full solution field.
- **Inverse problems**: Given sparse observations of the solution field, recover the PDE coefficient.
- **Both-spaces recovery**: Simultaneously recover both coefficient and solution fields from observations on both sides.
- **Time-dependent recovery**: For time-dependent PDEs (e.g., Burgers', Navier-Stokes), recover the full solution throughout a time interval from sparse sensor measurements.

The method works by training a diffusion model on the joint distribution of (coefficient, solution) pairs, then sampling with **guided diffusion** -- where the reverse diffusion process is steered by gradients from PDE residual losses and observation losses (physics-informed guidance).

## 2. Overall Project Structure

```
denoising_smc/
  |-- README.md                     # Project README
  |-- LICENSE                       # CC BY-NC-SA 4.0
  |-- AGENTS.md                     # Agent instructions (device auto-detect, setup, gotchas)
  |-- .gitignore                    # Git exclusion rules (large files, images, artifacts)
  |-- requirements.txt              # Python dependencies (PyTorch installed separately)
  |-- train.py                       # Main training entry point (derived from EDM)
  |-- generate_pde.py                # Main PDE solving entry point
  |-- merge_data.py                  # Utility to merge .mat data into .npy for training
  |-- idea.md                       # [ADDED] Research proposal: Girsanov correction, Heun-SDE, experimental design
  |-- recipe.md                     # [ADDED] Implementation guide: architecture, pseudocode, Python code, gotchas
  |-- papers.md                     # [ADDED] Literature survey of all 10 papers
  |
  |-- smc/                          # [ADDED] SMC module: proposals, weights, particle filter
  |   |-- __init__.py
  |   |-- hutchinson.py             # Hutchinson Laplacian trace estimator (feasibility study)
  |   |-- hutchinson_findings.md    # Findings: Laplacian cost/variance on real Burgers model
  |   |-- toy_smc.py                # 1D Gaussian-mixture toy: validates λ-ρ weighting vs analytic posterior
  |   |-- toy_smc_findings.md       # Findings: Girsanov sign fix, variant comparison, N-sweep
  |   |   # Planned (recipe.md): schedule.py, proposals.py, weights.py, core.py
  |
  |-- configs/                       # YAML configs for each PDE & task type
  |   |-- darcy.yaml                 # Darcy flow - both spaces
  |   |-- darcy-forward.yaml         # Darcy flow - forward problem
  |   |-- darcy-inverse.yaml         # Darcy flow - inverse problem
  |   |-- burgers.yaml               # Burgers' equation - time-dependent
  |   |-- poisson.yaml               # Poisson equation - both spaces
  |   |-- poisson-forward.yaml
  |   |-- poisson-inverse.yaml
  |   |-- helmholtz.yaml             # Helmholtz equation
  |   |-- helmholtz-forward.yaml
  |   |-- helmholtz-inverse.yaml
  |   |-- ns-bounded.yaml            # Bounded Navier-Stokes
  |   |-- ns-bounded-forward.yaml
  |   |-- ns-bounded-inverse.yaml
  |   |-- ns-nonbounded.yaml         # Non-bounded Navier-Stokes
  |   |-- ns-nonbounded-forward.yaml
  |   |-- ns-nonbounded-inverse.yaml
  |
  |-- scripts/                       # PDE-specific generation (solving) logic
  |   |-- __init__.py
  |   |-- generate_burgers.py
  |   |-- generate_darcy.py
  |   |-- generate_helmholtz.py
  |   |-- generate_ns_bounded.py
  |   |-- generate_ns_nonbounded.py
  |   |-- generate_poisson.py
  |
  |-- training/                      # Diffusion model training core
  |   |-- __init__.py
  |   |-- training_loop.py           # Main training loop (from EDM)
  |   |-- dataset.py                 # Dataset loading (from EDM)
  |   |-- loss.py                    # Loss functions: VP, VE, EDM (from EDM)
  |   |-- networks.py                # Model architectures: SongUNet, DhariwalUNet, preconditioners (from EDM)
  |   |-- augment.py                 # Data augmentation pipeline (from EDM)
  |
  |-- torch_utils/                   # PyTorch utilities (from EDM)
  |   |-- __init__.py
  |   |-- distributed.py             # Distributed training helpers
  |   |-- misc.py                    # Misc utilities (constant cache, etc.)
  |   |-- persistence.py             # Model persistence (pickle)
  |   |-- resizer.py                 # Image resizing
  |   |-- training_stats.py          # Training statistics tracking
  |
  |-- dnnlib/                        # Utility library (from EDM)
  |   |-- __init__.py
  |   |-- util.py                    # EasyDict, construct_class_by_name, logging, etc.
  |
  |-- pretrained-models/             # Pre-trained diffusion model checkpoints
  |   |-- README.md
  |   |-- pretrained-darcy.pkl
  |   |-- pretrained-burgers.pkl
  |   |-- pretrained-helmholtz.pkl
  |   |-- pretrained-poisson.pkl
  |   |-- pretrained-ns-bounded.pkl
  |   |-- pretrained-ns-nonbounded.pkl
  |
  |-- data/                          # Test data (testing only; training data must be downloaded)
  |   |-- testing/
  |       |-- darcy.mat
  |       |-- burgers.mat
  |       |-- poisson.mat
  |       |-- helmholtz.mat
  |       |-- ns-nonbounded.mat
  |       |-- ns-bounded/
  |           |-- 1/ (vx.npy, vy.npy, pressure.npy, cx.npy, cy.npy, r.npy, v.npy, v0.npy)
  |           |-- 2/ (same structure)
  |
  |-- AGENTS.md                     # [ADDED] Agent instructions (gotchas, commands, setup)
  |-- .gitignore                    # [ADDED] Git exclusion rules
  |-- requirements.txt              # [ADDED] Python dependencies (no torch)
  |-- map.md, git.md                # [ADDED] Project maps
  |
  |-- dataset_generation/            # PDE data generation code
  |   |-- static/                    # MATLAB codes for static PDEs
  |   |   |-- GRF.m                  # Gaussian Random Field generator
  |   |   |-- GRF_zero.m             # Zero-mean GRF
  |   |   |-- generate_darcy.m       # Darcy flow data generation
  |   |   |-- generate_poisson.m     # Poisson equation data generation
  |   |   |-- generate_inhom_helmholtz.m  # Helmholtz equation data generation
  |   |   |-- solve_gwf.m           # Solve generalized wave equation / finite difference solver
  |   |-- burgers/                   # MATLAB codes for Burgers' equation
  |   |   |-- GRF1.m
  |   |   |-- burgers1.m
  |   |   |-- gen_burgers1.m
  |   |-- non-bounded-ns/            # Python codes for non-bounded Navier-Stokes
  |       |-- random_fields.py
  |       |-- ns_2d.py
  |
  |-- docs/                          # Documentation / images
  |   |-- architecture.jpg           # Architecture diagram from the paper
  |
  |-- literature/                    # [USER-ADDED] Reference papers (NOT part of original repo)
      |-- arXiv-0000.00000v/         # [USER-ADDED] paper.md - FPS/FPS-SMC methodology summary
      |-- arXiv-2006.11239v2/        # [USER-ADDED] Original DDPM paper (Ho et al.)
      |-- arXiv-2011.13456v2/        # [USER-ADDED] Score SDE paper
      |-- arXiv-2302.13834v2/        # [USER-ADDED] DPS (Diffusion Posterior Sampling)
      |-- arXiv-2306.17775v2/        # [USER-ADDED] FPS (Filtering Posterior Sampling) paper
      |-- arXiv-2308.07983v2/        # [USER-ADDED] SMC + Diffusion paper (noisy/auxiliary PF)
      |-- arXiv-2402.06320v2/        # [USER-ADDED] Related diffusion paper
      |-- arXiv-2512.11012v2/        # [USER-ADDED] SMC-related paper
      |-- arXiv-2601.08411v2/        # [USER-ADDED] Particle filter noise paper
      |-- arXiv-2601.23262v2/        # [USER-ADDED] SMC/Diffusion paper (ICML 2026 format)
```

## 3. Key Files and Their Roles

### Entry Points
| File | Role |
|---|---|
| `train.py` | Main entry point for training diffusion models. Uses `click` CLI; derived from NVIDIA's EDM codebase. Supports DDPM++, NCSN++, and ADM architectures. |
| `generate_pde.py` | Main entry point for solving PDEs. Dispatches to PDE-specific generation scripts based on YAML config. |
| `merge_data.py` | Merges raw `.mat` training data (coefficient + solution) into joint `.npy` files scaled to (-1, 1) for diffusion model training. |

### PDE-Solving Scripts (`scripts/`)
| File | Role |
|---|---|
| `generate_darcy.py` | Solves Darcy flow: defines Darcy PDE loss (-div(a * grad(u)) = 1), observation losses, and guided sampling logic. |
| `generate_burgers.py` | Solves Burgers' equation (time-dependent 1D). |
| `generate_poisson.py` | Solves Poisson equation. |
| `generate_helmholtz.py` | Solves Helmholtz equation (inhomogeneous). |
| `generate_ns_bounded.py` | Solves bounded Navier-Stokes equations. |
| `generate_ns_nonbounded.py` | Solves non-bounded Navier-Stokes equations. |

These all implement the same pattern: load pretrained diffusion model, run reverse diffusion (Heun's 2nd order ODE solver from EDM), and at each step compute PDE residual loss + observation loss gradients to guide sampling.

### Training Core (`training/`)
| File | Role |
|---|---|
| `training_loop.py` | Full distributed training loop (from EDM). Handles checkpointing, EMA, logging. |
| `networks.py` | Model architectures: `SongUNet` (DDPM++/NCSN++), `DhariwalUNet` (ADM), plus preconditioners `EDMPrecond`, `VPPrecond`, `VEPrecond`. |
| `loss.py` | Loss functions: `EDMLoss`, `VPLoss`, `VELoss`. |
| `dataset.py` | `ImageFolderDataset` class for loading images/arrays with optional labels and caching. |
| `augment.py` | Stochastic data augmentation pipeline (wavelet-based). |

### Config Files (`configs/`)
Each YAML specifies:
- `data.name`: which PDE
- `data.datapath`: path to test data
- `data.offset`: which test sample to use
- `test.pre-trained`: path to pretrained model `.pkl`
- `test.iterations`: number of diffusion steps (default 2000)
- `generate.seed`, `generate.device`, `generate.batch_size`
- `generate.sigma_min`, `generate.sigma_max`, `generate.rho`: EDM ODE solver parameters
- `generate.zeta_obs_a`, `generate.zeta_obs_u`, `generate.zeta_pde`: guidance weights for observation and PDE losses

### Pretrained Models (`pretrained-models/`)
Six `.pkl` files (Darcy, Burgers, Poisson, Helmholtz, NS-bounded, NS-nonbounded). Each contains the EMA model weights from the EDM-style training.

## 4. User-Added Content (Not Part of Original Repo)

The following were added for the SMC extension project:

### Root-level Docs

| File | Role |
|------|------|
| `idea.md` | Research proposal: methodology, derivations (Girsanov, Heun-SDE), experimental design |
| `recipe.md` | Implementation guide: architecture, pseudocode, full Python code, gotchas |
| `papers.md` | Comprehensive literature survey of all 10 papers in `literature/` |

### `smc/` Module

The `smc/` directory contains the SMC implementation: proposals (GEM, HeunSDE, SOSaG), unified λ-ρ weights with Girsanov correction, and the ParticleFilter loop. See `recipe.md` for details.

- `smc/toy_smc.py` — self-contained 1D Gaussian-mixture toy that drives the same λ-ρ machinery against a known analytic posterior. Runs all proposal/weight variants (ODE, SOSaG+pBS, GEM+pBS, HeunSDE+pBS, GEM+Girs, HeunSDE+Girs) plus an N-sweep. `smc/toy_smc_findings.md` documents results, including the Girsanov Itô sign fix (also applied to `idea.md`/`recipe.md`).
- `smc/hutchinson.py` — Hutchinson estimator for the Laplacian term of the Doob-transform weight (feasibility study; see `smc/hutchinson_findings.md`).

### `literature/` Directory

Contains LaTeX source code and PDF metadata for 10 reference papers related to diffusion models, SMC, and posterior sampling:

| Directory | Paper | Relevance |
|---|---|---|
| `arXiv-0000.00000v/` | `paper.md` - A **custom-written summary** of the **FPS (Filtering Posterior Sampling)** and **FPS-SMC** methodology |
| `arXiv-2006.11239v2/` | Denoising Diffusion Probabilistic Models (DDPM) - Ho et al., NeurIPS 2020 |
| `arXiv-2011.13456v2/` | Score-Based Generative Modeling through SDEs - Song et al., ICLR 2021 |
| `arXiv-2302.13834v2/` | Diffusion Posterior Sampling (DPS) - Chung et al., ICLR 2023 |
| `arXiv-2306.17775v2/` | Filtering Posterior Sampling (FPS) - Dou & Song, NeurIPS 2023 |
| `arXiv-2308.07983v2/` | SMC + Diffusion papers - Dou & Song, ICLR 2024 |
| `arXiv-2402.06320v2/` | Related diffusion methodology |
| `arXiv-2512.11012v2/` | SMC-related paper (2025) |
| `arXiv-2601.08411v2/` | Particle filter noise paper (2026) |
| `arXiv-2601.23262v2/` | SMC/Diffusion paper (Millard et al., ICML 2026) |

The combination of `idea.md`, `recipe.md`, `smc/`, `literature/`, and the repository name `denoising_smc` indicates this is a research project that **combines denoising diffusion models with Sequential Monte Carlo methods** for posterior sampling, extending the DiffusionPDE codebase.

## 5. Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.8+ (primary), MATLAB (data generation) |
| **Deep Learning** | PyTorch 1.12.1 |
| **Diffusion Framework** | EDM (Elucidating Diffusion Models) by NVIDIA - provides training loop, network architectures (SongUNet, DhariwalUNet), preconditioning, loss functions |
| **Distributed Training** | `torchrun` / `torch.distributed` |
| **CLI** | `click` for `train.py` |
| **Config** | YAML (via `pyyaml`) |
| **Data Handling** | NumPy, SciPy (`.mat` files), PIL/Pillow |
| **Environment** | venv + `requirements.txt` (PyTorch installed separately for CPU or CUDA) |
| **Additional** | `psutil`, `tqdm`, `requests`, `imageio`, `pyspng` |

## 6. How the Codebase is Organized

The repository follows a **modular, two-part architecture**:

**Part 1 - Diffusion Model Training** (derived from NVIDIA's EDM):
- `train.py` (entry) -> `training/training_loop.py` (loop) -> `training/loss.py` (loss) + `training/networks.py` (model) + `training/dataset.py` (data) + `training/augment.py` (augmentation)
- All orchestrated via `dnnlib` utilities and `torch_utils` for distributed training, persistence, and stats.

**Part 2 - PDE Solving (Guided Diffusion)**:
- `generate_pde.py` (entry) -> reads YAML config -> dispatches to PDE-specific script in `scripts/`
- Each `scripts/generate_*.py` loads a pretrained diffusion model and implements:
  1. The PDE-specific residual loss (using finite difference convolutions)
  2. Observation loss (sparse sensor matching)
  3. The guided reverse diffusion loop (EDM ODE solver + gradient-based guidance)
- PDE-specific parameters are configured via YAML configs.

**Part 3 - Data Generation** (supporting code, not needed for inference):
- `dataset_generation/static/` (MATLAB) - for Darcy, Poisson, Helmholtz
- `dataset_generation/burgers/` (MATLAB) - for Burgers' equation
- `dataset_generation/non-bounded-ns/` (Python) - for non-bounded Navier-Stokes

**Git initialized** on branch `main`. Large files (9.1 GB of datasets/weights/images) excluded via `.gitignore`. See `git.md` for tracking details.
