# Repository Map & Git Tracking: denoising_smc

Consolidated reference for how the repo is organized and how it is tracked in git.
(Formerly `map.md` + `git.md`.)

---

## Part 1 — Repository Map

### 1. What the Original Repo Does

**DiffusionPDE: Generative PDE-Solving Under Partial Observation** (NeurIPS 2024)

This repository implements the paper *"DiffusionPDE: Generative PDE-Solving Under Partial Observation"* by Jiahe Huang, Guandao Yang, Zichen Wang, and Jeong Joon Park (University of Michigan / Stanford University).

The core idea: Use **denoising diffusion probabilistic models** to solve **Partial Differential Equations (PDEs)** given sparse/partial observations. It covers:

- **Forward problems**: Given sparse observations of the PDE coefficient (e.g., permeability field), recover the full solution field.
- **Inverse problems**: Given sparse observations of the solution field, recover the PDE coefficient.
- **Both-spaces recovery**: Simultaneously recover both coefficient and solution fields from observations on both sides.
- **Time-dependent recovery**: For time-dependent PDEs (e.g., Burgers', Navier-Stokes), recover the full solution throughout a time interval from sparse sensor measurements.

The method works by training a diffusion model on the joint distribution of (coefficient, solution) pairs, then sampling with **guided diffusion** -- where the reverse diffusion process is steered by gradients from PDE residual losses and observation losses (physics-informed guidance).

### 2. Overall Project Structure

```
denoising_smc/
  |-- README.md                     # Project README (entry point for users)
  |-- AGENTS.md                     # Agent instructions (entry point for agents)
  |-- LICENSE                       # CC BY-NC-SA 4.0
  |-- .gitignore                    # Git exclusion rules (large files, images, artifacts)
  |-- requirements.txt              # Python dependencies (PyTorch installed separately)
  |-- train.py                      # Main training entry point (derived from EDM)
  |-- generate_pde.py               # Main PDE solving entry point
  |-- merge_data.py                 # Utility to merge .mat data into .npy for training
  |
  |-- docs/                         # Project documentation (all non-root markdown lives here)
  |   |-- idea.md                   # Methodology: Girsanov correction, Heun-SDE, experimental design
  |   |-- recipe.md                 # Implementation guide: architecture, pseudocode, Python code (design spec)
  |   |-- note_2.tex                # Math note: Girsanov-corrected SMC for diffusion models
  |   |-- references_2.bib          # Bibliography for note_2.tex
  |   |-- repo.md                   # This file: repository map + git tracking
  |   |-- architecture.jpg          # Architecture diagram from the paper
  |
  |-- smc/                          # SMC module: proposals, weights, particle filter
  |   |-- __init__.py
  |   |-- toy_smc.py                # 1D Gaussian-mixture toy: validates λ-ρ weighting vs analytic posterior
  |   |-- toy_smc_findings.md       # Validation note / meeting report: results, W1, N-sweep
  |   |-- hutchinson.py             # Hutchinson Laplacian trace estimator (feasibility study)
  |   |-- hutchinson_findings.md    # Findings: Laplacian cost/variance on real Burgers model
  |   |   # Planned (docs/recipe.md): schedule.py, proposals.py, weights.py, core.py
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
  |-- pretrained-models/             # Pre-trained diffusion model checkpoints (gitignored)
  |   |-- README.md
  |   |-- pretrained-*.pkl           # Darcy, Burgers, Helmholtz, Poisson, NS-bounded, NS-nonbounded
  |
  |-- data/                          # Test data (gitignored; training data must be downloaded)
  |   |-- testing/
  |       |-- darcy.mat
  |       |-- burgers.mat
  |       |-- poisson.mat
  |       |-- helmholtz.mat
  |       |-- ns-nonbounded.mat
  |       |-- ns-bounded/
  |
  |-- dataset_generation/            # PDE data generation code
  |   |-- static/                    # MATLAB codes for static PDEs
  |   |-- burgers/                   # MATLAB codes for Burgers' equation
  |   |-- non-bounded-ns/            # Python codes for non-bounded Navier-Stokes
  |
  |-- literature/                    # Reference papers (research extension)
  |   |-- README.md                  # Literature survey of all papers (was papers.md)
  |   |-- arXiv-*/                   # Per-paper LaTeX/summaries (incl. FPS/FPS-SMC writeup)
```

### 3. Key Files and Their Roles

#### Entry Points
| File | Role |
|---|---|
| `train.py` | Main entry point for training diffusion models. Uses `click` CLI; derived from NVIDIA's EDM codebase. Supports DDPM++, NCSN++, and ADM architectures. |
| `generate_pde.py` | Main entry point for solving PDEs. Dispatches to PDE-specific generation scripts based on YAML config. |
| `merge_data.py` | Merges raw `.mat` training data (coefficient + solution) into joint `.npy` files scaled to (-1, 1) for diffusion model training. |

#### PDE-Solving Scripts (`scripts/`)
| File | Role |
|---|---|
| `generate_darcy.py` | Solves Darcy flow: defines Darcy PDE loss (-div(a * grad(u)) = 1), observation losses, and guided sampling logic. |
| `generate_burgers.py` | Solves Burgers' equation (time-dependent 1D). |
| `generate_poisson.py` | Solves Poisson equation. |
| `generate_helmholtz.py` | Solves Helmholtz equation (inhomogeneous). |
| `generate_ns_bounded.py` | Solves bounded Navier-Stokes equations. |
| `generate_ns_nonbounded.py` | Solves non-bounded Navier-Stokes equations. |

These all implement the same pattern: load pretrained diffusion model, run reverse diffusion (Heun's 2nd order ODE solver from EDM), and at each step compute PDE residual loss + observation loss gradients to guide sampling.

#### Training Core (`training/`)
| File | Role |
|---|---|
| `training_loop.py` | Full distributed training loop (from EDM). Handles checkpointing, EMA, logging. |
| `networks.py` | Model architectures: `SongUNet` (DDPM++/NCSN++), `DhariwalUNet` (ADM), plus preconditioners `EDMPrecond`, `VPPrecond`, `VEPrecond`. |
| `loss.py` | Loss functions: `EDMLoss`, `VPLoss`, `VELoss`. |
| `dataset.py` | `ImageFolderDataset` class for loading images/arrays with optional labels and caching. |
| `augment.py` | Stochastic data augmentation pipeline (wavelet-based). |

#### Config Files (`configs/`)
Each YAML specifies:
- `data.name`: which PDE
- `data.datapath`: path to test data
- `data.offset`: which test sample to use
- `test.pre-trained`: path to pretrained model `.pkl`
- `test.iterations`: number of diffusion steps (default 2000)
- `generate.seed`, `generate.device`, `generate.batch_size`
- `generate.sigma_min`, `generate.sigma_max`, `generate.rho`: EDM ODE solver parameters
- `generate.zeta_obs_a`, `generate.zeta_obs_u`, `generate.zeta_pde`: guidance weights for observation and PDE losses

#### SMC Extension (`smc/` + `docs/`)
- `docs/note_2.tex` — math note: Girsanov-corrected SMC, unified $\lambda$-$\rho$ weight, Euler/Heun proposals, appendices.
- `docs/idea.md` — original methodology proposal: Girsanov correction, Heun-SDE, experimental design.
- `docs/recipe.md` — implementation guide (design spec; modules not yet implemented).
- `smc/toy_smc.py` — 1D Gaussian-mixture toy validating the $\lambda$-$\rho$ weighting against a known analytic posterior (report: `smc/toy_smc_findings.md`).
- `smc/hutchinson.py` — Hutchinson estimator for the Laplacian term of the Doob-transform weight (feasibility study; see `smc/hutchinson_findings.md`).

#### Pretrained Models (`pretrained-models/`)
Six `.pkl` files (Darcy, Burgers, Poisson, Helmholtz, NS-bounded, NS-nonbounded). Each contains the EMA model weights from the EDM-style training. Excluded from git (see Part 2).

### 4. Tech Stack

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

### 5. How the Codebase is Organized

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

---

## Part 2 — Git Setup & Tracking

### Size Breakdown

| Scope | Size |
|---|---|
| Entire repository (unfiltered) | **9.1 GB** |
| After `.gitignore` exclusions | **~4.8 MB** (source code only) |

### Excluded Paths

| Path | Size | Reason |
|------|------|--------|
| `data/testing/darcy.mat` | 3.5 GB | Simulation dataset |
| `data/testing/helmholtz.mat` | 2.3 GB | Simulation dataset |
| `data/testing/burgers.mat` | 1.2 GB | Simulation dataset |
| `data/testing/ns-nonbounded.mat` | 688 MB | Simulation dataset |
| `data/testing/poisson.mat` | 243 MB | Simulation dataset |
| `data/testing/ns-bounded/` (16 .npy files) | 12 MB | Simulation dataset |
| `pretrained-models/pretrained-*.pkl` (6 files) | 1.3 GB | Model weights (downloadable) |
| `.DS_Store` (7 files) | ~56 KB | macOS junk |
| Images (`*.jpg`, `*.png`, etc.) | various | Binary figures/artifacts |

**Total excluded**: ~9.1 GB

### `.gitignore` Contents

```gitignore
# Large data / model files
data/testing/
pretrained-models/
*.pkl
*.mat
*.npy
*.npz

# Images / figures
*.jpg
*.jpeg
*.png
*.gif
*.bmp
*.pdf
*.eps

# Python bytecode
__pycache__/
*.pyc
*.pyo

# macOS
.DS_Store

# IDE / editor
.vscode/
.idea/
*.swp

# OpenCode workspace config
.opencode/

# Virtual environment
venv/

# Runtime artifacts
*.log
*.out
outputs/
results/
logs/
checkpoints/
runs/
wandb/
```

### Tracked (Included)

| Category | Files |
|----------|-------|
| Python source | `train.py`, `generate_pde.py`, `merge_data.py`, all of `scripts/`, `smc/`, `training/`, `torch_utils/`, `dnnlib/`, `dataset_generation/non-bounded-ns/` |
| YAML configs | 16 files in `configs/` |
| MATLAB source | `dataset_generation/static/`, `dataset_generation/burgers/` |
| Literature | All `.tex`, `.bib`, `.sty`, `.csv`, `.md` files in `literature/` (3.5 MB) |
| Docs | `README.md`, `AGENTS.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `docs/` |
| Total | **~4.8 MB** |

### How to Reproduce Datasets & Weights

- **Training data**: Download per original repo instructions (Google Drive links)
- **Test data**: Generated by MATLAB/Python scripts in `dataset_generation/`
- **Pretrained models**: Download from Google Drive (see `pretrained-models/README.md`), unzip into `pretrained-models/`
