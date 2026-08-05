# DiffusionPDE + Denoising SMC — Agent Instructions

## Entry Points

| Command | Purpose |
|---|---|
| `torchrun --standalone --nproc_per_node=N train.py --outdir=DIR --data=PATH --cond=0 --arch=ddpmpp --batch=60 --batch-gpu=20 --duration=20 --ema=0.05` | Train diffusion model (EDM-style) |
| `python3 generate_pde.py --config configs/<pde>.yaml` | Solve PDE via guided diffusion |
| `python3 merge_data.py` | Merge raw `.mat` → scaled `.npy` for training |
| `venv/bin/python smc/toy_smc.py` | 1D Gaussian-mixture toy: validate SMC λ-ρ weighting vs analytic posterior |

## Architecture

- **`training/`** — EDM-derived training loop, networks (SongUNet/DhariwalUNet), loss functions (VP/VE/EDM), dataset loader (`ImageFolderDataset`), augmentation
- **`smc/`** — SMC module: proposals (GEM/HeunSDE/SOSaG), unified λ-ρ weights, ParticleFilter loop. Methodology in `docs/idea.md`, implementation in `docs/recipe.md`. `smc/toy_smc.py` is a 1D Gaussian-mixture validation toy (results in `smc/toy_smc_findings.md`); `smc/hutchinson.py` is a Laplacian-trace feasibility study
- **`scripts/generate_*.py`** — PDE-specific guided sampling. Each implements: PDE residual loss (finite-difference convs), observation loss (sparse sensor mask), and EDM reverse ODE with gradient guidance
- **`configs/*.yaml`** — All parameters: data path/offset, pretrained model path, ODE solver params (sigma_min/sigma_max/rho), guidance weights (zeta_obs_a, zeta_obs_u, zeta_pde). Naming: `<pde>.yaml` = both spaces, `<pde>-forward.yaml` = forward, `<pde>-inverse.yaml` = inverse.
- **`dnnlib/`**, **`torch_utils/`** — EDM utilities (EasyDict, distributed, persistence, training stats). Persistence pickles source code alongside weights; models load via `pickle.load(f)['ema']`. Device auto-detection lives in `torch_utils.misc.auto_device()`

## Data Flow

1. Raw PDE simulation → `.mat` files (or MATLAB/Python generators in `dataset_generation/`)
2. `merge_data.py` loads .mat, scales values to (-1, 1) range, stacks coefficient+solution channels → `.npy`
3. Training: `ImageFolderDataset` reads `.npy` files
4. Inference: `generate_pde.py` loads both the pretrained .pkl and test .mat data

## Key Conventions & Gotchas

- **Double precision**: All PDE guidance code uses `torch.float64` throughout (latents, network outputs, loss computation)
- **Guidance two-phase**: Steps `i <= 0.8 * num_steps` apply only observation gradients; final 20% add PDE residual gradients (at 0.1× observation weight). Hardcoded in each `generate_*.py`
- **Scaling**: Data is transformed to (-1, 1) for diffusion model. Inverse transform is applied each step before computing PDE/observation losses. Each PDE has its own scale factors (e.g. Darcy: `a = (a+1.5)/0.2`, `u = (u+0.9)/115`; Burgers: `x * 1.415`)
- **Pretrained models**: Stored as `.pkl` pickles. Load with `pickle.load(f)['ema'].to(device)`. Models are ~208 MB each, excluded from git
- **EDM ODE solver**: Heun's 2nd order method with sigma schedule `sigma_t = (sigma_max^(1/rho) + t/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho`
- **Config observations**: Random sensor masks (`random_index`/`random_sensor`) are seeded by a separate seed from the generation seed
- **`--cond=0`**: The diffusion model is trained unconditionally (no class labels). `--cond=1` would require labels

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA):
pip install torch torchvision
```

- **Python**: 3.8–3.10 (PyTorch 1.12.1 compat)
- **No tests, formatter, or linter** configured
- **License**: CC BY-NC-SA 4.0 (EDM), MIT (PDE data generation)

## Device Auto-Detection

- All `configs/*.yaml` use `device: 'auto'` — resolves to CUDA if available, else CPU
- Helper functions in `scripts/generate_*.py` use `auto_device()` from `torch_utils.misc`
- **Training** (`train.py`) requires GPU / `torchrun` (single-GPU training is possible but impractical on CPU)
- **Inference** (`generate_pde.py`) works on CPU (slow but functional)

## Reference Docs

- `docs/idea.md` — research proposal: methodology, derivations (Girsanov, Heun-SDE), experimental design
- `docs/recipe.md` — implementation guide: architecture, pseudocode, Python code, gotchas (design spec — `smc/` modules not yet implemented)
- `literature/README.md` — comprehensive literature survey of all papers in `literature/`
- `docs/repo.md` — repository map + git tracking (tree, file roles, tech stack, excluded paths)
- `smc/toy_smc_findings.md` — toy validation results (Girsanov sign fix, variant comparisons, N-sweep)

## Docs Policy

- Root holds only the two entry points: `README.md` (users) and `AGENTS.md` (agents)
- All other markdown lives under `docs/` or co-located with its subject (`smc/`, `literature/`, `pretrained-models/`)
- A doc that is only reachable from this reference list is a consolidation candidate

## Project Context (User Extension)

- `literature/` contains collected reference papers on diffusion models, SMC, and posterior sampling (DDPM, Score SDE, DPS, FPS, SMC+Diffusion)
- `literature/arXiv-0000.00000v/paper.md` is a custom methodology writeup for FPS/FPS-SMC — the user's intended extension
- `idea.md` contains the research proposal (Girsanov correction, Heun-SDE, experimental design) — now `docs/idea.md`
- `recipe.md` contains the implementation guide (architecture, pseudocode, Python code) — now `docs/recipe.md`
- The repo name `denoising_smc` and the literature dir indicate the user plans to integrate Sequential Monte Carlo methods into the DiffusionPDE guided sampling pipeline
