# DiffusionPDE + Denoising SMC — Agent Instructions

## Entry Points

| Command | Purpose |
|---|---|
| `torchrun --standalone --nproc_per_node=N train.py --outdir=DIR --data=PATH --cond=0 --arch=ddpmpp --batch=60 --batch-gpu=20 --duration=20 --ema=0.05` | Train diffusion model (EDM-style) |
| `python3 generate_pde.py --config configs/<pde>.yaml` | Solve PDE via guided diffusion |
| `python3 merge_data.py` | Merge raw `.mat` → scaled `.npy` for training |
| `.venv/bin/python smc/scripts_1/toy_smc.py` | 1D Gaussian-mixture toy: uniform grid over twist (exact/surrogate/consistent/plug-in) × proposal × weighting (PBS/Girs/Pot/Pot-tr), tables T1–T4, 4-seed mean±std |

## Architecture

- **`training/`** — EDM-derived training loop, networks (SongUNet/DhariwalUNet), losses, dataset loader, augmentation
- **`smc/`** — SMC module. Four weightings (pseudo-bootstrap, Girsanov, potential, trapezoidal potential; `docs/note_1.pdf` §3) validated in `smc/scripts_1/toy_smc.py` (NumPy) as a uniform twist (exact/surrogate/consistent/plug-in) × proposal × weighting grid (tables T1–T4, 4-seed mean±std; T2–T4 use the realistic plug-in twist `N(y;D(x,σ),γ²)` as baseline). V_tau / Doob-transform (`docs/note_2.pdf`): toy (closed-form) validation in `smc/scripts_1/toy_mixture.py` and `smc/scripts_1/check_toy_mixture*.py`; real-model implementation in `smc/scripts_2/weightings/doob_vtau.py` and `smc/scripts_2/weightings/hutchinson.py` (Burgers adapter in `smc/scripts_2/models/burgers.py`), checked by `smc/scripts_2/check_v_tau_*.py`.
- **`scripts/generate_*.py`** — PDE-specific guided sampling (finite-difference convs, sensor mask, EDM reverse ODE)
- **`configs/*.yaml`** — Parameters: data path, model path, ODE solver, guidance weights
- **`dnnlib/`**, **`torch_utils/`** — EDM utilities (EasyDict, persistence, device auto-detect)

## Data Flow

1. Raw PDE simulation → `.mat` files (`dataset_generation/`)
2. `merge_data.py` scales to (-1, 1), stacks coefficient+solution channels → `.npy`
3. Training: `ImageFolderDataset` reads `.npy` files
4. Inference: `generate_pde.py` loads pretrained .pkl + test .mat data

## Key Conventions & Gotchas

- **Double precision**: All PDE guidance uses `torch.float64` throughout
- **Guidance two-phase**: Steps `i <= 0.8 * num_steps` use observation gradients only; final 20% add PDE residual gradients (0.1× observation weight)
- **Scaling**: Data at (-1, 1) for diffusion; inverse-transform before PDE/observation losses. Each PDE has own scale factors (Darcy: `a=(a+1.5)/0.2`, `u=(u+0.9)/115`; Burgers: `x*1.415`)
- **Pretrained models**: `.pkl` pickles, load with `pickle.load(f)['ema'].to(device)`. ~208 MB, excluded from git
- **EDM ODE solver**: Heun 2nd order, sigma schedule `sigma_t = (sigma_max^(1/rho) + t/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho`
- **Config observations**: Random sensor masks seeded separately from generation seed
- **`--cond=0`**: Model trained unconditionally; `--cond=1` would require labels
- **Sign bug**: `d/dτ = −d/d(sigma_t)`. See `docs/note_1.pdf` eq.(8) for correct C_k sign.

## Setup

See `README.md` §Setup. Python 3.8–3.10 (PyTorch 1.12.1 compat). No tests, formatter, or linter configured. License: CC BY-NC-SA 4.0 (EDM), MIT (PDE data).

## Device Auto-Detection

- `configs/*.yaml` use `device: 'auto'` → CUDA if available, else CPU
- `scripts/generate_*.py` helpers use `auto_device()` from `torch_utils.misc`
- **Training** (`train.py`) requires GPU / `torchrun`
- **Inference** (`generate_pde.py`) works on CPU (slow but functional)

## Reference Docs

- `docs/note_1.pdf` — Girsanov‑corrected SMC: three weightings, toy experiments, appendices
- `docs/note_2.pdf` — V_tau / Doob‑transform discretisation
- `smc/scripts_2/hutchinson_findings.md` — Hutchinson trace estimator benchmark
- `literature/README.md` — literature survey
