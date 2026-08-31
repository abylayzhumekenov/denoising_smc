# Running on Ibex (KAUST)

## Why GPU, not CPU

Yes -- GPU is the right call here, not a nice-to-have. Concretely:

- `smc/hutchinson_findings.md` measured this exact codebase's real cost on an Apple M1 CPU:
  ~17-20s per forward pass, ~23-45s per forward+backward, all in float64 (this repo's convention
  throughout -- see AGENTS.md). At the baseline's default `iterations: 2000`, one lone trajectory
  (N=1) is already ~2000 x ~30s ~ 16-17 hours *sequentially*. A real SMC run (N particles, several
  PDEs, any ablation over N/K/lambda per `docs/idea.md` Sec. 5.3) multiplies that further.
- `scripts/generate_burgers_gem.py` already batches every particle into ONE network call per step
  (`x_cur` has shape `[N, C, H, W]`, single `net(x_cur, sigma)` call) -- on CPU this batching mostly
  just amortizes Python overhead, but on GPU it means going from N=1 to N=16 particles costs close
  to nothing extra per step (up to memory), which is exactly the regime `idea.md`'s own ablation
  range (`N in {1,2,4,8,16}`) needs to be affordable.
- V100/A100 (Ibex's GPU generations) have real double-precision hardware, unlike Apple Silicon's
  MPS backend (which has no float64 *at all* -- see `torch_utils/misc.py`'s `auto_device()`
  docstring for why this repo avoids MPS by default). So moving to GPU does not force a precision
  compromise the way it would on a Mac.
- `auto_device()` already checks `torch.cuda.is_available()` first, and every one of our new
  files (`smc/proposals.py`, `smc/weights.py`, `scripts/generate_burgers_gem.py`) creates tensors
  via `device=...` consistently -- so no code changes should be needed to run on CUDA. The thing to
  actually verify is environment setup, not the algorithm.

## Checklist

1. **Get the code onto Ibex.** The repo already has a GitHub remote
   (`github.com/abylayzhumekenov/denoising_smc.git`), so on an Ibex login node:
   ```bash
   git clone https://github.com/abylayzhumekenov/denoising_smc.git
   cd denoising_smc
   git pull   # make sure smc/proposals.py, smc/weights.py, scripts/generate_burgers_gem.py,
              # smc/check_gem_tds_real_model.py are present -- push from the Mac first if not
   ```

2. **Get the large files onto Ibex.** Pretrained checkpoints and test data are git-ignored
   (`.gitignore`: `pretrained-models/`, `data/testing/`, `*.pkl`, `*.mat`) and on this Mac live
   outside the repo entirely, symlinked in from `~/denoising_smc_external_data/`. Two options:
   - `scp -r`/`rsync` directly from the Mac to Ibex (simplest if that directory is intact and
     reasonably sized -- 6 checkpoints x ~208MB is ~1.2GB, worth using `rsync -avP` for resumability
     over a flaky connection):
     ```bash
     rsync -avP ~/denoising_smc_external_data/pretrained-models/ \
       <user>@glogin.ibex.kaust.edu.sa:~/denoising_smc/pretrained-models/
     ```
   - Or re-download from the original DiffusionPDE Google Drive links in `README.md` directly on
     Ibex, if login-node internet access allows it and the Mac-side copy is inconvenient to reach.
   - Note `data/testing/` did not exist in the current checkout when I checked (only the two
     pretrained-model symlinks were present) -- confirm where the actual `.mat` test files
     currently live (`~/denoising_smc_external_data/` too, most likely) before assuming they're
     ready to copy.

3. **Set up the Python environment on Ibex.** See `slurm/setup_env.sh` -- loads a CUDA module,
   creates a venv, installs the pinned `torch==1.12.1` (matching AGENTS.md's stated compatibility
   requirement) with CUDA support. Ibex's available CUDA module versions and GPU generations
   change over time and I can't see them from here -- run `module avail cuda` and `sinfo` yourself
   and adjust the placeholders marked `# TODO` in that script and in the `.sbatch` files below.
   If the pinned `torch==1.12.1` build doesn't have a wheel matching Ibex's current CUDA module,
   that's the one place worth deviating from the pin -- test the pinned version first since it's
   what this codebase was validated against, only move to a newer torch if forced to.

4. **Smoke-test on an actual GPU before submitting a real job.** `slurm/smoke_test.sbatch` runs
   `smc/check_gem_tds_real_model.py` (the closed-form correctness check) plus a tiny
   4-particle/100-step `generate_burgers_gem` run, and prints `torch.cuda.is_available()` /
   the detected device up front. Submit it first:
   ```bash
   sbatch slurm/smoke_test.sbatch
   ```
   and check the `.out` log names the GPU and both checks pass, before touching the full-scale job.
   (An interactive session, `srun --pty --time=00:30:00 --gres=gpu:1 bash`, works too if you'd
   rather poke around by hand first -- exact `--partition`/`--gres` syntax is Ibex-specific and
   marked `# TODO` in the templates for the same reason as above.)

5. **Then the real run.** `slurm/run_burgers_gem.sbatch` is the same idea at real scale
   (`--n-particles`, `--num-steps` as sbatch-script variables, defaulting to the full
   `configs/burgers.yaml` settings) -- adjust `--time`/`--mem` once you've seen how long the smoke
   test's 100-step run actually took on Ibex's GPUs, since that gives a real per-step timing to
   extrapolate from instead of guessing.

## What I can't do from here

I don't have any tool access to Ibex itself (no SSH/SLURM connector in this session) -- everything
above is a plan and ready-to-copy scripts, not something I've run or can monitor. Paste the actual
`sinfo`/`module avail cuda` output back to me and I'll fill in the placeholders precisely, and send
me whatever a run prints (or its `.out` log) the same way we've been doing locally.
