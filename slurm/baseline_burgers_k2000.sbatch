#!/bin/bash
# Runs the UNMODIFIED existing baseline (scripts/generate_burgers.py, via the repo's own
# generate_pde.py dispatcher) at the real default config -- iterations: 2000, batch_size: 1,
# straight from configs/burgers.yaml, no edits. This gives the missing apples-to-apples number:
# so far we only have the baseline at a coarse K=100 (relative error 1.2243535239722276, from an
# earlier CPU run) and the new GEM+SMC sampler at both K=100 (1.13) and the real K=2000
# (2.35, unexpectedly WORSE than K=100) -- we've never actually run the baseline itself at the
# matching K=2000. This tells us whether 2.35 is a GEM-specific regression or something the
# baseline also exhibits at this step count (in which case it's not GEM's guidance tuning at
# fault, but something about this particular PDE/observation setup at K=2000 generally).
#   sbatch slurm/baseline_burgers_k2000.sbatch
#
# Timing: the baseline does 2 network calls/step (Heun predictor+corrector) vs GEM's 1, but at
# batch_size=1 (vs GEM's N=8) -- expect roughly comparable or faster wall-time than the GEM K=2000
# run. `debug` is the confirmed-working partition for this account; --time is generous headroom
# under its 2h cap.
#SBATCH --job-name=baseline_k2000
#SBATCH --chdir=/home/alvarem/denoising_smc
#SBATCH --partition=debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miguelangel.alvarezballesteros@kaust.edu.sa

set -euo pipefail
mkdir -p slurm/logs

# Must match setup_env.sh exactly -- see smoke_test.sbatch's comment on this line for why this is
# required (Slurm jobs don't inherit login-node modules).
module load cuda/11.8
module load python/3.11.0

source .venv/bin/activate

echo "=== device check ==="
python3 -c "
import torch
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (unexpected on a GPU node)')
"

echo "=== unmodified baseline, configs/burgers.yaml as-is (iterations: 2000, batch_size: 1) ==="
python3 generate_pde.py --config configs/burgers.yaml

echo "=== done ==="
