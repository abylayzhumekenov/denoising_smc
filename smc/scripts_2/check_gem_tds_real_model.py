"""Correctness self-check for smc/scripts_2/weights.girsanov_increment, on the REAL Burgers network.

Two checks, cheapest first:

1. test_batch_reduction() -- pure tensor arithmetic, no network, milliseconds. Confirms the
   per-particle pixel-dim reduction in girsanov_increment matches an explicit per-particle loop.

2. test_real_network_identity() -- ONE real network forward+backward pass (~seconds on CPU/M1,
   per smc/scripts_2/hutchinson_findings.md's own timing). Confirms that for the GEM proposal, the closed-
   form weight C_hat = girsanov_increment(b_k, z, delta) exactly equals the literal Gaussian
   kernel-log-ratio log[p(x_next|x_cur) / q(x_next|x_cur)] computed directly from the two
   (mean, delta*I)-Gaussian densities -- i.e. docs/note_1.pdf Appendix B eq. (38), verified
   numerically on the actual pretrained network and actual test data rather than algebraically.

Run this BEFORE any full multi-step / multi-particle run (scripts/generate_burgers_gem.py):
it is the "GEM-Girsanov must equal GEM-TDS" sanity check from docs/idea.md Sec. 5.2, done as a
single-step numerical identity instead of a full trajectory comparison, so it costs one network
call instead of a K-step run.

Run: .venv/bin/python -m smc.scripts_2.check_gem_tds_real_model --config configs/burgers.yaml
"""

import argparse
import pickle

import numpy as np
import scipy.io
import torch

from torch_utils.misc import auto_device
from scripts.generate_burgers import get_burger_loss, random_sensor
from smc.scripts_2.proposals import denoise, gem_step
from smc.scripts_2.weights import girsanov_increment


def test_batch_reduction():
    torch.manual_seed(0)
    N = 5
    shape = (N, 1, 8, 8)
    b = torch.randn(shape, dtype=torch.float64)
    z = torch.randn(shape, dtype=torch.float64)
    delta = 0.37

    got = girsanov_increment(b, z, delta)

    expected = torch.empty(N, dtype=torch.float64)
    for i in range(N):
        bz = (b[i] * z[i]).sum()
        b2 = (b[i] ** 2).sum()
        expected[i] = -bz - 0.5 * delta * b2

    ok = torch.allclose(got, expected, atol=1e-12, rtol=1e-10)
    print(f"[test_batch_reduction] {'PASS' if ok else 'FAIL'}  "
          f"max abs diff = {float((got - expected).abs().max()):.3e}")
    assert ok


def test_real_network_identity(config_path, sigma_target=5.0):
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = auto_device()
    torch.manual_seed(config['generate']['seed'])

    data = scipy.io.loadmat(config['data']['datapath'])
    ground_truth = torch.tensor(data['output'][config['data']['offset'], :, :],
                                 dtype=torch.float64, device=device)

    with open(config['test']['pre-trained'], 'rb') as f:
        net = pickle.load(f)['ema'].to(device)

    # Recompute the actual configured sigma schedule and pick the step closest to sigma_target,
    # so this check exercises a realistic (delta, sigma) pair rather than an arbitrary one.
    num_steps = config['test']['iterations']
    rho = config['generate']['rho']
    sigma_min = max(config['generate']['sigma_min'], net.sigma_min)
    sigma_max = min(config['generate']['sigma_max'], net.sigma_max)
    idx = torch.arange(num_steps, dtype=torch.float64, device=device)
    sched = (sigma_max ** (1 / rho) + idx / (num_steps - 1) *
             (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sched = net.round_sigma(sched)
    k = int(torch.argmin((sched - sigma_target).abs()))
    k = min(k, num_steps - 2)  # need a sigma_next
    sigma_cur = float(sched[k])
    sigma_next = float(sched[k + 1])
    print(f"[test_real_network_identity] step {k}: sigma_cur={sigma_cur:.4f}, "
          f"sigma_next={sigma_next:.4f}")

    x_cur = torch.randn(1, net.img_channels, net.img_resolution, net.img_resolution,
                         dtype=torch.float64, device=device) * sigma_cur
    x_cur.requires_grad_(True)

    D, score = denoise(net, x_cur, sigma_cur)
    x_N = (D * 1.415).to(torch.float64)

    selected_index = random_sensor(5, 128, seed=0, device=device)
    pde_loss, observation_loss = get_burger_loss(x_N, ground_truth, selected_index, device)
    L_obs = torch.norm(observation_loss, 2) / (128 * 5)
    f_val = -config['generate']['zeta_obs'] * L_obs  # log-likelihood surrogate ell(x) (obs-only phase)

    b_k = torch.autograd.grad(outputs=f_val, inputs=x_cur)[0].detach()
    score = score.detach()
    x_cur = x_cur.detach()

    x_next, z, delta = gem_step(x_cur, score, b_k, sigma_cur, sigma_next)

    C_hat = girsanov_increment(b_k, z, delta)

    mu_p = x_cur + delta * score               # unguided kernel mean
    mu_q = mu_p + delta * b_k                   # guided kernel mean (= x_cur + delta*(score+b_k))
    log_ratio = -0.5 / delta * (
        torch.sum((x_next - mu_p) ** 2, dim=tuple(range(1, x_next.dim())))
        - torch.sum((x_next - mu_q) ** 2, dim=tuple(range(1, x_next.dim())))
    )

    diff = float((C_hat - log_ratio).abs().max())
    rel = diff / max(float(log_ratio.abs().max()), 1e-30)
    ok = diff < 1e-6 or rel < 1e-9
    print(f"[test_real_network_identity] C_hat={float(C_hat[0]):.6f}  "
          f"literal_kernel_log_ratio={float(log_ratio[0]):.6f}  "
          f"abs diff={diff:.3e}  rel diff={rel:.3e}  -> {'PASS' if ok else 'FAIL'}")
    assert ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/burgers.yaml')
    parser.add_argument('--sigma-target', type=float, default=5.0)
    args = parser.parse_args()

    test_batch_reduction()
    test_real_network_identity(args.config, args.sigma_target)
    print("\nAll checks passed.")
