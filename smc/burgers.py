"""GEM + Girsanov-weighted SMC for Burgers' equation (clean implementation).

Method (see docs/vision.md):

* **Likelihood** (Millard et al., normalized MSE, evaluated at the denoised estimate):

      ell(x, sigma) = - obs_weight * (1/n) || M .* (u_obs - u) ||^2
                      - pde_weight * (1/m) || f(u) ||^2,
      u = 1.415 * D_theta(x, sigma),

  with ``f`` the conservative Burgers residual ``u_t + d_x(u^2/2) - nu u_xx``, ``M`` the 5
  sensor-column mask, ``n`` the number of observed entries and ``m`` the number of pixels.
  Weights are constant over the trajectory (no two-phase anneal); ``obs_weight``/``pde_weight``
  are placeholders (both default 1.0) to be tuned.

* **Proposal**: guided Euler--Maruyama (smc/proposals/gem.py), delta-scaled guidance.
* **Weighting**: ``G_k = rho_temp * (Delta ell_k + lambda_girs * C_k)`` with the exact GEM kernel
  ratio ``C_k`` (smc/weightings/girsanov.py); ``lambda_girs=1`` corrected, ``0`` pseudo-bootstrap.
* **Boundary**: ``log w0 = rho_temp_init * ell_0(x_0)``.
* **Resampling**: systematic when ``ESS < resample_threshold * N``. Readout: weighted mean.

The declared surrogate *is* the target (no oracle); see docs/note_4 and docs/vision.md.
"""

import pickle

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import tqdm

from common.results import make_run_dir, write_config, write_metrics
from smc.proposals.gem import denoise, gem_step
from smc.weightings.girsanov import girsanov_increment
from torch_utils.misc import auto_device


def load_ground_truth(datapath, offset, device):
    """Load the offset-th Burgers ground-truth field ``[128, 128]`` from the .mat test set."""
    data = scipy.io.loadmat(datapath)
    return torch.tensor(data['output'][offset, :, :], dtype=torch.float64, device=device)


def load_network(network_pkl, device):
    """Unpickle the EMA-weighted pretrained Burgers denoiser."""
    with open(network_pkl, 'rb') as f:
        return pickle.load(f)['ema'].to(device)


def random_sensor(k, grid_size, seed=0, device=None):
    """Binary mask ``[grid_size, grid_size]`` with ``k`` full columns set to 1."""
    if device is None:
        device = auto_device()
    torch.manual_seed(seed)
    index = torch.zeros(grid_size, grid_size, dtype=torch.float64, device=device)
    for i in torch.randperm(grid_size, device=device)[:k]:
        index[:, i] = 1
    return index


def burger_residual(u):
    """Conservative Burgers residual ``u_t + d_x(u^2/2) - nu u_xx`` for ``u`` of shape [N,1,128,128].

    Returns ``[N, 128, 128]``.
    """
    deriv_t = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=u.device).view(1, 1, 3, 1) / 2
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=u.device).view(1, 1, 1, 3) / 2
    u_t = F.conv2d(u, deriv_t, padding=(1, 0))
    flux_x = F.conv2d(0.5 * u ** 2, deriv_x, padding=(0, 1))
    u_xx = F.conv2d(F.conv2d(u, deriv_x, padding=(0, 1)), deriv_x, padding=(0, 1))
    return (u_t + flux_x - 0.01 * u_xx).squeeze(1)


def likelihood(D, u_gt, mask, obs_weight, pde_weight):
    """Per-particle log surrogate ``[N]`` from an already-evaluated denoised estimate ``D``."""
    u = (D * 1.415).to(torch.float64)              # [N,1,128,128], physical units
    f = burger_residual(u)                         # [N,128,128]
    obs = (u.squeeze(1) - u_gt) * mask             # [N,128,128]
    n = mask.sum()
    m = f.shape[-1] * f.shape[-2]
    mse_obs = (obs ** 2).sum(dim=(1, 2)) / n
    mse_pde = (f ** 2).sum(dim=(1, 2)) / m
    return -obs_weight * mse_obs - pde_weight * mse_pde


def effective_sample_size(log_w):
    """ESS of a 1-D tensor of (unnormalized) log-weights."""
    lw = log_w - log_w.max()
    w = torch.exp(lw)
    w = w / w.sum()
    return float(1.0 / (w ** 2).sum())


def systematic_resample_indices(log_w, generator=None):
    """Systematic resampling indices from a 1-D tensor of log-weights."""
    lw = log_w - log_w.max()
    w = torch.exp(lw)
    w = w / w.sum()
    n = w.shape[0]
    u0 = torch.rand((), generator=generator, dtype=w.dtype, device=w.device)
    u = (u0 + torch.arange(n, dtype=w.dtype, device=w.device)) / n
    cw = torch.cumsum(w, dim=0)
    return torch.searchsorted(cw, u).clamp(max=n - 1)


def run(config):
    """Run one SMC inference for Burgers and write ``results/smc/burgers/<run_id>/``.

    ``config`` has ``data`` (``datapath``, ``offset``, ``sensors``, ``sensor_seed``), ``test``
    (``pre-trained``), ``generate`` (``seed``, ``device``, ``sigma_min``, ``sigma_max``, ``rho``)
    and ``smc`` (``n_particles``, ``num_steps``, ``rho_temp``, ``rho_temp_init``, ``lambda_girs``,
    ``resample_threshold``, ``likelihood.{obs_weight,pde_weight}``, ``out_dir``, ``run_id``).
    """
    gen_cfg = config['generate']
    smc = config.get('smc', {})
    device_cfg = gen_cfg.get('device', 'auto')
    device = auto_device() if device_cfg in (None, 'auto') else torch.device(device_cfg)

    seed = gen_cfg['seed']
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    ground_truth = load_ground_truth(config['data']['datapath'], config['data']['offset'], device)
    net = load_network(config['test']['pre-trained'], device)
    mask = random_sensor(config['data'].get('sensors', 5), 128,
                         seed=config['data'].get('sensor_seed', 0), device=device)

    n_particles = smc.get('n_particles', 4)
    num_steps = smc.get('num_steps', 2000)
    rho_temp = smc.get('rho_temp', 1.0)
    rho_temp_init = smc.get('rho_temp_init', 1.0)
    lambda_girs = smc.get('lambda_girs', 1.0)
    resample_threshold = smc.get('resample_threshold', 0.5)
    ll = smc.get('likelihood', {})
    obs_weight = ll.get('obs_weight', 1.0)
    pde_weight = ll.get('pde_weight', 1.0)

    sigma_min = max(gen_cfg['sigma_min'], net.sigma_min)
    sigma_max = min(gen_cfg['sigma_max'], net.sigma_max)
    rho_sched = gen_cfg['rho']
    idx = torch.arange(num_steps, dtype=torch.float64, device=device)
    sched = (sigma_max ** (1 / rho_sched) + idx / (num_steps - 1) *
             (sigma_min ** (1 / rho_sched) - sigma_max ** (1 / rho_sched))) ** rho_sched
    sched = net.round_sigma(sched)  # terminal point is sigma_min (>0), keeping delta_k > 0

    x = torch.randn(n_particles, net.img_channels, net.img_resolution, net.img_resolution,
                    dtype=torch.float64, device=device, generator=generator) * sched[0]

    with torch.no_grad():
        D0, _ = denoise(net, x, sched[0])
    log_w = rho_temp_init * likelihood(D0, ground_truth, mask, obs_weight, pde_weight)

    ess_history = []
    x_leaf, D, score = None, None, None
    need_fresh_D = True

    for i in tqdm.tqdm(range(num_steps - 1), unit='step'):
        sigma_cur, sigma_next = float(sched[i]), float(sched[i + 1])

        if need_fresh_D:
            x_leaf = x.detach().clone().requires_grad_(True)
            D, score = denoise(net, x_leaf, sigma_cur)
            need_fresh_D = False
        x_cur = x_leaf

        ell_cur = likelihood(D, ground_truth, mask, obs_weight, pde_weight)
        b_k = torch.autograd.grad(ell_cur.sum(), x_cur)[0].detach()

        x_next, z, delta = gem_step(x_cur.detach(), score.detach(), b_k,
                                    sigma_cur, sigma_next, generator=generator)

        # Evaluate the denoiser once at the post-step state; carried forward into iteration i+1.
        x_leaf_next = x_next.detach().clone().requires_grad_(True)
        D_next, score_next = denoise(net, x_leaf_next, sigma_next)
        ell_next = likelihood(D_next.detach(), ground_truth, mask, obs_weight, pde_weight)
        delta_ell = ell_next - ell_cur.detach()

        inc = rho_temp * (delta_ell + girsanov_increment(b_k, z, delta, lambda_girs))
        log_w = log_w + inc
        log_w = log_w - torch.logsumexp(log_w, dim=0)

        ess = effective_sample_size(log_w)
        ess_history.append(ess)
        if ess < resample_threshold * n_particles:
            ridx = systematic_resample_indices(log_w, generator=generator)
            x_next = x_next[ridx]
            log_w = torch.zeros(n_particles, dtype=torch.float64, device=device)
            need_fresh_D = True
        else:
            x_leaf, D, score = x_leaf_next, D_next, score_next
        x = x_next

    x_final = (x * 1.415).to(torch.float64)
    w = torch.exp(log_w - torch.logsumexp(log_w, dim=0))
    weighted_mean = (w.view(n_particles, 1, 1, 1) * x_final).sum(dim=0)
    relative_error = float((torch.norm(weighted_mean - ground_truth) /
                            torch.norm(ground_truth)).detach())

    run_dir, run_id = make_run_dir(config, 'burgers', default_root='results/smc', section='smc')
    np.savez(run_dir / 'result.npz',
             particles=x_final.detach().cpu().numpy(),
             weights=w.detach().cpu().numpy(),
             weighted_mean=weighted_mean.detach().cpu().numpy(),
             ess_history=np.array(ess_history),
             ground_truth=ground_truth.detach().cpu().numpy())
    write_config(run_dir, config, run_id, section='smc')
    write_metrics(run_dir, {
        'method': 'smc',
        'pde': 'burgers',
        'run_id': run_id,
        'seed': seed,
        'num_steps': num_steps,
        'device': str(device),
        'relative_error': relative_error,
    })
    print(f'saved run to {run_dir}')
    return run_dir
