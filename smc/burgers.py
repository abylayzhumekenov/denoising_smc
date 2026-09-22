"""GEM + Girsanov-weighted SMC for Burgers' equation (clean implementation).

Method (see docs/vision.md):

* **Likelihood** (Millard et al., normalized MSE, evaluated at the denoised estimate):
  ``log p = -obs_weight * l_obs - pde_weight * l_res`` with *dimensionless* per-element
  mean-squared errors ``l_obs`` and ``l_res`` (defined in `likelihood`).
* **Proposal**: guided Euler--Maruyama (`smc/proposals/gem.py`), delta-scaled guidance. The
  guidance gradient is ``b = grad ell`` (``ell`` is the log surrogate above), so the drift term
  ``+delta * b = -delta * (obs_weight * grad l_obs + pde_weight * grad l_res)`` descends the loss
  -- the same sign as Millard's ``-(...) * grad log p_tilde``. Score convention (see
  `smc/proposals/gem.py`): ``nabla_log_p = (D - x) / sigma**2`` is the true score ``+grad log p``
  (Tweedie), not EDM's ``(x - D) / sigma**2``.
* **Weighting**: ``G_k = rho_temp * (Delta ell_k + lambda_girs * C_k)`` with the exact GEM
  kernel ratio ``C_k`` (`smc/weightings/girsanov.py`); ``lambda_girs=1`` corrected, ``0`` PBS.
* **Boundary**: ``log w0 = rho_temp_init * ell_0(x_0)``.
* **Resampling**: systematic when ``ESS < resample_threshold * N``. Readout: weighted mean.

Normalization.  Raw ``.mat`` fields are converted to the network's training units (the
"normalized" units) once, on load, and the denoiser / proposal / losses all operate in those
units; we denormalize only for saving and for the raw evaluation metric.  This keeps a clean
separation between the data layer and the sampler, instead of sprinkling the inverse scale
through the loss as the ODE baseline does.

* ``NORMALIZATION_SCALE = 1.415`` (= sqrt(2) rounded).  Evidence: the raw Burgers test fields
  have global ``max|u| = sqrt(2)`` (1.41422), and the ODE baseline multiplies the network
  output by ``1.415`` to return to the raw units (`scripts/generate_burgers.py:107,123`), i.e.
  training divided the raw data by ``sqrt(2)`` to bring it to ~(-1, 1).  The forward map is not
  present in this repo for Burgers (training data absent); the constant is the declared
  convention, corroborated by the data.  (For other PDEs the forward map is explicit, e.g. Darcy
  in `merge_data.py`.)
* The residual is written in normalized units ``v`` (network output), so ``u = sqrt(2) * v`` and
  the conservative Burgers residual becomes ``f = v_t + sqrt(2) * d_x(v^2/2) - nu * v_xx``.

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

# --- Burgers / grid / normalization definitions ---------------------------------------------
# All are problem definitions (PDE + grid + declared data normalization); no tuned parameters.
NORMALIZATION_SCALE = 1.415   # raw -> normalized is /NORMALIZATION_SCALE (= sqrt(2) rounded)
VISCOSITY = 0.01              # nu, from dataset_generation/burgers/burgers1.m (visc = 1/100)
DOMAIN_LENGTH = 1.0           # x in [0, 1], periodic
TIME_SPAN = 1.0               # t in [0, 1]
FIELD_AMPLITUDE = 1.0         # normalized field amplitude A (data scaled to ~(-1, 1))


def load_ground_truth(datapath, offset, device):
    """Load the offset-th Burgers ground-truth field ``[N_t, N_x]`` (raw ``.mat`` units)."""
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


def residual_scale():
    """Natural scale of the Burgers residual in normalized units (A=1).

    Term scales: ``v_t ~ A/T``, ``sqrt(2) d_x(v^2/2) ~ sqrt(2) A^2 / L``,
    ``nu v_xx ~ nu A / L^2``.  Combined in quadrature.
    """
    terms = [
        FIELD_AMPLITUDE / TIME_SPAN,
        NORMALIZATION_SCALE * FIELD_AMPLITUDE ** 2 / DOMAIN_LENGTH,
        VISCOSITY * FIELD_AMPLITUDE / DOMAIN_LENGTH ** 2,
    ]
    return sum(t * t for t in terms) ** 0.5


def burger_residual(v):
    """Conservative Burgers residual in *normalized* units, ``v`` of shape ``[N,1,N_t,N_x]``.

    ``f = v_t + sqrt(2) d_x(v^2/2) - nu v_xx`` with physical derivatives (divided by the grid
    spacing) and the correct boundary conditions: periodic in space, replicated in time.
    Returns ``[N, N_t, N_x]``.
    """
    n_t, n_x = v.shape[-2], v.shape[-1]
    dt = TIME_SPAN / (n_t - 1)
    dx = DOMAIN_LENGTH / n_x
    kernel_t = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64, device=v.device)
    kernel_t = kernel_t.view(1, 1, 3, 1) / (2 * dt)
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float64, device=v.device)
    kernel_x = kernel_x.view(1, 1, 1, 3) / (2 * dx)

    v_t = F.conv2d(F.pad(v, (0, 0, 1, 1), mode='replicate'), kernel_t)
    flux_x = F.conv2d(F.pad(0.5 * NORMALIZATION_SCALE * v ** 2, (1, 1, 0, 0), mode='circular'),
                      kernel_x)
    v_xx = F.conv2d(F.pad(F.conv2d(F.pad(v, (1, 1, 0, 0), mode='circular'), kernel_x),
                          (1, 1, 0, 0), mode='circular'), kernel_x)
    return (v_t + flux_x - VISCOSITY * v_xx).squeeze(1)


def likelihood(v, y_norm, mask, obs_weight, pde_weight):
    """Dimensionless per-particle log surrogate ``[N]`` from denoised estimate ``v``.

    ``l_obs = (1/n) sum_obs (v - y_norm)^2`` (amplitude A=1), ``l_res = (1/m) sum (f/F)^2``
    with ``F = residual_scale()``; returns ``-obs_weight * l_obs - pde_weight * l_res``.
    """
    f = burger_residual(v)                          # [N, N_t, N_x], normalized units
    obs = (v.squeeze(1) - y_norm) * mask            # [N, N_t, N_x]
    n = mask.sum()
    m = f.shape[-1] * f.shape[-2]
    l_obs = (obs ** 2).sum(dim=(1, 2)) / (FIELD_AMPLITUDE ** 2 * n)
    l_res = (f ** 2).sum(dim=(1, 2)) / (m * residual_scale() ** 2)
    return -obs_weight * l_obs - pde_weight * l_res


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

    The denoiser, proposal, likelihood and weighting all operate in normalized units
    (raw ``.mat`` field divided by ``NORMALIZATION_SCALE``); only the saved arrays and the
    raw evaluation metric are denormalized.
    """
    gen_cfg = config['generate']
    smc = config.get('smc', {})
    device_cfg = gen_cfg.get('device', 'auto')
    device = auto_device() if device_cfg in (None, 'auto') else torch.device(device_cfg)

    seed = gen_cfg['seed']
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Data layer: convert the raw test field to the network's normalized units here, once.
    ground_truth_raw = load_ground_truth(config['data']['datapath'], config['data']['offset'], device)
    ground_truth = ground_truth_raw / NORMALIZATION_SCALE
    net = load_network(config['test']['pre-trained'], device)
    mask = random_sensor(config['data'].get('sensors', 5), ground_truth.shape[-1],
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
    grad_norm_history = []
    nabla_log_p_norm_history = []
    delta_history = []
    x_leaf, D, nabla_log_p = None, None, None
    need_fresh_D = True

    for i in tqdm.tqdm(range(num_steps - 1), unit='step'):
        sigma_cur, sigma_next = float(sched[i]), float(sched[i + 1])

        if need_fresh_D:
            x_leaf = x.detach().clone().requires_grad_(True)
            D, nabla_log_p = denoise(net, x_leaf, sigma_cur)
            need_fresh_D = False
        x_cur = x_leaf

        ell_cur = likelihood(D, ground_truth, mask, obs_weight, pde_weight)
        b_k = torch.autograd.grad(ell_cur.sum(), x_cur)[0].detach()

        x_next, z, delta = gem_step(x_cur.detach(), nabla_log_p.detach(), b_k,
                                    sigma_cur, sigma_next, generator=generator)

        # Evaluate the denoiser once at the post-step state; carried forward into iteration i+1.
        x_leaf_next = x_next.detach().clone().requires_grad_(True)
        D_next, nabla_log_p_next = denoise(net, x_leaf_next, sigma_next)
        ell_next = likelihood(D_next.detach(), ground_truth, mask, obs_weight, pde_weight)
        delta_ell = ell_next - ell_cur.detach()

        inc = rho_temp * (delta_ell + girsanov_increment(b_k, z, delta, lambda_girs))
        log_w = log_w + inc
        log_w = log_w - torch.logsumexp(log_w, dim=0)

        ess = effective_sample_size(log_w)
        ess_history.append(ess)
        grad_norm_history.append(float(b_k.norm()))
        nabla_log_p_norm_history.append(float(nabla_log_p.detach().norm()))
        delta_history.append(delta)

        if ess < resample_threshold * n_particles:
            ridx = systematic_resample_indices(log_w, generator=generator)
            x_next = x_next[ridx]
            log_w = torch.zeros(n_particles, dtype=torch.float64, device=device)
            need_fresh_D = True
            # The carried-forward D_next/nabla_log_p_next were computed on the pre-resample particle
            # ordering; release them so the discarded graph is freed before the fresh forward
            # at the top of the next iteration (otherwise two graphs are alive at once).
            x_leaf_next = D_next = nabla_log_p_next = None
        else:
            x_leaf, D, nabla_log_p = x_leaf_next, D_next, nabla_log_p_next
        x = x_next

    w = torch.exp(log_w - torch.logsumexp(log_w, dim=0))
    weighted_mean_norm = (w.view(n_particles, 1, 1, 1) * x).sum(dim=0)          # [1, N_t, N_x]
    particles_raw = (x * NORMALIZATION_SCALE).to(torch.float64)
    weighted_mean_raw = (weighted_mean_norm * NORMALIZATION_SCALE).to(torch.float64)

    # Raw units (comparable to the ODE baseline). The scalar NORMALIZATION_SCALE cancels in the
    # ratio, so this is identical to the normalized-units error; no separate metric is needed.
    relative_error = float((torch.norm(weighted_mean_raw - ground_truth_raw) /
                            torch.norm(ground_truth_raw)).detach())

    run_dir, run_id = make_run_dir(config, 'burgers', default_root='results/smc', section='smc')
    np.savez(run_dir / 'result.npz',
             particles=particles_raw.detach().cpu().numpy(),
             weights=w.detach().cpu().numpy(),
             weighted_mean=weighted_mean_raw.detach().cpu().numpy(),
             ess_history=np.array(ess_history),
             grad_norm_history=np.array(grad_norm_history),
             nabla_log_p_norm_history=np.array(nabla_log_p_norm_history),
             delta_history=np.array(delta_history),
             ground_truth=ground_truth_raw.detach().cpu().numpy())
    write_config(run_dir, config, run_id, section='smc')
    write_metrics(run_dir, {
        'method': 'smc',
        'pde': 'burgers',
        'run_id': run_id,
        'seed': seed,
        'num_steps': num_steps,
        'device': str(device),
        'relative_error': relative_error,            # raw units (denormalized); scale-invariant
        'normalization_scale': NORMALIZATION_SCALE,
        'residual_scale': residual_scale(),
    })
    print(f'saved run to {run_dir}')
    return run_dir
