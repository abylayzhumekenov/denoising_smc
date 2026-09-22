"""GEM + Girsanov-weighted SMC for Burgers' equation.

Method (Millard et al. 2026 convention; see docs/vision.md):

* **Units.** The reverse SDE, the score, the proposal and the Girsanov weight all live in the
  network's *latent* (training / "normalized") units. The likelihood/guidance is evaluated on
  the *raw* field ``u = to_raw(D)`` against the raw observations, exactly as the released ODE
  baseline and Millard's PDE likelihoods do. ``to_network``/``to_raw`` are the only place the
  two unit systems meet; the score/proposal never see them.
* **Likelihood**: ``ell = -obs_weight * L_obs - pde_weight * L_pde`` with the released losses
  ``L_obs = ||mask*(u - u_gt)||_2 / n`` and ``L_pde = ||u_t + u*u_x - nu*u_xx||_2 / m``
  (index-unit central differences, zero padding -- see ``scripts/generate_burgers.get_burger_loss``);
  ``obs_weight``/``pde_weight`` are tuned per PDE.
* **Proposal**: guided Euler--Maruyama (`smc/proposals/gem.py`), delta-scaled guidance with
  ``b = grad_x ell``; ``+delta * b`` descends the loss (same sign as Millard's guidance). Score
  convention: ``nabla_log_p = (D - x) / sigma**2`` is the true score ``+grad log p`` (Tweedie).
* **Weighting**: ``G_k = rho_temp * (Delta ell_k + lambda_girs * C_k)`` with the exact GEM
  kernel ratio ``C_k`` (`smc/weightings/girsanov.py`); ``lambda_girs=1`` corrected, ``0`` PBS.
* **Boundary**: ``log w0 = rho_temp_init * ell_0(x_0)``. **Resampling**: systematic when
  ``ESS < resample_threshold * N``. Readout: weighted mean.

Units / data normalization.  ``FIELD_SCALE = 1.415`` is the documented training normalization for
Burgers (the released ODE baseline denormalizes with ``*1.415``; ``merge_data.py`` documents the
inverse-transform principle; see ``docs/vision.md`` D14).  The raw field is ``u = to_raw(v)``.
The denoiser's output ``D`` is already in latent units, so the score ``(D - x)/sigma**2`` is used
as-is; only the likelihood and the saved output go through ``to_raw``.  In the likelihood the
field and the reference must both be raw (never mixed).

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
# Problem definitions (PDE + grid + declared data normalization); no tuned parameters.
FIELD_SCALE = 1.415           # raw = FIELD_SCALE * v  (documented training normalization, ~sqrt(2))
VISCOSITY = 0.01              # nu, from dataset_generation/burgers/burgers1.m (visc = 1/100)


def to_network(u_raw):
    """Per-field raw -> latent (network training units).  Inverse of ``to_raw``."""
    return u_raw / FIELD_SCALE


def to_raw(v):
    """Per-field latent (network training units) -> raw.  Inverse of ``to_network``."""
    return v * FIELD_SCALE


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


def burger_residual_raw(u):
    """Burgers residual in *raw* units, ``u`` of shape ``[N, 1, N_t, N_x]``.

    ``f = u_t + u*u_x - nu*u_xx`` with the released baseline's index-unit central differences
    and zero padding (cf. ``scripts/generate_burgers.get_burger_loss``).  Note: no grid-spacing
    division -- this is the convention whose weights the baseline/Millard tuned.  Returns
    ``[N, N_t, N_x]``.
    """
    deriv_t = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64, device=u.device)
    deriv_t = deriv_t.view(1, 1, 3, 1) / 2
    deriv_x = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float64, device=u.device)
    deriv_x = deriv_x.view(1, 1, 1, 3) / 2

    u_t = F.conv2d(u, deriv_t, padding=(1, 0))
    u_x = F.conv2d(u, deriv_x, padding=(0, 1))
    u_xx = F.conv2d(u_x, deriv_x, padding=(0, 1))
    return (u_t + u * u_x - VISCOSITY * u_xx).squeeze(1)


def likelihood(u, u_gt, mask, obs_weight, pde_weight):
    """Per-particle log surrogate ``[N]`` from a *raw*-units denoised field ``u``.

    ``L_obs = ||mask*(u - u_gt)||_2 / n``, ``L_pde = ||f(u)||_2 / m`` (the released baseline's
    unsquared-norm losses), returning ``-obs_weight * L_obs - pde_weight * L_pde``.  ``u`` and
    ``u_gt`` must both be in raw units (see module docstring).
    """
    pde = burger_residual_raw(u)                    # [N, N_t, N_x], raw units
    obs = (u.squeeze(1) - u_gt) * mask              # [N, N_t, N_x]
    L_obs = obs.norm(dim=(1, 2)) / mask.sum()
    L_pde = pde.norm(dim=(1, 2)) / (pde.shape[-1] * pde.shape[-2])
    return -obs_weight * L_obs - pde_weight * L_pde


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

    The latent state ``x``, the score, the proposal and the weights are in the network's latent
    units; the likelihood/guidance is evaluated on the raw field ``to_raw(D)`` against the raw
    ground truth, and the saved arrays/metric are raw.  See the module docstring for the unit
    discipline.
    """
    gen_cfg = config['generate']
    smc = config.get('smc', {})
    device_cfg = gen_cfg.get('device', 'auto')
    device = auto_device() if device_cfg in (None, 'auto') else torch.device(device_cfg)

    seed = gen_cfg['seed']
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Data layer: raw test field (the observation and evaluation space).
    ground_truth_raw = load_ground_truth(config['data']['datapath'], config['data']['offset'], device)
    net = load_network(config['test']['pre-trained'], device)
    mask = random_sensor(config['data'].get('sensors', 5), ground_truth_raw.shape[-1],
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
    log_w = rho_temp_init * likelihood(to_raw(D0), ground_truth_raw, mask, obs_weight, pde_weight)

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

        ell_cur = likelihood(to_raw(D), ground_truth_raw, mask, obs_weight, pde_weight)
        b_k = torch.autograd.grad(ell_cur.sum(), x_cur)[0].detach()

        x_next, z, delta = gem_step(x_cur.detach(), nabla_log_p.detach(), b_k,
                                    sigma_cur, sigma_next, generator=generator)

        # Evaluate the denoiser once at the post-step state; carried forward into iteration i+1.
        x_leaf_next = x_next.detach().clone().requires_grad_(True)
        D_next, nabla_log_p_next = denoise(net, x_leaf_next, sigma_next)
        ell_next = likelihood(to_raw(D_next.detach()), ground_truth_raw, mask, obs_weight, pde_weight)
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
    weighted_mean_norm = (w.view(n_particles, 1, 1, 1) * x).sum(dim=0)          # latent units
    particles_raw = to_raw(x).to(torch.float64)
    weighted_mean_raw = to_raw(weighted_mean_norm).to(torch.float64)

    # Raw units, against the raw ground truth (same metric as the ODE baseline).
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
        'relative_error': relative_error,            # raw units vs raw ground truth
        'field_scale': FIELD_SCALE,
    })
    print(f'saved run to {run_dir}')
    return run_dir
