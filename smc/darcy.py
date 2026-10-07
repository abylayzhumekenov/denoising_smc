"""GEM + Girsanov-weighted SMC for Darcy flow (joint coefficient/solution reconstruction).

Ported from ``smc/burgers.py`` (same method, same unit discipline); see that module and
``docs/vision.md`` D15/D16/D18 for the conventions.  Only the problem-specific parts differ:

* **Units.** The reverse SDE, the score, the proposal and the Girsanov weight live in the
  network's *latent* (training / "normalized") units.  The likelihood/guidance is evaluated on the
  *raw* field ``to_raw(D)`` against the raw observations.  ``to_network``/``to_raw`` are the only
  place the two unit systems meet; the score/proposal never see them.  Darcy has two channels
  ``[a, u]`` with distinct documented affine normalizations (from ``merge_data.py`` / the released
  ODE baseline): ``v_a = 0.2*a - 1.5`` and ``v_u = 115*u - 0.9``.
* **Likelihood** (Millard's squared form, mirroring ``smc/burgers.py`` -- *not* the ODE baseline's
  unsquared ``||.||_2`` norms): with raw ``a, u``,
  ``l_obs_a = (1/n_a) sum mask_a*(a - a_gt)^2``, ``l_obs_u = (1/n_u) sum mask_u*(u - u_gt)^2``,
  ``l_res = (1/m) sum f(a,u)^2``, and ``ell = -obs_weight_a*l_obs_a - obs_weight_u*l_obs_u
  - pde_weight*l_res``.  ``f = div(a grad u) + 1`` (physical derivatives) is the analytic raw
  Darcy residual; the weights absorb all units/discretization/physics scaling and are tuned per
  problem (no invariance claim).
* **Proposal / weighting / resampling / boundary**: identical to Burgers (EM or Heun proposal,
  Girsanov-exact ``C_k``, systematic resampling, ``log w0 = rho_temp_init * ell_0``).

The declared surrogate *is* the target (no oracle); see docs/note_4 and docs/vision.md.
"""

import pickle

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import tqdm

from common.results import make_run_dir, write_config, write_metrics
from smc.proposals.gem import brownian_increment, denoise, gem_step
from smc.proposals.heun import heun_correct, heun_predict
from smc.weightings.girsanov import girsanov_increment
from torch_utils.misc import auto_device

# --- Darcy / grid / normalization definitions ------------------------------------------------
# Problem definitions (PDE + grid + declared data normalization); no tuned parameters.
DOMAIN_LENGTH = 1.0           # domain [0, 1]^2
A_LATENT_SCALE = 0.2          # v_a = 0.2*a_raw - 1.5   (merge_data.py / generate_darcy.py)
A_LATENT_SHIFT = -1.5
U_LATENT_SCALE = 115.0        # v_u = 115*u_raw - 0.9   (merge_data.py / generate_darcy.py)
U_LATENT_SHIFT = -0.9
A_BINARY_MID = 7.5            # threshold used by the ODE baseline for the binary a field


def to_network_a(a_raw):
    """Raw coefficient field -> latent (inverse of ``to_raw_a``)."""
    return A_LATENT_SCALE * a_raw + A_LATENT_SHIFT


def to_raw_a(v_a):
    """Latent coefficient field -> raw (inverse of ``to_network_a``)."""
    return (v_a - A_LATENT_SHIFT) / A_LATENT_SCALE


def to_network_u(u_raw):
    """Raw solution field -> latent (inverse of ``to_raw_u``)."""
    return U_LATENT_SCALE * u_raw + U_LATENT_SHIFT


def to_raw_u(v_u):
    """Latent solution field -> raw (inverse of ``to_network_u``)."""
    return (v_u - U_LATENT_SHIFT) / U_LATENT_SCALE


def to_raw(v):
    """Latent field ``[N, 2, H, W]`` -> raw: channel 0 = ``a``, channel 1 = ``u`` (raw units)."""
    return torch.stack((to_raw_a(v[:, 0]), to_raw_u(v[:, 1])), dim=1)


def to_network(raw):
    """Raw field ``[N, 2, H, W]`` (channels ``[a, u]``) -> latent.  Inverse of ``to_raw``."""
    return torch.stack((to_network_a(raw[:, 0]), to_network_u(raw[:, 1])), dim=1)


def load_ground_truth(datapath, offset, device):
    """Load the offset-th Darcy ground truth, returning ``(a_gt, u_gt)`` raw ``[H, W]`` fields.

    ``a_gt = thresh_a_data[offset]`` (binary 3/12), ``u_gt = thresh_p_data[offset]`` (raw), matching
    the released ODE baseline (``scripts/generate_darcy.py``).
    """
    data = scipy.io.loadmat(datapath)
    a_gt = torch.tensor(data['thresh_a_data'][offset, :, :], dtype=torch.float64, device=device)
    u_gt = torch.tensor(data['thresh_p_data'][offset, :, :], dtype=torch.float64, device=device)
    return a_gt, u_gt


def load_network(network_pkl, device):
    """Unpickle the EMA-weighted pretrained Darcy denoiser."""
    with open(network_pkl, 'rb') as f:
        return pickle.load(f)['ema'].to(device)


def random_index(k, grid_size, seed=0, device=None):
    """Binary mask ``[grid_size, grid_size]`` with ``k`` scattered grid points set to 1.

    Mirrors ``scripts/generate_darcy.py`` (a flat ``np.random.choice`` over ``grid_size**2``), so
    the SMC sees the same observation pattern as the ODE baseline for a given seed.
    """
    if device is None:
        device = auto_device()
    np.random.seed(seed)
    indices = np.random.choice(grid_size ** 2, k, replace=False)
    rows, cols = np.unravel_index(indices, (grid_size, grid_size))
    mask = torch.zeros(grid_size, grid_size, dtype=torch.float64, device=device)
    mask[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = 1
    return mask


def darcy_residual(a, u):
    """Darcy residual in *training (index) units*, ``a, u`` of shape ``[N, H, W]``.

    ``f = d_x(a*d_x u) + d_y(a*d_y u) + 1`` with central differences in *index* units (divided by 2
    only -- no ``/dx``, ``/dy``) and zero padding, mirroring the released baseline / Millard
    convention (``scripts/generate_darcy.get_darcy_loss``).  This keeps the residual on the same
    scale as Millard's published likelihood weights; the discretization/scale is absorbed by
    ``pde_weight``.  Returns ``[N, H, W]``.
    """
    a = a.unsqueeze(1)
    u = u.unsqueeze(1)
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float64, device=u.device)
    kernel_x = kernel_x.view(1, 1, 1, 3) / 2.0
    kernel_y = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64, device=u.device)
    kernel_y = kernel_y.view(1, 1, 3, 1) / 2.0

    grad_x = F.conv2d(u, kernel_x, padding=(0, 1))
    grad_y = F.conv2d(u, kernel_y, padding=(1, 0))
    div_x = F.conv2d(a * grad_x, kernel_x, padding=(0, 1))
    div_y = F.conv2d(a * grad_y, kernel_y, padding=(1, 0))
    return (div_x + div_y + 1.0).squeeze(1)


def term_losses(a, u, a_gt, u_gt, mask_a, mask_u):
    """Squared per-term losses (raw units), each shape ``[N]`` (Millard's likelihood form).

    ``l_obs_a = (1/n_a) sum mask_a*(a - a_gt)^2``, ``l_obs_u = (1/n_u) sum mask_u*(u - u_gt)^2``,
    ``l_res = (1/m) sum f(a,u)^2``.  ``a, u`` and ``a_gt, u_gt`` must all be raw units.
    """
    f = darcy_residual(a, u)                                  # [N, H, W], raw units
    da = (a - a_gt) * mask_a                                  # [N, H, W]
    du = (u - u_gt) * mask_u
    L_obs_a = (da ** 2).sum(dim=(1, 2)) / mask_a.sum()
    L_obs_u = (du ** 2).sum(dim=(1, 2)) / mask_u.sum()
    L_pde = (f ** 2).sum(dim=(1, 2)) / (f.shape[-1] * f.shape[-2])
    return L_obs_a, L_obs_u, L_pde


def likelihood(raw, a_gt, u_gt, mask_a, mask_u, obs_weight_a, obs_weight_u, pde_weight):
    """Per-particle log surrogate ``[N]`` (raw units):
    ``-obs_weight_a*l_obs_a - obs_weight_u*l_obs_u - pde_weight*l_res``.
    """
    L_obs_a, L_obs_u, L_pde = term_losses(raw[:, 0], raw[:, 1], a_gt, u_gt, mask_a, mask_u)
    return -obs_weight_a * L_obs_a - obs_weight_u * L_obs_u - pde_weight * L_pde


def binary_a(a_raw):
    """Threshold a continuous raw coefficient field to the two dataset values (3 / 12)."""
    return torch.where(a_raw > A_BINARY_MID, torch.full_like(a_raw, 12.0),
                       torch.full_like(a_raw, 3.0))


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
    """Run one SMC inference for Darcy and write ``results/smc/darcy/<run_id>/``.

    The latent state ``x`` (two channels ``[a, u]``), the score, the proposal and the weights are in
    the network's latent units; the likelihood/guidance is evaluated on the raw field
    ``to_raw(D)`` against the raw ground truth, and the saved arrays/metrics are raw.  See the
    module docstring for the unit discipline.
    """
    gen_cfg = config['generate']
    smc = config.get('smc', {})
    device_cfg = gen_cfg.get('device', 'auto')
    device = auto_device() if device_cfg in (None, 'auto') else torch.device(device_cfg)

    seed = gen_cfg['seed']
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Data layer: raw test fields (the observation and evaluation space).
    a_gt, u_gt = load_ground_truth(config['data']['datapath'], config['data']['offset'], device)
    net = load_network(config['test']['pre-trained'], device)
    sensors = config['data'].get('sensors', 500)
    mask_a = random_index(sensors, u_gt.shape[-1],
                          seed=config['data'].get('sensor_seed_a', 1), device=device)
    mask_u = random_index(sensors, u_gt.shape[-1],
                          seed=config['data'].get('sensor_seed_u', 0), device=device)

    n_particles = smc.get('n_particles', 4)
    num_steps = smc.get('num_steps', 2000)
    rho_temp = smc.get('rho_temp', 1.0)
    rho_temp_init = smc.get('rho_temp_init', 1.0)
    lambda_girs = smc.get('lambda_girs', 1.0)
    resample_threshold = smc.get('resample_threshold', 0.5)
    ll = smc.get('likelihood', {})
    obs_weight_a = ll.get('obs_weight_a', 1.0)
    obs_weight_u = ll.get('obs_weight_u', 1.0)
    pde_weight = ll.get('pde_weight', 1.0)
    proposal = smc.get('proposal', 'em')
    if proposal not in ('em', 'heun'):
        raise ValueError(f"unknown smc.proposal: {proposal!r} (expected 'em' or 'heun')")

    # Guidance-attribution diagnostic: at ~sampled steps, decompose the total gradient b into its
    # per-term contributions by projection, f_c = -beta_c * <grad L_c, b> / ||b||^2  (sum_c f_c = 1).
    n_diag = min(12, num_steps - 1)
    diag_steps = set(int(round(j)) for j in
                     np.linspace(0, num_steps - 2, n_diag)) if n_diag > 0 else set()
    diag = {k: [] for k in ('step', 'sigma', 'loss_obs_a', 'loss_obs_u', 'loss_pde',
                            'frac_a', 'frac_u', 'frac_pde', 'b_norm', 'score_norm')}

    sigma_min = max(gen_cfg['sigma_min'], net.sigma_min)
    sigma_max = min(gen_cfg['sigma_max'], net.sigma_max)
    rho_sched = gen_cfg['rho']
    idx = torch.arange(num_steps, dtype=torch.float64, device=device)
    sched = (sigma_max ** (1 / rho_sched) + idx / (num_steps - 1) *
             (sigma_min ** (1 / rho_sched) - sigma_max ** (1 / rho_sched))) ** rho_sched
    sched = net.round_sigma(sched)  # terminal point is sigma_min (>0), keeping delta_k > 0

    x = torch.randn(n_particles, net.img_channels, net.img_resolution, net.img_resolution,
                    dtype=torch.float64, device=device, generator=generator) * sched[0]

    def ell_of(D):
        return likelihood(to_raw(D), a_gt, u_gt, mask_a, mask_u,
                          obs_weight_a, obs_weight_u, pde_weight)

    with torch.no_grad():
        D0, _ = denoise(net, x, sched[0])
    log_w = rho_temp_init * ell_of(D0)

    ess_history = []
    rel_err_history = []
    rel_err_particle_history = []
    grad_norm_history = []
    nabla_log_p_norm_history = []
    delta_history = []
    proposal_correction_rel_history = []
    x_leaf, D, nabla_log_p = None, None, None
    need_fresh_D = True

    for i in tqdm.tqdm(range(num_steps - 1), unit='step'):
        sigma_cur, sigma_next = float(sched[i]), float(sched[i + 1])

        if need_fresh_D:
            x_leaf = x.detach().clone().requires_grad_(True)
            D, nabla_log_p = denoise(net, x_leaf, sigma_cur)
            need_fresh_D = False
        x_cur = x_leaf

        ell_cur = ell_of(D)
        b_k = torch.autograd.grad(ell_cur.sum(), x_cur)[0].detach()

        if i in diag_steps:
            with torch.no_grad():
                b2 = (b_k ** 2).sum(dim=(1, 2, 3)).clamp_min(1e-30)          # [N]
                x_det = x_cur.detach()
                rms_x = (x_det ** 2).mean(dim=(1, 2, 3)).sqrt()             # [N]
                eps = (1e-3 * rms_x / b2.sqrt()).view(-1, 1, 1, 1)          # [N,1,1,1]
                raw_p = to_raw(denoise(net, x_det + eps * b_k, sigma_cur)[0])
                raw_m = to_raw(denoise(net, x_det - eps * b_k, sigma_cur)[0])
                raw_0 = to_raw(D.detach())
                La_p, Lu_p, Lp_p = term_losses(raw_p[:, 0], raw_p[:, 1], a_gt, u_gt, mask_a, mask_u)
                La_m, Lu_m, Lp_m = term_losses(raw_m[:, 0], raw_m[:, 1], a_gt, u_gt, mask_a, mask_u)
                La_0, Lu_0, Lp_0 = term_losses(raw_0[:, 0], raw_0[:, 1], a_gt, u_gt, mask_a, mask_u)
                dLa = (La_p - La_m) / (2 * eps.view(-1))
                dLu = (Lu_p - Lu_m) / (2 * eps.view(-1))
                dLp = (Lp_p - Lp_m) / (2 * eps.view(-1))
                diag['step'].append(i)
                diag['sigma'].append(sigma_cur)
                diag['loss_obs_a'].append(float(La_0.mean()))
                diag['loss_obs_u'].append(float(Lu_0.mean()))
                diag['loss_pde'].append(float(Lp_0.mean()))
                diag['frac_a'].append(float((-obs_weight_a * dLa / b2).mean()))
                diag['frac_u'].append(float((-obs_weight_u * dLu / b2).mean()))
                diag['frac_pde'].append(float((-pde_weight * dLp / b2).mean()))
                diag['b_norm'].append(float(b_k.norm()))
                diag['score_norm'].append(float(nabla_log_p.detach().norm()))

        if proposal == 'em':
            x_next, z, delta = gem_step(x_cur.detach(), nabla_log_p.detach(), b_k,
                                        sigma_cur, sigma_next, generator=generator)
            proposal_correction_rel = 0.0
        else:  # heun: predict, evaluate the drift at x_pred, trapezoidal correct (reusing z)
            delta = float(sigma_cur) ** 2 - float(sigma_next) ** 2
            if delta <= 0:
                raise ValueError(
                    f"non-decreasing sigma schedule: sigma_cur={sigma_cur}, sigma_next={sigma_next}")
            drift_cur = (nabla_log_p + b_k).detach()
            z = brownian_increment(x_cur.shape, delta, generator=generator,
                                   dtype=x_cur.dtype, device=x_cur.device)
            x_pred = heun_predict(x_cur.detach(), drift_cur, z, delta)
            x_pred_leaf = x_pred.detach().clone().requires_grad_(True)
            D_pred, nabla_log_p_pred = denoise(net, x_pred_leaf, sigma_next)
            ell_pred = ell_of(D_pred)
            b_pred = torch.autograd.grad(ell_pred.sum(), x_pred_leaf)[0].detach()
            drift_pred = (nabla_log_p_pred + b_pred).detach()
            x_next = heun_correct(x_cur.detach(), drift_cur, drift_pred, z, delta)
            with torch.no_grad():
                num = (0.5 * delta * (drift_pred - drift_cur)).flatten(1).norm(dim=1)
                den = (delta * drift_cur).flatten(1).norm(dim=1).clamp_min(1e-30)
                proposal_correction_rel = float((num / den).mean())
            # Release the predictor graph now; only the post-step graph at x_next is carried on.
            x_pred_leaf = D_pred = nabla_log_p_pred = ell_pred = b_pred = None

        # Evaluate the denoiser once at the post-step state; carried forward into iteration i+1.
        x_leaf_next = x_next.detach().clone().requires_grad_(True)
        D_next, nabla_log_p_next = denoise(net, x_leaf_next, sigma_next)
        ell_next = ell_of(D_next.detach())
        delta_ell = ell_next - ell_cur.detach()

        inc = rho_temp * (delta_ell + girsanov_increment(b_k, z, delta, lambda_girs))
        log_w = log_w + inc
        log_w = log_w - torch.logsumexp(log_w, dim=0)

        ess = effective_sample_size(log_w)
        ess_history.append(ess)
        grad_norm_history.append(float(b_k.norm()))
        nabla_log_p_norm_history.append(float(nabla_log_p.detach().norm()))
        delta_history.append(delta)
        proposal_correction_rel_history.append(proposal_correction_rel)

        if ess < resample_threshold * n_particles:
            ridx = systematic_resample_indices(log_w, generator=generator)
            x_next = x_next[ridx]
            log_w = torch.zeros(n_particles, dtype=torch.float64, device=device)
            need_fresh_D = True
            x_leaf_next = D_next = nabla_log_p_next = None
        else:
            x_leaf, D, nabla_log_p = x_leaf_next, D_next, nabla_log_p_next
        x = x_next

        # Per-step relative error of u to the raw ground truth (cheap; no forward pass).
        with torch.no_grad():
            ur = to_raw_u(x[:, 1])                                            # raw, [N, H, W]
            denom = u_gt.norm()
            per = (ur - u_gt).flatten(1).norm(dim=1) / denom                  # [N]
            rel_err_particle_history.append(float(per.mean()))
            wn = torch.exp(log_w - torch.logsumexp(log_w, dim=0))
            wm = (wn.view(-1, 1, 1) * ur).sum(dim=0)                          # [H, W]
            rel_err_history.append(float((wm - u_gt).norm() / denom))

    w = torch.exp(log_w - torch.logsumexp(log_w, dim=0))
    weighted_mean_norm = (w.view(n_particles, 1, 1, 1) * x).sum(dim=0)          # latent [2, H, W]
    particles_raw = to_raw(x).to(torch.float64)                                 # [N, 2, H, W]
    weighted_mean_raw = to_raw(weighted_mean_norm.unsqueeze(0)).squeeze(0).to(torch.float64)

    # Raw units, against the raw ground truth (same metrics as the ODE baseline).
    a_wm = binary_a(weighted_mean_raw[0])
    u_wm = weighted_mean_raw[1]
    relative_error_u = float((torch.norm(u_wm - u_gt) / torch.norm(u_gt)).detach())
    error_rate_a = float((a_wm != a_gt).to(torch.float64).mean().detach())

    run_dir, run_id = make_run_dir(config, 'darcy', default_root='results/smc', section='smc')
    np.savez(run_dir / 'result.npz',
             particles=particles_raw.detach().cpu().numpy(),
             weights=w.detach().cpu().numpy(),
             weighted_mean=weighted_mean_raw.detach().cpu().numpy(),
             ess_history=np.array(ess_history),
             rel_err_history=np.array(rel_err_history),
             rel_err_particle_history=np.array(rel_err_particle_history),
             grad_norm_history=np.array(grad_norm_history),
             nabla_log_p_norm_history=np.array(nabla_log_p_norm_history),
             delta_history=np.array(delta_history),
             proposal_correction_rel_history=np.array(proposal_correction_rel_history),
             diag_step=np.array(diag['step']),
             diag_sigma=np.array(diag['sigma']),
             diag_loss_obs_a=np.array(diag['loss_obs_a']),
             diag_loss_obs_u=np.array(diag['loss_obs_u']),
             diag_loss_pde=np.array(diag['loss_pde']),
             diag_frac_a=np.array(diag['frac_a']),
             diag_frac_u=np.array(diag['frac_u']),
             diag_frac_pde=np.array(diag['frac_pde']),
             diag_b_norm=np.array(diag['b_norm']),
             diag_score_norm=np.array(diag['score_norm']),
             a_gt=a_gt.detach().cpu().numpy(),
             u_gt=u_gt.detach().cpu().numpy())
    write_config(run_dir, config, run_id, section='smc')
    write_metrics(run_dir, {
        'method': 'smc',
        'pde': 'darcy',
        'proposal': proposal,
        'run_id': run_id,
        'seed': seed,
        'num_steps': num_steps,
        'device': str(device),
        'relative_error_u': relative_error_u,        # raw units vs raw ground truth
        'error_rate_a': error_rate_a,                # binary a mismatch fraction
        # Guidance attribution, mean over sampled steps (per-step values in result.npz; sum ~ 1).
        'guidance_frac_a_mean': float(np.mean(diag['frac_a'])) if diag['frac_a'] else None,
        'guidance_frac_u_mean': float(np.mean(diag['frac_u'])) if diag['frac_u'] else None,
        'guidance_frac_pde_mean': float(np.mean(diag['frac_pde'])) if diag['frac_pde'] else None,
        'proposal_correction_rel_mean': (float(np.mean(proposal_correction_rel_history))
                                         if proposal_correction_rel_history else None),
    })
    print(f'saved run to {run_dir}')
    return run_dir
