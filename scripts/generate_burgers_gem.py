"""GEM (guided Euler-Maruyama) + Girsanov/TDS-weighted SMC for Burgers' equation.

First real-model arm of docs/idea.md's experimental design (Sec. 5.2): the simplest new
proposal (one network call/step, closed-form exact weight, no Hessian), run BEFORE the
Heun-SDE proposal idea.md proposes as the primary new result. See docs/note_1.pdf Sec. 4
("High dimensional model: PDE") -- this script is what fills that (currently empty) section.

Differences from the existing scripts/generate_burgers.py baseline, deliberately:
  * Proposal: guided Euler-Maruyama (docs/idea.md Sec. 3.1), not the deterministic 2nd-order
    Heun ODE + flat post-hoc gradient subtraction the baseline uses. Guidance now enters as a
    delta_k-SCALED drift term, delta_k*b_k, not an unscaled correction -- this is the actual
    algorithmic change under test, not a detail. It means the existing zeta_obs/zeta_pde values
    (tuned for the old unscaled correction) are only a starting guess here, not a given.
  * Weighting: incremental particles carry a Girsanov/TDS log-weight (docs/note_1.pdf eq. 18/38),
    with systematic resampling at ESS < 0.5*N. The baseline has no weights or resampling at all
    (batch_size=1, particles -- if any -- are independent, not an SMC population).
  * Cost: ONE network forward+backward per step (matching GEM's Table 3.4 entry: 1 denoiser call)
    vs the baseline's 2 (Heun predictor + corrector) -- so this should be *cheaper* per step at
    matched N=1, before any SMC benefit from multiple particles is even considered.

Run the correctness self-check (smc/scripts_2/check_gem_tds_real_model.py) before trusting any output here.

Recommended first run (fast smoke test, minutes not hours on CPU):
    .venv/bin/python -m scripts.generate_burgers_gem --config configs/burgers.yaml \\
        --n-particles 4 --num-steps 100

Then scale n-particles/num-steps up once the smoke test looks sane (nonzero ESS, no NaNs,
relative error in a plausible range).
"""

import argparse

import numpy as np
import torch
import tqdm
import yaml

from torch_utils.misc import auto_device
from smc.scripts_2.models.burgers import random_sensor, load_ground_truth, load_network, burger_loss
from smc.scripts_2.proposals.gem import denoise, gem_step
from smc.scripts_2.weightings.girsanov import girsanov_increment, effective_sample_size, systematic_resample_indices


def guidance_grad(x_cur, D, ground_truth, mask, zeta_obs, zeta_pde, use_pde, device):
    """Returns b_k = grad_x f(x_cur), f = log-likelihood surrogate (negative loss), and the two
    raw loss scalars per particle for logging. One backward pass (two if use_pde, matching the
    baseline's own two-phase cost)."""
    N = x_cur.shape[0]
    x_N = (D * 1.415).to(torch.float64)
    pde_loss, observation_loss = burger_loss(x_N, ground_truth, mask, device)
    dims = (1, 2)
    L_obs = torch.sqrt((observation_loss ** 2).sum(dim=dims)) / (128 * 5)     # [N], per-particle norm
    # Sum-then-grad: L_obs[i] depends only on x_cur[i] (no cross-particle coupling anywhere in
    # burger_loss), so d(sum_j L_obs[j])/d(x_cur[i]) = d(L_obs[i])/d(x_cur[i]) exactly.
    # This gets the whole batch's per-particle gradients from ONE backward call instead of N.
    f_val = -zeta_obs * L_obs.sum()
    grad_obs = torch.autograd.grad(outputs=f_val, inputs=x_cur, retain_graph=use_pde)[0]

    if use_pde:
        L_pde = torch.sqrt((pde_loss ** 2).sum(dim=dims)) / (128 * 128)
        f_pde = -zeta_pde * L_pde.sum()
        grad_pde = torch.autograd.grad(outputs=f_pde, inputs=x_cur)[0]
        b_k = grad_obs + grad_pde
        return b_k.detach(), L_obs.detach(), L_pde.detach()

    return grad_obs.detach(), L_obs.detach(), torch.zeros(N, dtype=torch.float64, device=device)


def generate_burgers_gem(config, n_particles=None, num_steps=None, resample_threshold=0.5,
                          out_path='burger-gem-results.npz'):
    device_cfg = config['generate']['device']
    device = auto_device() if device_cfg in (None, 'auto') else torch.device(device_cfg)

    ground_truth = load_ground_truth(config['data']['datapath'], config['data']['offset'], device)

    N = n_particles if n_particles is not None else config['generate']['batch_size']
    K = num_steps if num_steps is not None else config['test']['iterations']
    seed = config['generate']['seed']
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    net = load_network(config['test']['pre-trained'], device)

    sigma_min = max(config['generate']['sigma_min'], net.sigma_min)
    sigma_max = min(config['generate']['sigma_max'], net.sigma_max)
    rho = config['generate']['rho']
    idx = torch.arange(K, dtype=torch.float64, device=device)
    sched = (sigma_max ** (1 / rho) + idx / (K - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sched = net.round_sigma(sched)
    # Terminal point uses sigma_min itself rather than 0 (GEM's delta = sigma_cur^2 - sigma_next^2
    # needs sigma_next > 0 to stay well-defined and matches the toy_smc.py schedule convention,
    # which found appending a hard 0.0 produces a disproportionate final-step variance spike --
    # see smc/scripts_1/toy_smc.py build_sigma() docstring).

    zeta_obs = config['generate']['zeta_obs']
    zeta_pde = config['generate']['zeta_pde']
    selected_index = random_sensor(5, 128, seed=0, device=device)

    x = torch.randn(N, net.img_channels, net.img_resolution, net.img_resolution,
                     dtype=torch.float64, device=device, generator=generator) * sched[0]
    log_w = torch.zeros(N, dtype=torch.float64, device=device)   # f0(xi_0): flat prior at pure noise
    # No separate terminal correction (note_1.pdf eq. 23) is applied: that term corrects the
    # surrogate twist back to a *separately specified* exact final-time likelihood log p(y|xi_T).
    # Here there is no such separate likelihood -- the PDE-residual + sparse-observation loss IS
    # the only notion of "fit to y" this problem has, at every step including the last. So the
    # surrogate is being treated as exact by construction, not approximated; if a genuinely
    # different terminal criterion is introduced later this omission should be revisited.

    ess_history = []
    n_resample = 0

    for i in tqdm.tqdm(range(K - 1), unit='step'):
        sigma_cur, sigma_next = float(sched[i]), float(sched[i + 1])
        x_cur = x.detach().clone().requires_grad_(True)

        D, score = denoise(net, x_cur, sigma_cur)
        # Two-phase guidance schedule, matching scripts/generate_burgers.py's baseline exactly
        # (see AGENTS.md "Guidance two-phase"): obs-only at full weight for the first 80% of
        # steps; final 20% add PDE-residual gradients and cut the obs weight to zeta_obs/10.
        use_pde = i > 0.8 * K
        obs_weight = (zeta_obs / 10) if use_pde else zeta_obs
        b_k, L_obs, L_pde = guidance_grad(x_cur, D, ground_truth, selected_index,
                                           obs_weight, zeta_pde if use_pde else 0.0, use_pde, device)
        score = score.detach()
        x_cur = x_cur.detach()

        x_next, z, delta = gem_step(x_cur, score, b_k, sigma_cur, sigma_next, generator=generator)

        inc = girsanov_increment(b_k, z, delta)
        log_w = log_w + inc
        log_w = log_w - torch.logsumexp(log_w, dim=0)

        ess = effective_sample_size(log_w)
        ess_history.append(ess)
        if ess < resample_threshold * N:
            ridx = systematic_resample_indices(log_w, generator=generator)
            x_next = x_next[ridx]
            log_w = torch.zeros(N, dtype=torch.float64, device=device)
            n_resample += 1

        x = x_next

    x_final = (x * 1.415).to(torch.float64)
    w = torch.exp(log_w - torch.logsumexp(log_w, dim=0))
    weighted_mean = (w.view(N, 1, 1, 1) * x_final).sum(dim=0)

    per_particle_rel_err = (torch.norm((x_final - ground_truth).reshape(N, -1), dim=1) /
                             torch.norm(ground_truth))
    weighted_rel_err = torch.norm((weighted_mean - ground_truth).reshape(-1)) / torch.norm(ground_truth)

    print(f"N={N} particles, K={K} steps, {n_resample} resample events")
    print(f"weighted-mean relative error: {float(weighted_rel_err):.5f}")
    print(f"per-particle relative error: min={float(per_particle_rel_err.min()):.5f} "
          f"max={float(per_particle_rel_err.max()):.5f} mean={float(per_particle_rel_err.mean()):.5f}")
    print(f"final ESS: {ess_history[-1]:.2f} / {N}")

    np.savez(out_path,
             x_final=x_final.detach().cpu().numpy(),
             weights=w.detach().cpu().numpy(),
             weighted_mean=weighted_mean.detach().cpu().numpy(),
             per_particle_rel_err=per_particle_rel_err.detach().cpu().numpy(),
             weighted_rel_err=float(weighted_rel_err),
             ess_history=np.array(ess_history),
             n_resample=n_resample,
             ground_truth=ground_truth.detach().cpu().numpy())
    print(f"saved diagnostics to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/burgers.yaml')
    parser.add_argument('--n-particles', type=int, default=None,
                         help='overrides generate.batch_size in the config; start small (e.g. 4) for a smoke test')
    parser.add_argument('--num-steps', type=int, default=None,
                         help='overrides test.iterations in the config; start small (e.g. 100) for a smoke test')
    parser.add_argument('--resample-threshold', type=float, default=0.5)
    parser.add_argument('--out', type=str, default='burger-gem-results.npz')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    generate_burgers_gem(cfg, n_particles=args.n_particles, num_steps=args.num_steps,
                          resample_threshold=args.resample_threshold, out_path=args.out)
