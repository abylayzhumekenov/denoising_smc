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
  * Weighting: incremental particles carry the full corrected Girsanov/TDS log-weight
    (denoising_smc_manuscript/notes/reconcile.md Sec. 3-6):
        G_k = Delta_ell_k + C_k^phi
    where Delta_ell_k = ell_{k-1}(x_{k-1}) - ell_k(x_k) is the telescoping twist-ratio increment
    (ell_k = log of the guidance potential phi_k = -zeta_obs*L_obs - zeta_pde*L_pde, the SAME
    surrogate that b_k = grad ell_k is built from) and C_k^phi is the stochastic Girsanov
    correction (girsanov_increment). Earlier versions of this script only ever accumulated
    C_k^phi -- Delta_ell_k was silently never computed, which is neither the full corrected
    weight nor the PBS/Millard baseline (Delta_ell_k alone); it was an incomplete hybrid with no
    mechanism to reflect how well a particle currently fits the observations. See reconcile.md
    Sec. 6 and Sec. 8 for the lambda-ablation this generalizes to:
        G_k^lambda = Delta_ell_k + lambda * C_k^phi   (lambda=0 -> PBS/Millard, lambda=1 -> full)
    Systematic resampling still fires at ESS < resample_threshold*N. The baseline has no weights
    or resampling at all (batch_size=1, particles -- if any -- are independent, not an SMC
    population).
  * Cost: computing Delta_ell_k needs a second (no-grad, forward-only) network call per step, to
    evaluate the twist at the post-step state x_next/sigma_next -- so this is no longer strictly
    "one network call per step" the way the original design intended, though it's still cheaper
    than the baseline's 2 calls each requiring a full backward pass (this second call needs no
    backward at all).

Run the correctness self-check (smc/scripts_2/check_gem_tds_real_model.py) before trusting any output here.

Recommended first run (fast smoke test, minutes not hours on CPU):
    .venv/bin/python -m scripts.generate_burgers_gem --config configs/burgers.yaml \\
        --n-particles 4 --num-steps 100

Then scale n-particles/num-steps up once the smoke test looks sane (nonzero ESS, no NaNs,
relative error in a plausible range).

Diagnostic controls (all default to the corrected GEM behavior when omitted):
  --no-noise          zero out the injected Brownian increment in gem_step (inject_noise=False).
  --flat-guidance     add guidance flat/unscaled by delta instead of folding it into the
                      delta-scaled score drift (scale_guidance=False) -- mirrors
                      scripts/generate_burgers.py's baseline convention.
  --no-guidance       zero out the guidance term entirely (apply_guidance=False) -- pure
                      unconditional guided-Euler-Maruyama with no observation/PDE conditioning.
  --lambda-girsanov   weight on the Girsanov correction C_k^phi in G_k = Delta_ell_k +
                      lambda*C_k^phi (default 1.0, the full corrected weight). Set to 0.0 to
                      reproduce the Millard-style pseudo-bootstrap (PBS) weight, Delta_ell_k
                      alone, on this real Burgers problem.
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


def twist_log_likelihood(D, ground_truth, mask, obs_weight, pde_weight, device):
    """ell(x,sigma) = -obs_weight*L_obs(x,sigma) - pde_weight*L_pde(x,sigma): the per-particle
    log-twist value (log of the guidance potential phi_k), computed from an ALREADY-EVALUATED
    denoised estimate D = D_theta(x,sigma). This is the ell_k(x_k) of
    denoising_smc_manuscript/notes/reconcile.md Sec. 2-3 -- b_k = grad_x ell_k(x_cur) is exactly
    this function's gradient (see guidance_grad's f_val), so the two must stay consistent: same
    zeta_obs/zeta_pde, same use_pde phase, for the SAME step index.

    No autograd needed here -- this is used only for the Delta_ell_k = ell_next - ell_cur
    telescoping term in the SMC weight, never for a gradient.
    """
    x_N = (D * 1.415).to(torch.float64)
    pde_loss, observation_loss = burger_loss(x_N, ground_truth, mask, device)
    dims = (1, 2)
    L_obs = torch.sqrt((observation_loss ** 2).sum(dim=dims)) / (128 * 5)
    ell = -obs_weight * L_obs
    if pde_weight != 0.0:
        L_pde = torch.sqrt((pde_loss ** 2).sum(dim=dims)) / (128 * 128)
        ell = ell - pde_weight * L_pde
    return ell


def generate_burgers_gem(config, n_particles=None, num_steps=None, resample_threshold=0.5,
                          out_path='burger-gem-results.npz', inject_noise=True, scale_guidance=True,
                          apply_guidance=True, lambda_girsanov=1.0):
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
    log_w = torch.zeros(N, dtype=torch.float64, device=device)   # ell_K(x_K) := 0: flat prior at
    # pure noise, a shared additive constant across particles (irrelevant after normalization).
    # No separate terminal correction (note_1.pdf eq. 23 / reconcile.md Sec. 9) is applied: that
    # term corrects the surrogate twist back to a *separately specified* exact final-time
    # likelihood log p(y|xi_T). Here there is no such separate likelihood -- the PDE-residual +
    # sparse-observation loss IS the only notion of "fit to y" this problem has, at every step
    # including the last. So the surrogate is being treated as terminally consistent
    # (phi_0(x_0) = L(x_0)) by construction, not approximated; if a genuinely different terminal
    # criterion is introduced later this omission should be revisited.

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
        if apply_guidance:
            obs_weight = (zeta_obs / 10) if use_pde else zeta_obs
            pde_weight_cur = zeta_pde if use_pde else 0.0
            b_k, L_obs, L_pde = guidance_grad(x_cur, D, ground_truth, selected_index,
                                               obs_weight, pde_weight_cur, use_pde, device)
        else:
            # Skip the guidance backward passes entirely -- pure unconditional GEM, no
            # observation/PDE conditioning at all. b_k=0 makes scale_guidance irrelevant (both
            # conventions reduce to the same drift when there is no guidance to scale).
            b_k = torch.zeros_like(D)
            L_obs = torch.zeros(N, dtype=torch.float64, device=device)
            L_pde = torch.zeros(N, dtype=torch.float64, device=device)
        score = score.detach()
        x_cur = x_cur.detach()

        x_next, z, delta = gem_step(x_cur, score, b_k, sigma_cur, sigma_next, generator=generator,
                                     inject_noise=inject_noise, scale_guidance=scale_guidance)

        # Delta_ell_k = ell_{k-1}(x_{k-1}) - ell_k(x_k) (reconcile.md Sec. 2-3): the telescoping
        # twist-ratio increment that was previously missing from this script's weight entirely.
        # ell_cur reuses L_obs/L_pde already computed above (no extra cost); ell_next needs one
        # extra no-grad (forward-only, no backward) network call at the post-step state, using
        # the guidance schedule appropriate to the NEXT step index (i+1) -- b_k was built from
        # ell at index i, so ell_next must use ell at index i+1 for Delta_ell_k to be the correct
        # single-step telescoping difference.
        if apply_guidance:
            use_pde_next = (i + 1) > 0.8 * K
            obs_weight_next = (zeta_obs / 10) if use_pde_next else zeta_obs
            pde_weight_next = zeta_pde if use_pde_next else 0.0
            with torch.no_grad():
                D_next, _ = denoise(net, x_next, sigma_next)
            ell_next = twist_log_likelihood(D_next, ground_truth, selected_index,
                                             obs_weight_next, pde_weight_next, device)
            ell_cur = -obs_weight * L_obs - pde_weight_cur * L_pde
            delta_ell = ell_next - ell_cur
        else:
            delta_ell = torch.zeros(N, dtype=torch.float64, device=device)

        inc = delta_ell + lambda_girsanov * girsanov_increment(b_k, z, delta)
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

    print(f"N={N} particles, K={K} steps, {n_resample} resample events, "
          f"inject_noise={inject_noise}, scale_guidance={scale_guidance}, "
          f"apply_guidance={apply_guidance}, lambda_girsanov={lambda_girsanov}")
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
             inject_noise=inject_noise,
             scale_guidance=scale_guidance,
             apply_guidance=apply_guidance,
             lambda_girsanov=lambda_girsanov,
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
    parser.add_argument('--no-noise', action='store_true',
                         help=('diagnostic control: zero out the injected Brownian increment in '
                               'gem_step, leaving the delta-scaled score+guidance drift as the '
                               'only update (see smc/scripts_2/proposals/gem.py inject_noise). '
                               'Combine with --resample-threshold 0 for a fully deterministic run.'))
    parser.add_argument('--flat-guidance', action='store_true',
                         help=('diagnostic control: add guidance FLAT (unscaled by delta) instead of '
                               'folding it into the delta-scaled score drift -- mirrors the baseline '
                               'generate_burgers.py convention. See gem_step scale_guidance.'))
    parser.add_argument('--no-guidance', action='store_true',
                         help=('diagnostic control: zero out the guidance term entirely -- pure '
                               'unconditional guided-Euler-Maruyama with no observation/PDE '
                               'conditioning. Reference floor for how the unconditional SDE alone '
                               '(score + noise, no guidance) behaves.'))
    parser.add_argument('--lambda-girsanov', type=float, default=1.0,
                         help=('weight on the Girsanov correction C_k^phi in the SMC increment '
                               'G_k = Delta_ell_k + lambda*C_k^phi (denoising_smc_manuscript/'
                               'notes/reconcile.md Sec. 8). Default 1.0 is the full corrected '
                               'weight; 0.0 reproduces the Millard-style pseudo-bootstrap (PBS) '
                               'weight, Delta_ell_k alone.'))
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    generate_burgers_gem(cfg, n_particles=args.n_particles, num_steps=args.num_steps,
                          resample_threshold=args.resample_threshold, out_path=args.out,
                          inject_noise=not args.no_noise, scale_guidance=not args.flat_guidance,
                          apply_guidance=not args.no_guidance, lambda_girsanov=args.lambda_girsanov)
