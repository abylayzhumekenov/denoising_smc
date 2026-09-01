"""Validate the V_tau / Girsanov-SMC correction against a closed-form 1D Gaussian-mixture
ground truth (smc/scripts_1/toy_mixture.py). Compares four sample sets:

  (a) exact posterior samples       -- direct ancestral draw, our ground truth
  (b) exact-guided diffusion        -- SDE simulated with the TRUE p(y|x_sigma) drift
  (c) approx-guided, uncorrected    -- SDE simulated with the plug-in tilde_h_tau drift
  (d) approx-guided, SMC-corrected  -- (c)'s particles, self-normalized-importance-reweighted
                                        by the Girsanov/V_tau weight

If the math and code are right, (d) should sit much closer to (a)/(b) than (c) does.

Sampling convention: physical time / decreasing sigma (matches scripts/generate_burgers.py's
Heun sampler), not the paper's increasing sampling-time tau -- i.e. we loop sigma from
sigma_max down to sigma_min, exactly like the real EDM sampler, with step size
Delta = sigma_cur - sigma_next > 0 playing the role of Delta_tau_k.

SMC weight discretization matches the paper's own telescoping scheme (section 1.3), not a
naive left-endpoint quadrature of the whole V_tau: the d(ell)/d(tau) part is handled via the
EXACT ell(z;sigma_next) - ell(z;sigma_cur) difference at the frozen pre-step state (no
derivative needed), and only the H_tau part uses left-endpoint quadrature. See conversation
notes for the derivation of why this differs from -- and is more faithful than -- directly
quadrature-integrating d(ell)/d(tau) via autodiff.
"""

import torch

from smc.scripts_1.toy_mixture import (
    GaussianMixture, sigma_schedule, exact_guidance_grad, approx_guidance_grad, batched_h_tau,
)


def systematic_resample(w, generator=None):
    """Low-variance systematic resampling: one random offset, N evenly-spaced draws."""
    n = w.shape[0]
    u0 = torch.rand(1, generator=generator).item()
    positions = (u0 + torch.arange(n, dtype=torch.float64)) / n
    cumsum = torch.cumsum(w, dim=0)
    cumsum[-1] = 1.0  # guard against floating-point shortfall
    return torch.searchsorted(cumsum, positions)


def integrate_h_tau(mixture, x, sigma_cur, sigma_next, y, r, n_substeps):
    """Approximate integral_{sigma_next}^{sigma_cur} H_tau(x, sigma) d(sigma) with a midpoint-rule
    quadrature of n_substeps sub-intervals, x held FROZEN at its pre-step value throughout -- no
    new SDE/Brownian steps, no resampling impact, just a better estimate of the deterministic
    time-integral of H_tau at a fixed state. n_substeps=1 reduces to the original left-endpoint
    rule. This targets only the H_tau quadrature error; the d(ell)/d(tau) part already has zero
    quadrature error (the ell-difference trick is exact given the frozen-state assumption), so
    refining it further wouldn't help -- only refining the actual particle-propagation step size
    (and hence resampling frequency) would address the frozen-state assumption itself.
    """
    sub_delta = (sigma_cur - sigma_next) / n_substeps
    total = torch.zeros_like(x)
    for j in range(n_substeps):
        sigma_j = sigma_cur - (j + 0.5) * sub_delta  # midpoint of sub-interval j
        total = total + batched_h_tau(mixture, x, sigma_j, y, r) * sub_delta
    return total


def simulate(mixture, n_particles, sigmas, y, r, guidance, generator, accumulate_weight=False,
             ess_threshold=0.5, h_tau_substeps=1):
    """ess_threshold: resample whenever ESS/N drops below this fraction (standard adaptive-SMC
    resampling trigger). Only meaningful when accumulate_weight=True; None disables resampling
    (single terminal importance weight, for comparison). h_tau_substeps: see integrate_h_tau."""
    x = torch.randn(n_particles, dtype=torch.float64, generator=generator) * sigmas[0]
    log_weight = torch.zeros(n_particles, dtype=torch.float64)
    n_resamples = 0

    if accumulate_weight:
        log_weight += mixture.log_approx_likelihood(x, y, sigmas[0], r)  # the e^{h_0(Z_0)} correction term

    for i in range(len(sigmas) - 1):
        sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
        delta = sigma_cur - sigma_next

        s_theta = (x - mixture.denoise(x, sigma_cur)) / sigma_cur ** 2
        if guidance == 'exact':
            grad_ell = exact_guidance_grad(mixture, x, sigma_cur, y, r)
        else:
            grad_ell = approx_guidance_grad(mixture, x, sigma_cur, y, r)

        # Exact-delta convention (Millard et al. 2025 / note_1.tex): delta_var = sigma_cur^2 -
        # sigma_next^2 is the EXACT integral of a(t)=2*sigma(t) over this step (fundamental
        # theorem of calculus, no approximation), equivalently the midpoint-rule evaluation of
        # a_bar since a_bar is linear in sigma -- strictly more accurate than the left-endpoint
        # a_bar(sigma_cur)*delta used previously. Replaces both the drift scale and the noise
        # variance (the Wiener quadratic variation over the step is exactly delta_var too).
        delta_var = sigma_cur ** 2 - sigma_next ** 2
        drift_direction = -s_theta + grad_ell  # sign flip -- see toy_mixture.batched_h_tau
        xi = torch.randn(n_particles, dtype=torch.float64, generator=generator)
        x_next = x + drift_direction * delta_var + torch.sqrt(delta_var) * xi

        if accumulate_weight:
            ell_cur = mixture.log_approx_likelihood(x, y, sigma_cur, r)
            ell_next = mixture.log_approx_likelihood(x, y, sigma_next, r)
            h_integral = integrate_h_tau(mixture, x, sigma_cur, sigma_next, y, r, h_tau_substeps)
            log_weight += (ell_next - ell_cur) + h_integral

        x = x_next

        if accumulate_weight and ess_threshold is not None:
            w = torch.softmax(log_weight, dim=0)
            ess = 1.0 / (w ** 2).sum()
            if ess.item() < ess_threshold * n_particles:
                idx = systematic_resample(w, generator=generator)
                x = x[idx]
                log_weight = torch.zeros(n_particles, dtype=torch.float64)  # resampled -> uniform again
                n_resamples += 1

    if accumulate_weight and ess_threshold is not None:
        print(f'  [{guidance} guidance, resampling] triggered {n_resamples} resamples over {len(sigmas)-1} steps')

    return x, log_weight


def summarize(name, x, weights=None):
    if weights is None:
        w = torch.ones_like(x) / x.shape[0]
    else:
        w = torch.softmax(weights, dim=0)
    mean = (w * x).sum()
    var = (w * (x - mean) ** 2).sum()
    p_neg = (w * (x < 0).double()).sum()  # proxy for "posterior weight on the mu=-3 component"
    ess = 1.0 / (w ** 2).sum()
    print(f'{name:32s} mean={mean.item():+.4f}  std={var.sqrt().item():.4f}  '
          f'P(x<0)={p_neg.item():.4f}  ESS={ess.item():.1f}/{x.shape[0]}')


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    mixture = GaussianMixture(w=[0.5, 0.5], mu=[-3.0, 3.0], var=[1.0, 1.0])
    y, r = 0.5, 1.0
    n_particles = 20000
    sigmas = sigma_schedule(sigma_min=0.02, sigma_max=20.0, num_steps=1000, rho=7.0)

    true_posterior = mixture.posterior(y, r)
    print(f'Exact posterior mixture weights: {true_posterior.w.tolist()}  '
          f'means: {true_posterior.mu.tolist()}  vars: {true_posterior.var.tolist()}')
    print()

    post_samples = true_posterior.sample_x0(n_particles, generator=gen)
    summarize('(a) exact posterior (ground truth)', post_samples)

    exact_samples, _ = simulate(mixture, n_particles, sigmas, y, r, guidance='exact', generator=gen)
    summarize('(b) exact-guided SDE', exact_samples)

    approx_samples, log_w = simulate(mixture, n_particles, sigmas, y, r, guidance='approx',
                                      generator=gen, accumulate_weight=True, ess_threshold=None)
    summarize('(c) approx-guided, uncorrected', approx_samples)
    summarize('(d) approx-guided, SMC-corrected, no resampling', approx_samples, weights=log_w)

    resampled_x, resampled_log_w = simulate(mixture, n_particles, sigmas, y, r, guidance='approx',
                                             generator=gen, accumulate_weight=True, ess_threshold=0.5)
    summarize('(e) approx-guided, SMC-corrected, resampled', resampled_x, weights=resampled_log_w)


if __name__ == '__main__':
    main()
