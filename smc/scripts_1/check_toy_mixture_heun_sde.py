"""Validate the Heun-SDE proposal (idea.md Sec 3.3, "New -- This Work") against the same
closed-form 1D Gaussian-mixture toy model used for GEM-TDS/GEM-Girsanov (check_toy_mixture_millard.py).

Heun-SDE is a 2-stage stochastic Runge-Kutta integrator: an Euler prediction, then a Heun
(trapezoidal) correction of the DETERMINISTIC drift only, reusing the SAME Brownian increment
z in both stages. Because the noise structure is identical to GEM's, the Girsanov correction
C_k is unchanged -- same formula, same b_k evaluated at the step's starting point (x_k, sigma_k),
not the Heun-corrected midpoint. Only the propagation step differs from GEM.

Success criteria (idea.md Sec 5.5), tested here on ground truth we actually have:
  - Heun-SDE-Girsanov error <= GEM-TDS error (higher-order drift should reduce bias)
  - Heun-SDE-Girsanov ESS >= GEM-TDS ESS (no penalty for the 2nd-order proposal)
"""

import torch

from smc.scripts_1.toy_mixture import approx_guidance_grad
from smc.scripts_1.check_toy_mixture import systematic_resample


def simulate_heun_sde(mixture, n_particles, sigmas, y, r, generator, lam=1.0, rho=1.0,
                       accumulate_weight=True, ess_threshold=0.5):
    x = torch.randn(n_particles, dtype=torch.float64, generator=generator) * sigmas[0]
    log_weight = torch.zeros(n_particles, dtype=torch.float64)
    n_resamples = 0

    if accumulate_weight:
        log_weight += rho * mixture.log_approx_likelihood(x, y, sigmas[0], r)  # boundary correction

    for i in range(len(sigmas) - 1):
        sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
        delta = sigma_cur ** 2 - sigma_next ** 2

        a_k = (mixture.denoise(x, sigma_cur) - x) / sigma_cur ** 2  # true score, idea.md convention
        b_k = approx_guidance_grad(mixture, x, sigma_cur, y, r)

        xi = torch.randn(n_particles, dtype=torch.float64, generator=generator)

        # Stage 1: Euler prediction
        x_pred = x + delta * (a_k + b_k) + torch.sqrt(delta) * xi

        if sigma_next > 0:
            # Stage 2: Heun correction (averaged drift at start and predicted point), SAME xi
            a_pred = (mixture.denoise(x_pred, sigma_next) - x_pred) / sigma_next ** 2
            b_pred = approx_guidance_grad(mixture, x_pred, sigma_next, y, r)
            x_next = x + 0.5 * delta * (a_k + b_k + a_pred + b_pred) + torch.sqrt(delta) * xi
        else:
            # Score is singular at sigma=0 (division by sigma_next^2); the real EDM sampler
            # never evaluates it there either -- fall back to the Euler prediction for the
            # one final step that reaches sigma=0 exactly.
            x_next = x_pred

        if accumulate_weight:
            ell_cur = mixture.log_approx_likelihood(x, y, sigma_cur, r)
            ell_next = mixture.log_approx_likelihood(x_next, y, sigma_next, r)
            # Same Girsanov C_k as GEM: b_k at the step's START point, same xi -- idea.md Sec 3.3
            C = -torch.sqrt(delta) * b_k * xi - 0.5 * delta * b_k ** 2
            log_weight += rho * (ell_next - ell_cur) + lam * C

        x = x_next

        if accumulate_weight and ess_threshold is not None:
            w = torch.softmax(log_weight, dim=0)
            ess = 1.0 / (w ** 2).sum()
            if ess.item() < ess_threshold * n_particles:
                idx = systematic_resample(w, generator=generator)
                x = x[idx]
                log_weight = torch.zeros(n_particles, dtype=torch.float64)
                n_resamples += 1

    if accumulate_weight and ess_threshold is not None:
        print(f'  [Heun-SDE lambda={lam} rho={rho}] triggered {n_resamples} resamples over {len(sigmas)-1} steps')

    return x, log_weight
