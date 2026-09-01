"""Validate Millard et al. (2025)'s GEM/pBS/SOSaG lambda-rho SMC weight (as derived in
denoising_smc_manuscript/notes/note_1.tex) against the same closed-form 1D Gaussian-mixture
toy model used for the V_tau/Doob-transform scheme in check_toy_mixture.py -- so the two
schemes can be compared directly on identical ground truth.

Key structural differences from the V_tau scheme (see conversation notes):
  - No Laplacian / Hutchinson term at all: the Girsanov correction only needs the guidance
    gradient b and the REALIZED Brownian increment xi from the same step -- much cheaper.
  - The telescoping likelihood-ratio term is evaluated at the ACTUAL simulated states
    (x_cur, x_next), not a frozen pre-step state -- an exact telescoping, not an approximation.
  - Uses the exact delta_k = sigma_cur^2 - sigma_next^2 convention natively (that's what the
    derivation in note_1.tex is built on), not an add-on.
  - A one-time boundary correction (mirroring the V_tau scheme's e^{h_0(Z_0)} term) is needed
    because the exact telescoping sum leaves a leftover -log(tilde_p(y|x_K; sigma_max)) term
    that the target weight doesn't want; adding it back once before the step loop cancels it.

lambda=1, rho=1 is the exact Girsanov-corrected weight ("GEM-TDS"); lambda=0 drops the
correction entirely, keeping only the tempered telescoping-likelihood term ("pBS/SOSaG").
"""

import torch

from smc.scripts_1.toy_mixture import GaussianMixture, sigma_schedule, approx_guidance_grad
from smc.scripts_1.check_toy_mixture import systematic_resample


def simulate_millard(mixture, n_particles, sigmas, y, r, generator, lam=1.0, rho=1.0,
                      accumulate_weight=True, ess_threshold=0.5):
    x = torch.randn(n_particles, dtype=torch.float64, generator=generator) * sigmas[0]
    log_weight = torch.zeros(n_particles, dtype=torch.float64)
    n_resamples = 0

    if accumulate_weight:
        log_weight += rho * mixture.log_approx_likelihood(x, y, sigmas[0], r)  # boundary correction

    for i in range(len(sigmas) - 1):
        sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
        delta = sigma_cur ** 2 - sigma_next ** 2  # exact delta_k, native to this scheme

        s_theta = (x - mixture.denoise(x, sigma_cur)) / sigma_cur ** 2
        b = approx_guidance_grad(mixture, x, sigma_cur, y, r)

        xi = torch.randn(n_particles, dtype=torch.float64, generator=generator)
        x_next = x + delta * (-s_theta + b) + torch.sqrt(delta) * xi

        if accumulate_weight:
            ell_cur = mixture.log_approx_likelihood(x, y, sigma_cur, r)
            ell_next = mixture.log_approx_likelihood(x_next, y, sigma_next, r)
            C = -torch.sqrt(delta) * b * xi - 0.5 * delta * b ** 2
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
        print(f'  [Millard lambda={lam} rho={rho}] triggered {n_resamples} resamples over {len(sigmas)-1} steps')

    return x, log_weight


def main():
    mixture = GaussianMixture(w=[0.5, 0.5], mu=[-3.0, 3.0], var=[1.0, 1.0])
    y, r = 0.5, 1.0
    n_particles = 200000
    sigmas = sigma_schedule(sigma_min=0.02, sigma_max=20.0, num_steps=1000, rho=7.0)
    true_posterior = mixture.posterior(y, r)
    true_mean = (true_posterior.w * true_posterior.mu).sum().item()
    print(f'true posterior mean: {true_mean:.4f}')

    torch.manual_seed(7)
    gen = torch.Generator().manual_seed(7)
    x, log_w = simulate_millard(mixture, n_particles, sigmas, y, r, gen, lam=1.0, rho=1.0)
    w = torch.softmax(log_w, dim=0)
    mean = (w * x).sum().item()
    ess = (1.0 / (w ** 2).sum()).item()
    print(f'GEM-TDS (lambda=1, rho=1): mean={mean:.4f}  ESS={ess:.0f}/{n_particles}')


if __name__ == '__main__':
    main()
