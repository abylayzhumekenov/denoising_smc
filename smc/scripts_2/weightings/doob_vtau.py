"""Computation of V_tau(x), the Doob-transform potential from "Consistent Particle Guided
Denoising" (draft manuscript, not checked into this repo).

V_tau(x) = d(ell)/d(tau) + H_tau(x)
H_tau(x) = b_ap(x,tau) . grad_ell(x) + (1/2)*a_bar(tau)*Laplacian(ell)(x) - (1/2)*a_bar(tau)*||grad_ell(x)||^2

Two simplifications used here, adapting the manuscript's math to this EDM-based codebase
(see conversation notes / smc/scripts_2/hutchinson_findings.md for the derivation):

1. b_ap(x,tau) = a_bar(tau) * [s_theta(x,tau) + grad_ell(x)] (manuscript eq., page 4), so
   H_tau collapses to a_bar(tau) * [s_theta . grad_ell + (1/2)||grad_ell||^2 + (1/2)*Laplacian(ell)] --
   no separate drift model is needed, just the score (already available from the same denoiser
   call that computes ell) and grad_ell (already computed via autograd).

2. This codebase's EDM convention identifies physical time t with the noise level sigma_t
   directly (sigma(t) = t, confirmed: scripts/generate_burgers.py never has a variable
   distinct from sigma_t). Sampling time is tau = T - t with T = sigma_max, so
   d(ell)/d(tau) = -d(ell)/d(sigma_t) (chain rule, an affine flip -- NOT the same sign as the
   raw autodiff-on-sigma_t derivative computed in the earlier ad-hoc exploration).

a_bar(tau) = 2*sigma_t follows from a(t) := d(sigma(t)^2)/dt = 2*sigma_t under sigma(t) = t.
"""

from typing import Optional

import torch
from torch import Tensor

from smc.scripts_2.weightings.hutchinson import hutchinson_hvp_probes


def compute_v_tau_terms(ell_fn, x: Tensor, sigma_t: Tensor, num_probes: int,
                         generator: Optional[torch.Generator] = None) -> dict:
    """Compute ell, grad_ell, d(ell)/d(tau), the Hutchinson-estimated Laplacian, and the
    resulting H_tau(x) / V_tau(x), from a single fused forward+backward pass.

    `ell_fn(x, sigma_t)` must return (ell, s_theta) where `ell` is the scalar log-likelihood
    surrogate and `s_theta` is the score s_theta(x, sigma_t) = (x - D_theta(x,sigma_t))/sigma_t^2,
    both built from the same denoiser call -- see smc/scripts_2/models/burgers.py's burgers_ell_fn for the Burgers instance.
    """
    x = x.detach().requires_grad_(True)
    sigma_t = sigma_t.detach().requires_grad_(True)

    ell, s_theta = ell_fn(x, sigma_t)
    grad_x, grad_sigma = torch.autograd.grad(ell, (x, sigma_t), create_graph=True)

    samples = hutchinson_hvp_probes(grad_x, x, num_probes, generator=generator)
    laplacian_mean = samples.mean()
    laplacian_sem = (samples.std(unbiased=True) / (num_probes ** 0.5)) if num_probes > 1 else None

    a_bar = 2 * sigma_t.detach()
    dtau_ell = -grad_sigma.detach()
    # Despite the name, this is a_bar * (s_theta . grad_ell) only -- the *score*, not the full
    # drift b_ap = a_bar*(s_theta + grad_ell). The b_ap-dotted-with-grad_ell term would be
    # drift_dot_grad + a_bar*||grad_ell||^2; that extra a_bar*||grad_ell||^2 piece is already
    # folded into grad_norm_sq_term below (it's why grad_norm_sq_term is +1/2, not -1/2, relative
    # to H_tau's original, pre-substitution definition -- see the module docstring).
    drift_dot_grad = a_bar * (s_theta.detach() * grad_x.detach()).sum()
    grad_norm_sq_term = 0.5 * a_bar * (grad_x.detach() ** 2).sum()
    laplacian_term = 0.5 * a_bar * laplacian_mean
    H_tau = drift_dot_grad + grad_norm_sq_term + laplacian_term
    V_tau = dtau_ell + H_tau

    return {
        "ell": ell.detach().item(),
        "grad_ell_norm": grad_x.detach().norm().item(),
        "dell_dsigma": grad_sigma.detach().item(),
        "dell_dtau": dtau_ell.item(),
        "laplacian_mean": laplacian_mean.item(),
        "laplacian_sem": laplacian_sem.item() if laplacian_sem is not None else None,
        "a_bar": a_bar.item(),
        "drift_dot_grad": drift_dot_grad.item(),
        "grad_norm_sq_term": grad_norm_sq_term.item(),
        "laplacian_term": laplacian_term.item(),
        "H_tau": H_tau.item(),
        "V_tau": V_tau.item(),
        "samples": samples.detach().tolist(),
    }
