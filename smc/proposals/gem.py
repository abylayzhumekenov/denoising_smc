"""Guided Euler--Maruyama (GEM) proposal.

One denoiser forward pass per step. Guidance enters the drift scaled by the step variance
``delta = sigma_cur**2 - sigma_next**2`` -- the theory-consistent placement that makes the
Girsanov kernel ratio exact (see docs/note_1.pdf). Ported from
``smc_archive/scripts_2/proposals/gem.py``; the flat-guidance / no-noise diagnostic switches
were dropped (see docs/vision.md).

Conventions: sigma decreases along the trajectory, so ``delta > 0``; the marginal score is
``(D_theta(x, sigma) - x) / sigma**2`` (Tweedie).
"""

import torch


def denoise(net, x_cur, sigma_cur, class_labels=None):
    """One denoiser call. Returns ``(D, score)`` with ``D`` the denoised estimate and
    ``score = (D - x_cur) / sigma_cur**2``. ``x_cur`` must have ``requires_grad_`` set by the
    caller if a guidance gradient through it is needed."""
    sigma_t = torch.as_tensor(sigma_cur, dtype=torch.float64, device=x_cur.device)
    D = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
    score = (D - x_cur) / (sigma_t ** 2)
    return D, score


def gem_step(x_cur, score, guidance_grad, sigma_cur, sigma_next, generator=None):
    """Advance one guided Euler--Maruyama step:

        x_next = x_cur + delta * (score + guidance_grad) + sqrt(delta) * eps,
        delta  = sigma_cur**2 - sigma_next**2.

    ``x_cur``, ``score``, ``guidance_grad`` are tensors of identical shape ``[N, ...]`` with the
    particle dimension first, passed detached. Returns ``(x_next, z, delta)``: the detached next
    state, the realized Brownian increment ``z = sqrt(delta) * eps``, and the scalar ``delta``.
    """
    delta = float(sigma_cur) ** 2 - float(sigma_next) ** 2
    if delta <= 0:
        raise ValueError(
            f"non-decreasing sigma schedule: sigma_cur={sigma_cur}, sigma_next={sigma_next}")

    eps = torch.randn(x_cur.shape, generator=generator, dtype=x_cur.dtype, device=x_cur.device)
    z = delta ** 0.5 * eps
    x_next = x_cur + delta * (score + guidance_grad) + z
    return x_next.detach(), z.detach(), delta
