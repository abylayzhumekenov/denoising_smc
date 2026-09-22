"""Guided Euler--Maruyama (GEM) proposal.

One denoiser forward pass per step. Guidance enters the drift scaled by the step variance
``delta = sigma_cur**2 - sigma_next**2`` -- the theory-consistent placement that makes the
Girsanov kernel ratio exact (see docs/note_1.pdf). Ported from
``smc_archive/scripts_2/proposals/gem.py``; the flat-guidance / no-noise diagnostic switches
were dropped (see docs/vision.md).

Sign conventions (the single easiest bug to introduce when porting -- see note_2 Remark 3.1)::

    nabla_log_p(x, sigma) = (D_theta(x, sigma) - x) / sigma**2   # true score, +grad log p
    EDM's "s_theta"       = (x - D_theta(x, sigma)) / sigma**2   # = -nabla_log_p
    baseline's d_cur      = (x - D_theta) / sigma                # = -sigma * nabla_log_p

``d_cur`` is the probability-flow-ODE slope (docs/note_4, eq. ``pfode``), not the score. Millard
et al. (2026) *print* their score as ``(x - D)/sigma**2`` (i.e. EDM's ``s_theta``) and their GEM
pseudocode as ``(x - D)/sigma``; both are the opposite convention from this module -- do not copy
those formulas literally. This module uses the true score ``+nabla_log_p`` throughout.

Integrator scope: ``gem_step`` is Euler--Maruyama specifically (drift frozen at the left endpoint,
Gaussian kernel with covariance ``delta * I``). ``delta = Sigma_k = int Sigma dtau =
sigma_cur**2 - sigma_next**2`` is *integrator-independent* (exact interval integral, Ito
isometry), but the left-endpoint drift and the ``delta * I`` kernel are EM's. Other SDE
integrators (Heun-SDE, SOSaG, ...) must supply their own transition kernel -- and, for the weight,
their own kernel ratio or note_1's potential form ``V_k`` (any quadrature, Hessian required). A
deterministic proposal (probability-flow ODE) is a different object: its guidance coefficient is
``sigma * b`` (half of ``Sigma * b``), and Girsanov does not apply.

Conventions: sigma decreases along the trajectory, so ``delta > 0``.
"""

import torch


def denoise(net, x_cur, sigma_cur, class_labels=None):
    """One denoiser call. Returns ``(D, nabla_log_p)`` with ``D`` the denoised estimate and
    ``nabla_log_p = (D - x_cur) / sigma_cur**2`` (the true score, Tweedie). ``x_cur`` must have
    ``requires_grad_()`` set by the caller if a guidance gradient through it is needed."""
    sigma_t = torch.as_tensor(sigma_cur, dtype=torch.float64, device=x_cur.device)
    D = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
    nabla_log_p = (D - x_cur) / (sigma_t ** 2)
    return D, nabla_log_p


def gem_step(x_cur, nabla_log_p, guidance_grad, sigma_cur, sigma_next, generator=None):
    """Advance one guided Euler--Maruyama step (EM only; see module docstring):

        x_next = x_cur + delta * (nabla_log_p + guidance_grad) + sqrt(delta) * eps,
        delta  = sigma_cur**2 - sigma_next**2.

    ``x_cur``, ``nabla_log_p``, ``guidance_grad`` are tensors of identical shape ``[N, ...]``
    with the particle dimension first, passed detached. Returns ``(x_next, z, delta)``: the
    detached next state, the realized Brownian increment ``z = sqrt(delta) * eps``, and the
    scalar ``delta``.
    """
    delta = float(sigma_cur) ** 2 - float(sigma_next) ** 2
    if delta <= 0:
        raise ValueError(
            f"non-decreasing sigma schedule: sigma_cur={sigma_cur}, sigma_next={sigma_next}")

    eps = torch.randn(x_cur.shape, generator=generator, dtype=x_cur.dtype, device=x_cur.device)
    z = delta ** 0.5 * eps
    x_next = x_cur + delta * (nabla_log_p + guidance_grad) + z
    return x_next.detach(), z.detach(), delta
