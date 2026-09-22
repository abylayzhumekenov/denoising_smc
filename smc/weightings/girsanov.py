"""Girsanov / TDS incremental weight for the GEM proposal.

For a Gaussian proposal with covariance ``delta * I`` the exact per-step kernel log-ratio is

    C_k = -b_k^T z_k - (1/2) * delta_k * ||b_k||^2,

summed over every non-batch dimension. The incremental weight is
``G_k = Delta_ell_k + lambda_girs * C_k`` (lambda=1 corrected Girsanov, lambda=0 pseudo-bootstrap).
Ported from ``smc_archive/scripts_2/weightings/girsanov.py``.

This is the *exact* discrete kernel ratio only for the Euler--Maruyama proposal, whose unguided
and guided kernels share covariance ``delta_k * I`` and differ only in mean by ``delta_k * b_k``
(note_1 App. B; it uses ``z_k ~ N(0, delta_k * I)``). For a non-EM integrator this expression is
at best asymptotically valid as ``K -> infinity``; either derive that integrator's exact kernel
ratio or use note_1's potential increment ``V_k`` instead.
"""

import torch


def girsanov_increment(guidance_grad, z, delta, lam=1.0):
    """``lam * C_k`` where ``C_k`` is the exact GEM kernel log-ratio.

    ``guidance_grad`` and ``z`` are tensors ``[N, ...]`` (particle dimension first); ``delta`` is
    a python float. Returns a 1-D tensor ``[N]``.
    """
    dims = tuple(range(1, guidance_grad.dim()))
    prod = guidance_grad * z
    sq = guidance_grad ** 2
    if dims:
        bz = prod.sum(dim=dims)
        b2 = sq.sum(dim=dims)
    else:
        # Particle dim is the only dim (scalar per particle): reduce nothing. Note ``sum(dim=())``
        # would reduce *all* dims in current PyTorch, so this case must be handled explicitly.
        bz, b2 = prod, sq
    return lam * (-bz - 0.5 * delta * b2)
