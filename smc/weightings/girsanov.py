"""Girsanov / TDS incremental weight for the GEM proposal.

For a Gaussian proposal with covariance ``delta * I`` the exact per-step kernel log-ratio is

    C_k = -b_k^T z_k - (1/2) * delta_k * ||b_k||^2,

summed over every non-batch dimension. The incremental weight is
``G_k = Delta_ell_k + lambda_girs * C_k`` (lambda=1 corrected Girsanov, lambda=0 pseudo-bootstrap).
Ported from ``smc_archive/scripts_2/weightings/girsanov.py``.
"""

import torch


def girsanov_increment(guidance_grad, z, delta, lam=1.0):
    """``lam * C_k`` where ``C_k`` is the exact GEM kernel log-ratio.

    ``guidance_grad`` and ``z`` are tensors ``[N, ...]`` (particle dimension first); ``delta`` is
    a python float. Returns a 1-D tensor ``[N]``.
    """
    dims = tuple(range(1, guidance_grad.dim()))
    bz = (guidance_grad * z).sum(dim=dims)
    b2 = (guidance_grad ** 2).sum(dim=dims)
    return lam * (-bz - 0.5 * delta * b2)
