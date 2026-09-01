"""Girsanov/TDS incremental weight, ESS, and systematic resampling for the real (PDE) network.

Ported 1:1 from smc/scripts_1/toy_smc.py's validated toy recursion (systematic_resample, and the
`C = -st['grad'] * z - 0.5 * st['grad'] ** 2 * delta` line inside run_smc), generalized from
scalar particles to tensor-valued (multi-pixel) particles by summing the per-pixel contributions,
per docs/note_1.pdf Appendix B eq. (38): for a diagonal covariance Sigma_k = delta * I, the
Euler-Maruyama kernel-ratio log-weight is

    C_k = - b_k^T z_k  -  (1/2) * delta * ||b_k||^2

with the inner products taken over *all* pixel dimensions of one particle (not just a scalar).
This is proven (not merely observed) to equal the exact closed-form Gaussian kernel ratio for the
GEM proposal at every step size -- see smc/scripts_2/check_gem_tds_real_model.py for a direct numerical
check of that identity on the real network, which costs one forward+backward pass and should be
run before trusting any full multi-step result.
"""

import torch


def girsanov_increment(guidance_grad, z, delta):
    """C_k of note_1.pdf eq. (18)/(38), summed over every non-batch dimension.

    guidance_grad, z: tensors [N, ...] (particle dim = 0), already detached.
    delta: python float, the accumulated diffusion Sigma_k for this step.

    Returns a 1-D tensor of shape [N]: one log-weight increment per particle.
    """
    dims = tuple(range(1, guidance_grad.dim()))
    bz = (guidance_grad * z).sum(dim=dims)
    b2 = (guidance_grad ** 2).sum(dim=dims)
    return -bz - 0.5 * delta * b2


def effective_sample_size(log_w):
    """ESS from possibly-unnormalized log-weights (1-D tensor [N])."""
    lw = log_w - log_w.max()
    w = torch.exp(lw)
    w = w / w.sum()
    return float(1.0 / torch.sum(w ** 2))


def systematic_resample_indices(log_w, generator=None):
    """Systematic resampling indices from log-weights (1-D tensor [N]).

    Identical algorithm to smc/scripts_1/toy_smc.py's systematic_resample: single uniform offset,
    N equally-spaced strata, searchsorted on the cumulative normalized weight.
    """
    lw = log_w - log_w.max()
    w = torch.exp(lw)
    w = w / w.sum()
    N = w.shape[0]
    u0 = torch.rand((), generator=generator, dtype=w.dtype, device=w.device)
    u = (u0 + torch.arange(N, dtype=w.dtype, device=w.device)) / N
    cw = torch.cumsum(w, dim=0)
    idx = torch.searchsorted(cw, u)
    return idx.clamp(max=N - 1)
