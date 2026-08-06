"""Hutchinson trace estimator for the Laplacian term in the Doob/Ito-generator SMC weight.

Background: docs/idea.md's Girsanov correction (C_k) needs only the guidance gradient and the
already-sampled noise -- no second derivatives. An alternative Doob-transform / Ito's-formula
derivation of the same change-of-measure ("Consistent Particle Guided Denoising", draft not yet
checked into this repo) instead requires the Laplacian (trace of the Hessian) of the observation
log-likelihood, which is infeasible to compute exactly at PDE-field dimensionality. This module
estimates it stochastically via Hutchinson's estimator, using Rademacher probe vectors and
reverse-over-reverse double-backward (torch.autograd.grad with create_graph=True).

See smc/hutchinson_findings.md for measured cost/variance on the real pretrained Burgers model.
"""

from typing import Optional

import torch
from torch import Tensor


def hutchinson_hvp_probes(grad_output: Tensor, x: Tensor, num_probes: int,
                           generator: Optional[torch.Generator] = None) -> Tensor:
    """Draw `num_probes` independent Hutchinson samples v^T H v for the Hessian H of the scalar
    whose gradient w.r.t. `x` (computed with create_graph=True) is `grad_output`.

    Each sample costs one more reverse-mode backward pass through the existing graph -- no
    additional forward passes through the denoiser network are needed.
    """
    samples = torch.empty(num_probes, dtype=x.dtype, device=x.device)
    for i in range(num_probes):
        v = torch.randint(0, 2, x.shape, dtype=x.dtype, device=x.device, generator=generator) * 2 - 1
        hvp, = torch.autograd.grad((grad_output * v).sum(), x, retain_graph=(i < num_probes - 1))
        samples[i] = (v * hvp).sum()
    return samples


def estimate_laplacian(fn, x: Tensor, num_probes: int,
                        generator: Optional[torch.Generator] = None) -> dict:
    """Estimate the Laplacian (trace of the Hessian) of scalar function `fn` at `x`.

    Returns a dict with `value` (fn(x)), `grad` (gradient of fn at x), `laplacian_mean`
    (Hutchinson point estimate), `laplacian_sem` (standard error of the mean, None if
    num_probes == 1), and `samples` (raw per-probe draws, for diagnostics).

    Cost/variance are strongly problem- and noise-level-dependent -- see
    smc/hutchinson_findings.md before choosing num_probes for a real run.
    """
    x = x.detach().requires_grad_(True)
    value = fn(x)
    grad_x, = torch.autograd.grad(value, x, create_graph=True)
    samples = hutchinson_hvp_probes(grad_x, x, num_probes, generator=generator)

    mean = samples.mean()
    sem = (samples.std(unbiased=True) / (num_probes ** 0.5)) if num_probes > 1 else None

    return {
        "value": value.detach(),
        "grad": grad_x.detach(),
        "laplacian_mean": mean.item(),
        "laplacian_sem": sem.item() if sem is not None else None,
        "samples": samples.detach().tolist(),
    }
