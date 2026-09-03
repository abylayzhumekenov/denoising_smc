"""Guided Euler-Maruyama (GEM) proposal for the real (PDE) network, per docs/idea.md Sec. 3.1
and docs/note_1.pdf Sec. 2.4 / Appendix B.

This is deliberately the *simplest* proposal in the idea.md progression (GEM before Heun-SDE):
one network forward + one backward per step, exact closed-form Girsanov/TDS weight (no Hessian,
no Hutchinson estimator -- see smc/hutchinson_findings.md for why that route is deprioritized).

Ported 1:1 from the validated toy-model recursion in smc/scripts_1/toy_smc.py
(`run_smc(..., proposal='em')`, lines ~248-264): same recurrence structure,
same variable roles (score, guidance grad, delta, z), just tensor-valued instead of scalar.

Convention (matches toy_smc.py and note_1.pdf Sec. 2.1/2.4): sigma decreases from sigma_max to
sigma_min as the reverse process advances; `sigma_cur` is the more-noisy (larger) sigma of the
step, `sigma_next` the less-noisy (smaller) one, and
    delta = sigma_cur**2 - sigma_next**2 > 0
is the accumulated diffusion (Sigma_k of eq. 12/27).

Score convention: EDM's network returns the denoiser D_theta(x, sigma) directly (`net(x, sigma)`),
and by Tweedie's identity the marginal score is
    score(x, sigma) = (D_theta(x, sigma) - x) / sigma**2.
Note this is *not* the same normalization as the existing deterministic ODE solver's `d_cur =
(x - D_theta) / sigma` (divided by sigma, not sigma**2) used in scripts/generate_burgers.py and
its siblings -- that is the probability-flow-ODE slope, not the score. Relation: score = -d_cur /
sigma. Keep this straight; it is the single easiest sign/scale bug to introduce when porting
between the deterministic-ODE scripts and this SDE proposal.
"""

import torch


def denoise(net, x_cur, sigma_cur, class_labels=None):
    """One network call. x_cur must already have requires_grad_(True) set by the caller if a
    guidance gradient w.r.t. x_cur will be needed downstream (autograd.grad traces through this).

    `sigma_cur` may be a plain python float or a tensor -- coerced to a tensor here since
    EDMPrecond.forward calls `sigma.to(torch.float32)` internally and has no float fallback
    (this bit the first draft of this module: passing a bare float raised AttributeError deep
    inside net.forward, since a python float has no `.to()`). Mirrors net.round_sigma's own
    `torch.as_tensor(sigma)` defensiveness.

    Returns D (denoised estimate, float64) and score = (D - x_cur) / sigma_cur**2.
    """
    sigma_t = torch.as_tensor(sigma_cur, dtype=torch.float64, device=x_cur.device)
    D = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
    score = (D - x_cur) / (sigma_t ** 2)
    return D, score


def gem_step(x_cur, score, guidance_grad, sigma_cur, sigma_next, generator=None,
             inject_noise=True, scale_guidance=True):
    """Advance one guided Euler-Maruyama step (docs/idea.md eq. in Sec. 3.1; note_1.pdf eq. 12).

    x_cur, score, guidance_grad: tensors of identical shape [N, ...], particle dim = 0.
    sigma_cur, sigma_next: python floats or 0-dim tensors (same schedule for every particle).

    inject_noise: if False, skips adding the Brownian increment (z is returned as all-zeros
    instead of sqrt(delta)*eps). This turns the update into the deterministic mean-drift
    recursion x_next = x_cur + delta*(score + guidance_grad) -- same sigma**2-based score and
    same delta-SCALED guidance as the real GEM step, just with the SDE's noise term switched
    off. Used as a diagnostic control to separate "does the injected noise cause the GEM-vs-Heun
    error gap" from "does the delta-scaled guidance convention itself cause it" -- see
    scripts/generate_burgers_gem.py's --no-noise flag.

    scale_guidance: if True (default), guidance_grad is folded into the score before scaling by
    delta -- x_next = x_cur + delta*(score + guidance_grad) + z -- the Bayesian score-composition
    convention (guidance is treated as part of the total score, so it inherits the SDE's
    step-size weighting same as the prior score). If False, guidance is instead added FLAT,
    unscaled by delta -- x_next = x_cur + delta*score + guidance_grad + z -- mirroring
    scripts/generate_burgers.py's baseline convention (a fixed-size post-hoc nudge, same
    magnitude every step regardless of how much sigma-time the step covers). Tests whether
    reusing the baseline's flat-guidance convention, with the same zeta_obs/zeta_pde it was
    tuned for, recovers Heun-level accuracy inside the noisy SDE.

    Returns (x_next, z, delta):
      x_next  -- detached, ready to be the next step's x_cur
      z       -- the *realized* Brownian increment sqrt(delta)*eps actually used (needed by the
                 Girsanov weight -- see smc/scripts_2/weightings/girsanov.py girsanov_increment),
                 or all-zeros when inject_noise=False
      delta   -- the scalar accumulated diffusion Sigma_k for this step (needed by the weight too)

    x_cur, score, and guidance_grad must NOT carry gradient history into this call (detach them
    first) -- gem_step only advances state, it does not itself need autograd.
    """
    delta = float(sigma_cur) ** 2 - float(sigma_next) ** 2
    if delta <= 0:
        raise ValueError(f"non-decreasing sigma schedule: sigma_cur={sigma_cur}, sigma_next={sigma_next}")

    if inject_noise:
        eps = torch.randn(x_cur.shape, generator=generator, dtype=x_cur.dtype, device=x_cur.device)
        z = (delta ** 0.5) * eps
    else:
        z = torch.zeros_like(x_cur)

    if scale_guidance:
        x_next = x_cur + delta * (score + guidance_grad) + z
    else:
        x_next = x_cur + delta * score + guidance_grad + z
    return x_next.detach(), z.detach(), delta
