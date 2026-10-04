"""Stochastic Heun (2-stage, second-order) proposal for the guided reverse SDE.

Integrates the *same* guided reverse SDE as ``gem.py`` (Euler--Maruyama) to higher order in the
drift, reusing the *same* additive Brownian increment ``z ~ N(0, delta*I)`` in both stages::

    drift_cur  = nabla_log_p(x_k, sigma_k)      + b(x_k, sigma_k)
    x_pred     = x_k + delta * drift_cur + z
    drift_pred = nabla_log_p(x_pred, sigma_{k+1}) + b(x_pred, sigma_{k+1})
    x_next     = x_k + 0.5 * delta * (drift_cur + drift_pred) + z

Stage 1 (predictor) exists only to evaluate the drift at the far end of the step; the noise is *not*
redrawn. The reverse SDE here has state-independent (additive) noise, so the trapezoidal rule applies
to the deterministic drift alone and is genuinely second order in the drift (the Milstein correction
vanishes). ``delta = sigma_cur**2 - sigma_next**2`` is the shared, integrator-independent step measure
(Ito isometry); the guidance coefficient is unchanged from EM.

Weighting is *not* handled here. The incremental weight stays the same as EM's (``girsanov.py``),
with ``b_k`` evaluated at the step's *start* and the same ``z``. For this non-EM integrator that
expression is asymptotically consistent as ``K -> inf`` rather than finite-step exact: the guided and
unguided transition means differ by ``delta * b_k + O(delta**2)``, so EM's kernel ratio is the correct
ratio up to ``O(delta**3/2)`` -- below the scheme's own discretization error. See ``docs/vision.md``
and ``docs/note_1.pdf``.

This module is pure (no network): the caller owns the two denoiser/guidance evaluations and passes
the two drift tensors in.
"""


def heun_predict(x_cur, drift_cur, z, delta):
    """Stage 1: left-endpoint (Euler) prediction ``x_cur + delta*drift_cur + z``."""
    return x_cur + delta * drift_cur + z


def heun_correct(x_cur, drift_cur, drift_pred, z, delta):
    """Stage 2: trapezoidal corrector reusing the *same* ``z``.

    ``x_cur + 0.5 * delta * (drift_cur + drift_pred) + z``. The difference from an EM step on the
    same ``z`` is exactly ``0.5 * delta * (drift_pred - drift_cur)``.
    """
    return x_cur + 0.5 * delta * (drift_cur + drift_pred) + z
