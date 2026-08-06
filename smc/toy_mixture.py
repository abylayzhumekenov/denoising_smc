"""1D Gaussian-mixture toy model for validating the Doob-transform / Girsanov SMC weight
(V_tau, H_tau -- see smc/v_tau.py) against a closed-form ground truth. Everything here is
analytic: no trained network. This lets us isolate whether the SMC/Girsanov machinery itself
is correct, separate from any neural-network approximation error.

Convention matches the real pipeline exactly: forward noising is x_sigma = x0 + sigma*eps
(sigma(t)=t, VE/EDM-style), and the "denoiser" D(x,sigma) below is the exact Bayes-optimal
E[x0 | x_sigma=x] for this prior -- i.e. the same role D_theta plays in smc/v_tau.py, just
closed-form instead of a network call.

Why a mixture and not a single Gaussian: with a single-Gaussian prior, D(x,sigma) is affine in
x, and the plug-in approximate likelihood ends up proportional to the true intermediate
likelihood (same mode, only differs by an overall scale) -- too easy a test, since the
approximate and exact guided drifts would point the same direction. A mixture prior makes
D(x,sigma) a genuinely nonlinear, responsibility-weighted function of x, so the approximate
and exact intermediate likelihoods can differ in shape, not just scale -- a meaningful test of
whether the V_tau correction actually fixes that gap.
"""

import math

import torch
from torch import Tensor


class GaussianMixture:
    """p(x0) = sum_k w_k * N(x0; mu_k, var_k), with the machinery needed for VE/EDM-style
    forward noising: responsibilities, exact score, exact denoiser, exact posterior, and the
    exact intermediate likelihood p(y | x_sigma) under a linear-Gaussian observation model.
    """

    def __init__(self, w, mu, var):
        self.w = torch.as_tensor(w, dtype=torch.float64)
        self.mu = torch.as_tensor(mu, dtype=torch.float64)
        self.var = torch.as_tensor(var, dtype=torch.float64)

    def sample_x0(self, n, generator=None):
        """Ancestral sampling from the prior -- exact, no SDE needed."""
        k = torch.multinomial(self.w, n, replacement=True, generator=generator)
        eps = torch.randn(n, dtype=torch.float64, generator=generator)
        return self.mu[k] + eps * self.var[k].sqrt()

    def _responsibilities(self, x, sigma):
        """gamma_k(x) for the noised marginal p_sigma(x) = sum_k w_k N(x; mu_k, var_k+sigma^2)."""
        V = self.var + sigma ** 2
        log_a = (torch.log(self.w) - 0.5 * torch.log(2 * math.pi * V)
                 - 0.5 * (x.unsqueeze(-1) - self.mu) ** 2 / V)
        return torch.softmax(log_a, dim=-1)

    def _component_posterior(self, x, sigma):
        """Per-component posterior mean/var of x0 given x_sigma=x, plus responsibilities."""
        V = self.var + sigma ** 2
        m_k = self.mu + (self.var / V) * (x.unsqueeze(-1) - self.mu)
        c_k = self.var * sigma ** 2 / V
        gamma = self._responsibilities(x, sigma)
        return gamma, m_k, c_k

    def denoise(self, x, sigma):
        """Exact Bayes-optimal D(x,sigma) = E[x0 | x_sigma=x] -- plays D_theta's role."""
        gamma, m_k, _ = self._component_posterior(x, sigma)
        return (gamma * m_k).sum(-1)

    def score(self, x, sigma):
        """Exact d/dx log p_sigma(x), for cross-checking s_theta = (x-D(x,sigma))/sigma^2."""
        V = self.var + sigma ** 2
        gamma = self._responsibilities(x, sigma)
        per_k = -(x.unsqueeze(-1) - self.mu) / V
        return (gamma * per_k).sum(-1)

    def posterior(self, y, r):
        """Exact p(x0|y) for y = x0 + N(0,r^2): also a mixture (ground truth)."""
        V = self.var + r ** 2
        log_a = torch.log(self.w) - 0.5 * torch.log(2 * math.pi * V) - 0.5 * (y - self.mu) ** 2 / V
        post_w = torch.softmax(log_a, dim=-1)
        post_mu = self.mu + (self.var / V) * (y - self.mu)
        post_var = self.var * r ** 2 / V
        return GaussianMixture(post_w, post_mu, post_var)

    def log_approx_likelihood(self, x, y, sigma, r):
        """log(tilde h_tau)(x) := log p(y | D(x,sigma), r^2) -- the plug-in approximation
        actually used for guidance (same role as burgers_ell_fn's `ell`)."""
        D = self.denoise(x, sigma)
        return -0.5 * math.log(2 * math.pi * r ** 2) - 0.5 * (y - D) ** 2 / r ** 2

    def log_exact_likelihood(self, x, y, sigma, r):
        """log p(y | x_sigma=x) -- the EXACT intermediate likelihood, normally intractable,
        closed-form here via the mixture-in-y derivation."""
        gamma, m_k, c_k = self._component_posterior(x, sigma)
        var_y = r ** 2 + c_k
        log_terms = torch.log(gamma) - 0.5 * torch.log(2 * math.pi * var_y) - 0.5 * (y - m_k) ** 2 / var_y
        return torch.logsumexp(log_terms, dim=-1)


def sigma_schedule(sigma_min, sigma_max, num_steps, rho):
    """Same rho-schedule as scripts/generate_burgers.py, appended with a final exact-0 target
    (never used as sigma_cur, only as the last step's sigma_next -- matches the real sampler)."""
    i = torch.arange(num_steps, dtype=torch.float64)
    s = (sigma_max ** (1 / rho) + i / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([s, torch.zeros(1, dtype=torch.float64)])


def exact_guidance_grad(mixture, x, sigma, y, r):
    """grad_x log p(y|x_sigma=x), the TRUE guidance gradient (no approximation, no V_tau
    correction needed since this drift is already exact)."""
    x = x.detach().requires_grad_(True)
    log_lik = mixture.log_exact_likelihood(x, y, sigma, r)
    grad_x, = torch.autograd.grad(log_lik.sum(), x)
    return grad_x.detach()


def approx_guidance_grad(mixture, x, sigma, y, r):
    """grad_x log(tilde h_tau)(x), the approximate (plug-in) guidance gradient actually used
    by the approximate-guided sampler."""
    x = x.detach().requires_grad_(True)
    ell = mixture.log_approx_likelihood(x, y, sigma, r)
    grad_x, = torch.autograd.grad(ell.sum(), x)
    return grad_x.detach()


def batched_h_tau(mixture, x, sigma, y, r):
    """H_tau(x) = a_bar * [s_theta.grad_ell + (1/2)*grad_ell^2 + (1/2)*d2(ell)/dx^2], batched
    over independent 1D particles sharing the same sigma. Same formula as smc/v_tau.py's
    H_tau, but the "Laplacian" here is an EXACT second derivative (not Hutchinson-MC): with
    one scalar dimension per particle, a Rademacher probe v in {-1,+1} always has v^2=1, so
    Hutchinson's estimator is exact/zero-variance in this special case anyway -- computing
    d2(ell)/dx^2 directly via double-backward is just the cheaper way to get the same thing.
    """
    x = x.detach().requires_grad_(True)
    ell = mixture.log_approx_likelihood(x, y, sigma, r)
    grad_x, = torch.autograd.grad(ell, x, grad_outputs=torch.ones_like(ell), create_graph=True)
    d2ell_dx2, = torch.autograd.grad(grad_x, x, grad_outputs=torch.ones_like(grad_x))

    s_theta = (x.detach() - mixture.denoise(x.detach(), sigma)) / sigma ** 2
    a_bar = 2 * sigma
    # SIGN FLIP (see conversation notes): b_ap = a_bar*(-s_theta + grad_ell), not
    # a_bar*(+s_theta + grad_ell) -- confirmed via Tweedie's formula, Anderson's reverse-SDE
    # theory, and the real generate_burgers.py sampler, all three independently agreeing that
    # s_theta (as literally defined, (x-D)/sigma^2) is the *negative* of the true score, so the
    # correct denoising drift needs a minus sign here.
    drift_dot_grad = -a_bar * s_theta * grad_x.detach()
    grad_norm_sq_term = 0.5 * a_bar * grad_x.detach() ** 2
    laplacian_term = 0.5 * a_bar * d2ell_dx2.detach()
    return drift_dot_grad + grad_norm_sq_term + laplacian_term
