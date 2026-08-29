"""Closed-form 1D Gaussian-mixture validation of the Girsanov-corrected SMC weight,
organised as a uniform grid of experiments (see `docs/note_1.pdf` Sec. 2-3).

Every ingredient -- prior, noised marginal, score, exact intermediate likelihood, guidance
drift, Hessian, and analytic posterior -- is available in closed form, so any deviation from
the analytic posterior is algorithm error.

Grid axes (each experiment evaluates the full cell set with the same metrics):

  * twist (misspecification of the surrogate twist p~(y | xi_tau)):
      - exact       : gamma~ = gamma (Sec. 3.3; V_tau = 0, corrected increments vanish)
      - surrogate   : gamma~ = kappa*gamma, constant (Sec. 3.4; terminally inconsistent)
      - consistent  : gamma~(sigma) annealed to gamma as sigma -> 0 (Sec. 2.5 remedy;
                      terminally consistent, terminal correction (23) vanishes)
      - plug_in     : p~(y | xi_sigma) = N(y; D(xi,sigma), r^2), the realistic surrogate
                      used for guidance in practice, with D the exact Bayes-optimal denoiser
                      (note_2 eq. 2.6).  Terminally consistent (D(xi,0)=xi) but overconfident
                      at intermediate sigma: the naive width r^2 = gamma^2 ignores the
                      residual uncertainty c_bar(sigma) = E_x[sigma^2 gradD], making the
                      guidance up to ~4.5x too strong at sigma_max (gamma^2=1).
      - plug_in_corrected : same plug-in with the residual-corrected width
                      r^2(sigma) = gamma^2 + c_bar(sigma) (c_bar a one-time quadrature of
                      the exact marginal; same oracle denoiser, seed-independent).  This is
                      the deployable variant: terminally consistent and no intermediate-sigma
                      overconfidence (see `plan_plug_in.md`).
    with kappa = gamma~_max / gamma the observation-std ratio at sigma_max.

  * proposal: EM (Euler-Maruyama, App. B; C_hat_k exact kernel ratio) or stochastic Heun
    (2-stage RK, same z; kernel non-Gaussian so corrections are consistent-asymptotic).

  * weighting (each used as the whole per-step increment G_k of eq. 19):
      - pbs      : G_k = Delta f_k                       (eq. 30; drops path-measure correction)
      - girs     : G_k = Delta f_k + C_hat_k             (eq. 31; C_hat_k left-endpoint, eq. 18)
      - pot      : G_k = V_hat_k, frozen-state + left-endpoint rectangle quadrature (eq. 17)
      - pot_trap : G_k = V_hat_k, frozen-state + trapezoidal quadrature of the H integrand
                   (second-order in the sigma step; makes the quadrature error directly
                   readable as the Pot/Pot-tr column gap)

Tables T1-T4 (all one-perturbation from the canonical setting N=2048, K=500, gamma2=1,
kappa=0.5; every table is the mean over N_SEEDS seeds):

  * T1 Validity     : twist x weighting (EM)  -- machine check (eq. 33/34) + corrected-vs-PBS
                      claim + terminal consistency, over all five twists.
  * T2 Base grid    : proposal x weighting at the canonical plug-in corr twist
                      (residual-corrected width, r^2(sigma) = gamma^2 + c_bar(sigma)).
  * T3 Convergence  : K-sweep and N-sweep over the base grid (discretisation and MC error).
  * T4 Regime       : gamma2-sweep over the base grid (observational regime where SMC works
                      vs fails) with the plug-in corr twist: as gamma^2 shrinks the posterior
                      sharpens and the corrected weightings degenerate at the
                      observation-informativeness limit (no surrogate-induced overconfidence).

Metrics (Sec. 3.5): Wasserstein-1 (eq. 35), weighted mean/std error, ESS, resampling count.
For each cell we also record per-step increments, per-step ESS, the Girs-vs-Pot bridge gap
(rectangle and trapezoidal quadrature error), and the terminal correction (23).  Tables
report mean +/- std over the N_SEEDS seeds; figures show mean curves (no error bars).

Working convention (reverse-time, Sec. 2.1/3.1): sigma decreases sigma_max -> sigma_min as
tau goes 0 -> T, accumulated diffusion Sigma_k = s^2_{tau_{k-1}} - s^2_{tau_k} (eq. 27).
The chain rule d/dtau = -d/dsigma fixes the sign of the potential term.

Outputs: figures in `figures/` and LaTeX tables in `tables/`, named `toy_t1*`, `toy_t2*`,
`toy_t3*`, `toy_t4*` (figures and tables share base names).

Run: .venv/bin/python smc/scripts_1/toy_smc.py
"""

import os
import resource
import time
from functools import lru_cache

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, logsumexp
from scipy.stats import gaussian_kde, norm

resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024))

# ----------------------------------------------------------------- model (Sec. 3.1, eq. 25-26)
C = np.array([0.6, 0.4])
MU = np.array([-1.5, 2.0])
VAR = np.array([0.7, 1.3])
OBS_Y = 0.0
KAPPA = 0.5                       # gamma~_max / gamma (obs-std ratio of the twist at sigma_max)
N_SEEDS = 4                       # every table is the mean over seeds 0..N_SEEDS-1
SEEDS = tuple(range(N_SEEDS))
K_SWEEP = [25, 50, 100, 200, 500]  # T3: step count (coarse regime)
N_SWEEP = [32, 128, 512, 2048]     # T3: particle count (powers of 4)
GAMMA2_SWEEP = [4.0, 1.0, 0.25, 0.0625]   # T4: exact obs variance (std {2,1,0.5,0.25})


class ExactFormTwist:
    """Twist whose likelihood is the exact-form intermediate likelihood of (28) with
    observation variance `obs_var` (scalar or callable of sigma).  Backs the exact,
    surrogate, and consistent families."""
    def __init__(self, obs_var):
        self.obs_var = obs_var

    def obs(self, sigma):
        return _twist_obs(self.obs_var, sigma)

    def loglik(self, x, sigma):
        return loglik(x, sigma, self.obs(sigma))

    def stats(self, x, sigma):
        return lik_stats(x, sigma, self.obs(sigma))


class PlugInTwist:
    """Realistic plug-in surrogate: p~(y | x_sigma) = N(y; D(x,sigma), r^2), with D the exact
    Bayes-optimal denoiser E[x0 | x_sigma] (note_2 eq. 2.3/2.6).  This is the surrogate
    structure actually used for guidance in practice (the PDE pipeline's plug-in likelihood).
    Terminally consistent when r^2(sigma) -> gamma^2 as sigma -> 0: D(x,0) = x, so
    f(x,0) = log p(y|x).  With the naive width r^2 = gamma^2 it ignores the residual
    uncertainty c_k(sigma) at intermediate sigma and is overconfident by a factor
    ~ 1 + sigma^2/r^2 that grows with sigma; the corrected variant r^2(sigma) = gamma^2 +
    cbar(sigma) adds the expected residual.  `r2` is a scalar or a callable of sigma.
    b, H are computed analytically from the mixture responsibilities."""
    def __init__(self, r2):
        self.r2 = r2

    def obs(self, sigma):
        return _twist_obs(self.r2, sigma)

    def loglik(self, x, sigma):
        return self.stats(x, sigma)['log_lik']

    def stats(self, x, sigma):
        return plug_in_stats(x, sigma, self.obs(sigma))


def exact_twist(gamma2=1.0):
    """Twist = exact intermediate likelihood: p~(y|xi_tau) = p(y|xi_tau) (Sec. 3.3)."""
    return ExactFormTwist(gamma2)


def surrogate_twist(gamma2=1.0, kappa=KAPPA):
    """Constant misspecified observation scale gamma~ = kappa*gamma (Sec. 3.4, terminally
    inconsistent surrogate)."""
    v = (kappa * np.sqrt(gamma2)) ** 2
    return ExactFormTwist(v)


def consistent_twist(gamma2=1.0, kappa=KAPPA):
    """Terminally-consistent surrogate (Sec. 2.5 remedy): gamma~(sigma) = gamma*
    (1 + (kappa-1)*sigma/sigma_max), so gamma~ -> gamma as sigma -> 0 and the terminal
    correction (23) vanishes."""
    g = np.sqrt(gamma2)
    return ExactFormTwist(lambda sigma: (g * (1.0 + (kappa - 1.0) * sigma / SIGMA_MAX)) ** 2)


def plug_in_twist(gamma2=1.0):
    """Realistic plug-in surrogate with the true observation width r^2 = gamma^2."""
    return PlugInTwist(r2=gamma2)


def plug_in_corrected_twist(gamma2=1.0):
    """Plug-in surrogate with the residual-corrected width r^2(sigma) = gamma^2 + cbar(sigma),
    cbar(sigma) = E_x[sigma^2 * gradD(x, sigma)] the expected residual posterior variance
    (derived from the same oracle denoiser D the naive plug-in uses).  Terminally consistent
    (cbar -> 0 as sigma -> 0, so eq. 23 vanishes) and removes the plug-in's intermediate-sigma
    overconfidence that the naive width gamma^2 leaves (up to ~4.5x too narrow at sigma_max)."""
    cbar = _cbar_schedule()
    return PlugInTwist(r2=lambda sigma: gamma2 + cbar(sigma))


# ----------------------------------------------------------------- VE schedule (Sec. 2.4 / eq. 27)
SIGMA_MAX, SIGMA_MIN, RHO_SCHED = 6.0, 1e-3, 7.0


def build_sigma(K):
    """Decreasing EDM power-law over [sigma_max, sigma_min], K intervals, terminal at sigma_min.

    No hard-appended 0.0: an appended terminal point made the final step's accumulated
    variance Sigma_K = sigma_min^2 ~57x larger than its neighbours, producing a terminal spike
    in the per-step increments. Ending the power-law smoothly at the moderate sigma_min keeps
    the step sizes smooth (Sigma ratio ~1) and wastes no steps near 0.
    """
    idx = np.arange(K + 1)
    s = (SIGMA_MAX ** (1 / RHO_SCHED) + idx / K *
         (SIGMA_MIN ** (1 / RHO_SCHED) - SIGMA_MAX ** (1 / RHO_SCHED))) ** RHO_SCHED
    return s


# ----------------------------------------------------------------- analytic posterior (eq. 29)
def posterior_params(gamma2=1.0):
    S = VAR + gamma2
    q = C * np.exp(-0.5 * (OBS_Y - MU) ** 2 / S) / np.sqrt(2 * np.pi * S)
    abar = q / q.sum()
    var_p = 1 / (1 / VAR + 1 / gamma2)
    mu_p = var_p * (MU / VAR + OBS_Y / gamma2)
    pm = (abar * mu_p).sum()
    pv = (abar * (var_p + mu_p ** 2)).sum() - pm ** 2
    return abar, mu_p, var_p, float(pm), float(np.sqrt(max(pv, 0)))


def posterior_cdf(xx, gamma2=1.0):
    abar, mu_p, var_p, _, _ = posterior_params(gamma2)
    return (abar[:, None] *
            0.5 * (1 + erf((xx[None, :] - mu_p[:, None]) /
                           np.sqrt(2 * var_p[:, None])))).sum(axis=0)


def posterior_pdf(xx, gamma2=1.0):
    abar, mu_p, var_p, _, _ = posterior_params(gamma2)
    return (abar[:, None] *
            norm.pdf(xx[None, :], mu_p[:, None], np.sqrt(var_p[:, None]))).sum(axis=0)


def wasserstein1(samples, log_w=None, gamma2=1.0, lo=-6, hi=6, grid=8000):
    g = np.linspace(lo, hi, grid)
    Fp = posterior_cdf(g, gamma2)
    order = np.argsort(samples)
    if log_w is not None:
        w = np.exp(log_w - np.max(log_w))
        w = w / w.sum()
        Fn = np.interp(g, samples[order], np.cumsum(w[order]), left=0.0, right=1.0)
    else:
        Fn = np.searchsorted(samples[order], g) / len(samples)
    return float(np.trapezoid(np.abs(Fn - Fp), g))


# ----------------------------------------------------------------- closed forms (Sec. 3.1)
def _resp(x, sigma):
    """Mixture responsibilities gamma_m and component variances V_m = sigma_m^2 + sigma^2."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    V = VAR + sigma * sigma                     # (M,)
    logpi = (np.log(C) - 0.5 * np.log(2 * np.pi * V) -
             0.5 * (x[:, None] - MU[None, :]) ** 2 / V[None, :])   # (N, M)
    lse = logsumexp(logpi, axis=-1)             # (N,)
    gamma = np.exp(logpi - lse[:, None])
    return gamma, V, lse


def noised_logpdf(x, sigma):
    _, _, lse = _resp(x, sigma)
    return lse


def score(x, sigma):
    """Exact marginal score s = grad_xi log p(xi; sigma)."""
    gamma, V, _ = _resp(x, sigma)
    return (gamma * (MU[None, :] - x[:, None]) / V[None, :]).sum(-1)


def lik_stats(x, sigma, obs_var):
    """Exact intermediate likelihood and its derivatives (eq. 28, 8).

    f(xi; sigma) = log p(y | xi_sigma) with observation variance `obs_var`; b = grad f;
    H = grad^2 f; score is the prior marginal score (obs-independent).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    gamma, V, _ = _resp(x, sigma)
    r = VAR / V                                  # (M,)
    c = VAR * sigma * sigma / V                  # (M,)
    s2 = obs_var + c                             # (M,)
    mm = MU[None, :] + r[None, :] * (x[:, None] - MU[None, :])      # (N, M)
    logphi = -0.5 * np.log(2 * np.pi * s2) - 0.5 * (OBS_Y - mm) ** 2 / s2[None, :]
    log_omega = np.log(gamma) + logphi
    log_S = logsumexp(log_omega, axis=-1)        # f = log p(y | xi, sigma)  (N,)
    rho = np.exp(log_omega - log_S[:, None])     # (N, M)

    u = (MU[None, :] - x[:, None]) / V[None, :]                       # grad log N(xi;mu_m,V_m)
    v = (OBS_Y - mm) * r[None, :] / s2[None, :]                       # grad log phi_m
    s = (gamma * u).sum(-1)                                          # marginal score
    grad_s = (gamma * (u * (u - s[:, None]) - 1 / V[None, :])).sum(-1)  # grad s
    d = u + v - s[:, None]
    b = (rho * d).sum(-1)                                            # eq. 28 gradient
    H = ((rho * ((d - b[:, None]) * d - 1 / V[None, :] -
                 r[None, :] ** 2 / s2[None, :])).sum(-1) - grad_s)  # grad^2 f
    return dict(log_lik=log_S, grad=b, hess=H, score=s)


def loglik(x, sigma, obs_var):
    return lik_stats(x, sigma, obs_var)['log_lik']


def plug_in_stats(x, sigma, r2):
    """Exact plug-in surrogate and its derivatives.

    f(xi; sigma) = log N(y; D(xi,sigma), r^2) with D(xi,sigma) = E[x0 | xi_sigma = xi] the
    Bayes-optimal denoiser (note_2 eq. 2.3/2.6).  b = grad f; H = grad^2 f; score is the prior
    marginal score (obs-independent).  Derivatives are closed form via the mixture
    responsibilities:
      dD/dx   = sum_m gamma_m[(u_m - s) m_m + r_m]
      d2D/dx2 = sum_m gamma_m{ m_m[(u_m-s)^2 - 1/V_m - grad_s] + 2(u_m-s) r_m }
    with m_m = mu_m + r_m (x - mu_m), r_m = var_m/V_m, u_m = (mu_m - x)/V_m, s = sum gamma u.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    gamma, V, _ = _resp(x, sigma)
    r = VAR / V                                  # (M,)
    mm = MU[None, :] + r[None, :] * (x[:, None] - MU[None, :])      # (N, M) posterior comp means
    D = (gamma * mm).sum(-1)                     # denoiser (N,)
    u = (MU[None, :] - x[:, None]) / V[None, :]                     # (N, M)
    s = (gamma * u).sum(-1)                                          # marginal score (N,)
    grad_s = (gamma * (u * (u - s[:, None]) - 1 / V[None, :])).sum(-1)

    gradD = (gamma * ((u - s[:, None]) * mm + r[None, :])).sum(-1)
    grad2D = (gamma * (mm * ((u - s[:, None]) ** 2 - 1 / V[None, :] - grad_s[:, None])
                       + 2.0 * (u - s[:, None]) * r[None, :])).sum(-1)

    g = (OBS_Y - D) / r2
    f = -0.5 * np.log(2 * np.pi * r2) - 0.5 * (OBS_Y - D) ** 2 / r2
    b = g * gradD
    H = (-gradD ** 2 + (OBS_Y - D) * grad2D) / r2
    return dict(log_lik=f, grad=b, hess=H, score=s, D=D, gradD=gradD)


def _twist_obs(obs_var, sigma):
    """Observation variance of the twist at noise level sigma (scalar or callable)."""
    return obs_var(sigma) if callable(obs_var) else obs_var


@lru_cache(maxsize=None)
def _cbar_schedule(n_grid=96, n_x=40000):
    """Tabulate cbar(sigma) = E_x[sigma^2 * gradD(x, sigma)] on a log-sigma grid over
    [SIGMA_MIN, SIGMA_MAX], by quadrature of sigma^2 * gradD(x, sigma) against the exact
    mixture marginal p(x; sigma).  cbar is a prior expectation (seed-independent), so it is
    computed once and interpolated in log-sigma; the returned object is a callable r2(sigma).
    Independent of gamma2.  At sigma_max and gamma2=1, cbar ~ 3.5, i.e. the naive plug-in
    width gamma^2 is ~4.5x too narrow there."""
    sigs = np.geomspace(SIGMA_MIN, SIGMA_MAX, n_grid)
    x = np.linspace(-40.0, 40.0, n_x)
    cbars = np.empty_like(sigs)
    for i, s in enumerate(sigs):
        st = plug_in_stats(x, s, 1.0)          # r2 irrelevant for D / gradD
        p = np.exp(noised_logpdf(x, s))
        cbars[i] = s * s * np.trapezoid(st['gradD'] * p, x) / np.trapezoid(p, x)
    return lambda sigma: np.interp(np.log(max(sigma, SIGMA_MIN)),
                                   np.log(sigs), cbars)


# ----------------------------------------------------------------- proposal (Appendix B)
def init_particles(N, rng):
    m = rng.choice(len(C), size=N, p=C)
    return MU[m] + np.sqrt(VAR[m] + SIGMA_MAX ** 2) * rng.standard_normal(N)


def systematic_resample(log_w, rng):
    w = np.exp(log_w - np.max(log_w))
    w = w / w.sum()
    N = w.size
    u = (rng.random() + np.arange(N)) / N
    return np.searchsorted(np.cumsum(w), u)


# ----------------------------------------------------------------- core sampler (Sec. 2.4-2.5)
def run_smc(N, K, seed, twist, weighting, proposal='em', gamma2=1.0, resample_threshold=0.5):
    """Guided proposal + one of the four weightings.  Returns result dict.

    `proposal` is 'em' (Euler-Maruyama, App. B; C_hat_k exact kernel ratio) or 'heun'
    (stochastic Heun / 2-stage RK with the same z in both stages; the kernel is non-Gaussian,
    so C_hat_k is consistent-asymptotic rather than finite-step exact).

    `twist` is a twist object with `.loglik(x, sigma)` and `.stats(x, sigma)` (the exact /
    surrogate / consistent / plug-in families above).  `gamma2` is the exact observation
    variance, used only for the terminal correction (23).

    Pipeline (eq. 19-23): initial f0(xi_0); per-step G_k; terminal correction
    log p(y|xi_T) - fT(xi_T) (exact gamma always).  Systematic resampling at ESS < N/2.
    """
    sig = build_sigma(K)
    rng = np.random.default_rng(seed)
    x = init_particles(N, rng)

    log_w = twist.loglik(x, SIGMA_MAX)                  # initial weight f0(xi_0) (eq. 21)
    inc_hist = []
    bridge_rect = []
    bridge_trap = []
    n_resample = 0
    ess_history = []

    for step in range(K):
        sk, skm1 = sig[step], sig[step + 1]
        delta = sk ** 2 - skm1 ** 2                # Sigma_k = s^2_{tau_{k-1}} - s^2_{tau_k} (eq. 27)
        st = twist.stats(x, sk)                     # f, b, H, score at (xi_{k-1}, sigma_{k-1})

        eps = rng.standard_normal(N)
        z = np.sqrt(delta) * eps                    # z_k ~ N(0, Sigma_k) (eq. 12)
        if proposal == 'em':
            x_next = x + delta * (st['score'] + st['grad']) + z    # EM proposal (App. B)
        elif proposal == 'heun':
            # stochastic Heun: predictor then corrector, same z in both stages
            x_pred = x + delta * (st['score'] + st['grad']) + z
            stp = twist.stats(x_pred, skm1)
            x_next = x + 0.5 * delta * ((st['score'] + st['grad']) +
                                        (stp['score'] + stp['grad'])) + z
        else:
            raise ValueError(f'unknown proposal: {proposal}')

        f_km1 = twist.loglik(x_next, skm1)          # f(xi_k; sigma_k)
        f_k = st['log_lik']                         # f(xi_{k-1}; sigma_{k-1})
        dll = f_km1 - f_k                           # Delta f_k (eq. 13)

        # Girsanov left-endpoint correction (eq. 18)
        C = -st['grad'] * z - 0.5 * st['grad'] ** 2 * delta
        # Potential increments (eq. 17, 11): frozen-state telescoping of d_tau f + quadrature
        # of the remaining integrand 2*sigma*[score*b + 1/2 b^2 + 1/2 H].  st_next is the
        # frozen left state at the step's lower noise (shared by pot and pot_trap).
        st_next = twist.stats(x, skm1)
        Hk = st['score'] * st['grad'] + 0.5 * st['grad'] ** 2 + 0.5 * st['hess']
        Hkm1 = st_next['score'] * st_next['grad'] + 0.5 * st_next['grad'] ** 2 + 0.5 * st_next['hess']
        Vpot = (st_next['log_lik'] - f_k) + (sk - skm1) * 2.0 * sk * Hk
        Vpot_trap = (st_next['log_lik'] - f_k) + 0.5 * (sk - skm1) * (2.0 * sk * Hk + 2.0 * skm1 * Hkm1)

        G = {'pbs': dll, 'girs': dll + C, 'pot': Vpot, 'pot_trap': Vpot_trap}[weighting]
        log_w += G
        log_w -= logsumexp(log_w)
        ess_cur = float(1.0 / np.sum(np.exp(2 * log_w)))
        ess_history.append(ess_cur)

        inc_hist.append(np.mean(np.abs(G)))
        bridge_rect.append(np.mean(np.abs((dll + C) - Vpot)))
        bridge_trap.append(np.mean(np.abs((dll + C) - Vpot_trap)))

        if ess_cur < resample_threshold * N:
            idx = systematic_resample(log_w, rng)
            x_next = x_next[idx]
            log_w = np.zeros(N)
            n_resample += 1

        x = x_next

    pre_terminal = log_w.copy()
    # terminal correction (eq. 23), always with the exact likelihood
    terminal_corr = loglik(x, 0.0, gamma2) - twist.loglik(x, 0.0)
    log_w += terminal_corr
    log_w -= logsumexp(log_w)
    ess_final = float(1.0 / np.sum(np.exp(2 * log_w)))
    ess_history.append(ess_final)   # includes the post-terminal-correction collapse (eq. 23)

    return {'particles': x, 'weights': log_w, 'pre_terminal': pre_terminal,
            'ess': ess_final, 'terminal_corr': terminal_corr,
            'ess_history': np.array(ess_history), 'resamples': n_resample,
            'inc_hist': np.array(inc_hist),
            'bridge_rect': np.array(bridge_rect), 'bridge_trap': np.array(bridge_trap)}


def weighted_stats(x, log_w):
    lw = log_w - np.max(log_w)
    w = np.exp(lw)
    w = w / w.sum()
    m = float((w * x).sum())
    v = float((w * x ** 2).sum()) - m ** 2
    return m, float(np.sqrt(max(v, 0)))


# ----------------------------------------------------------------- closed-form self-checks
def selfcheck():
    h = 1e-6
    xs = np.array([-2.0, 0.3, 1.7])
    sigs = np.array([0.05, 0.5, 2.0, 5.0])
    err_b = err_H = err_score = 0.0
    for gamma2 in (1.0, 0.0625):
        for tw in (exact_twist(gamma2), surrogate_twist(gamma2, KAPPA),
                   plug_in_twist(gamma2), plug_in_corrected_twist(gamma2)):
            for x in xs:
                for sig in sigs:
                    st = tw.stats(x, sig)
                    err_b = max(err_b, float(np.abs(st['grad'] -
                                           (tw.loglik(x + h, sig) - tw.loglik(x - h, sig)) / (2 * h))[0]))
                    err_H = max(err_H, float(np.abs(st['hess'] -
                                           (tw.stats(x + h, sig)['grad'] -
                                            tw.stats(x - h, sig)['grad']) / (2 * h))[0]))
                    err_score = max(err_score, float(np.abs(st['score'] -
                                                   (noised_logpdf(x + h, sig) - noised_logpdf(x - h, sig)) / (2 * h))[0]))
    return float(err_b), float(err_H), float(err_score)


# ----------------------------------------------------------------- grid infrastructure
WEIGHTS = ('pbs', 'girs', 'pot', 'pot_trap')
WEIGHT_LABEL = {'pbs': 'PBS', 'girs': 'Girs', 'pot': 'Pot', 'pot_trap': 'Pot-tr'}
PROPOSALS = ('em', 'heun')
PROPOSAL_LABEL = {'em': 'EM', 'heun': 'Heun'}
COLOR = {'pbs': '#d62728', 'girs': '#1f77b4', 'pot': '#2ca02c', 'pot_trap': '#9467bd'}
MARKER = {'pbs': 'o', 'girs': 's', 'pot': '^', 'pot_trap': 'D'}
TWIST_LABEL = {'exact': 'Exact', 'surrogate': 'Surrogate', 'consistent': 'Consistent',
               'plug_in': 'Plug-in', 'plug_in_corrected': 'Plug-in corr'}


def run_cell(N, K, seeds, twist, proposal='em', gamma2=1.0):
    """Run every weighting at one (twist, proposal, N, K, gamma2) cell, averaged over the
    given seeds.  Scalar metrics are mean +/- std over seeds; per-step arrays are the mean;
    particles/weights/pre_terminal are from the first seed (representative, for plotting and
    diagnostics)."""
    pm, ps = posterior_params(gamma2)[-2:]
    results = {}
    for w in WEIGHTS:
        runs = [run_smc(N, K, s, twist, w, proposal=proposal, gamma2=gamma2) for s in seeds]
        r0 = runs[0]
        w1s = [wasserstein1(r['particles'], r['weights'], gamma2) for r in runs]
        pairs = [weighted_stats(r['particles'], r['weights']) for r in runs]
        dmeans = [abs(m - pm) for m, _ in pairs]
        dstds = [abs(v - ps) for _, v in pairs]
        esss = [r['ess'] for r in runs]
        resamps = [r['resamples'] for r in runs]
        results[w] = dict(
            w1=float(np.mean(w1s)), w1_std=float(np.std(w1s)),
            dmean=float(np.mean(dmeans)), dmean_std=float(np.std(dmeans)),
            dstd=float(np.mean(dstds)), dstd_std=float(np.std(dstds)),
            ess=float(np.mean(esss)), ess_std=float(np.std(esss)),
            resamples=float(np.mean(resamps)), resamples_std=float(np.std(resamps)),
            terminal_corr=float(np.mean([np.abs(r['terminal_corr']).mean() for r in runs])),
            particles=r0['particles'], weights=r0['weights'], pre_terminal=r0['pre_terminal'],
            inc_hist=np.mean([r['inc_hist'] for r in runs], axis=0),
            bridge_rect=np.mean([r['bridge_rect'] for r in runs], axis=0),
            bridge_trap=np.mean([r['bridge_trap'] for r in runs], axis=0),
            ess_history=np.mean([r['ess_history'] for r in runs], axis=0))
    return results


def _print_metrics(title, res, pm, ps, gamma2):
    print(f'{title}  (N={res["N"]}, K={res["K"]}, gamma2={gamma2}, '
          f'proposal={PROPOSAL_LABEL[res["proposal"]]}, seeds={N_SEEDS})')
    print(f'  analytic posterior: mean={pm:.4f}  std={ps:.4f}')
    print(f'  {"weight":>7}  {"W1":>14}  {"|dmean|":>13}  {"|dstd|":>14}  {"ESS":>10}  {"resamp":>10}')
    print('  ' + '-' * 70)
    for w in WEIGHTS:
        r = res['results'][w]
        print(f'  {WEIGHT_LABEL[w]:>7}  {_f4pm(r["w1"], r["w1_std"]):>14}  '
              f'{_f4pm(r["dmean"], r["dmean_std"]):>13}  {_f4pm(r["dstd"], r["dstd_std"]):>14}  '
              f'{_f0pm(r["ess"], r["ess_std"]):>10}  {_f0pm(r["resamples"], r["resamples_std"]):>10}')


# ----------------------------------------------------------------- outputs: figures + tables
FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables')
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def _setup_style():
    """Consistent matplotlib style shared by every figure."""
    plt.rcParams.update({
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'legend.fontsize': 8, 'axes.grid': True, 'grid.alpha': 0.3,
        'grid.linewidth': 0.5, 'figure.dpi': 100, 'savefig.dpi': 150,
        'axes.spines.top': False, 'axes.spines.right': False,
    })


def _f4(x):
    return f'{x:.4f}'


def _f0(x):
    return f'{x:.0f}'


def _fe(x):
    return f'{x:.3e}'


def _f4pm(x, s):
    return f'{x:.4f}$\\pm${s:.4f}'


def _f0pm(x, s):
    return f'{x:.0f}$\\pm${s:.0f}'


def write_latex_table(path, caption, label, headers, rows):
    """Single shared booktabs-style LaTeX table writer (consistent across all experiments)."""
    with open(path, 'w') as f:
        f.write('\\begin{table}[ht]\n\\centering\n')
        f.write(f'\\caption{{{caption}}}\n')
        f.write(f'\\label{{{label}}}\n')
        f.write('\\begin{tabular}{' + 'l' + 'c' * (len(headers) - 1) + '}\n')
        f.write('\\toprule\n')
        f.write(' & '.join(headers) + ' \\\\\n')
        f.write('\\midrule\n')
        for row in rows:
            f.write(' & '.join(row) + ' \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n\\end{table}\n')
    print(f'Saved: {path}')


def write_metrics_table(res, path, caption, label):
    """Standard 4-row metrics table (PBS/Girs/Pot/Pot-tr x W1, |dmean|, |dstd|, ESS, resamp),
    mean +/- std over N_SEEDS seeds."""
    pm, ps = posterior_params(res['gamma2'])[-2:]
    headers = ['weight', 'W1', '|dmean|', '|dstd|', 'ESS', 'resamp']
    rows = [[WEIGHT_LABEL[w], _f4pm(res['results'][w]['w1'], res['results'][w]['w1_std']),
             _f4pm(res['results'][w]['dmean'], res['results'][w]['dmean_std']),
             _f4pm(res['results'][w]['dstd'], res['results'][w]['dstd_std']),
             _f0pm(res['results'][w]['ess'], res['results'][w]['ess_std']),
             _f0pm(res['results'][w]['resamples'], res['results'][w]['resamples_std'])]
            for w in WEIGHTS]
    write_latex_table(path, caption + f' (analytic posterior: $\\mu={pm:.3f}$, '
                      f'$\\sigma={ps:.3f}$; {N_SEEDS} seeds, mean$\\pm$std).',
                      label, headers, rows)


def _density_panel(ax, results, xgrid, pdf_vals):
    ax.plot(xgrid, pdf_vals, 'k--', linewidth=2.0, label='Analytic posterior', zorder=10)
    for w in WEIGHTS:
        xf = results[w]['particles']
        lw = results[w]['weights']
        wgt = np.exp(lw - np.max(lw)); wgt = wgt / wgt.sum()
        kde = gaussian_kde(xf, weights=wgt, bw_method=0.3)
        ax.plot(xgrid, kde(xgrid), color=COLOR[w], alpha=0.8, linewidth=1.2,
                label=WEIGHT_LABEL[w])
    ax.set_xlabel(r'$\xi_T$'); ax.set_ylabel('Density')
    ax.legend(fontsize=7)


def plot_cell(res, title, filename):
    """Shared 1x3 layout used identically by every single-setting experiment so they are
    directly comparable.  Panels: weighted density vs analytic posterior (shape; representative
    seed), per-step increment magnitude (mechanism, eq. 33/34; mean over seeds), ESS over steps
    (terminal-correction collapse, eq. 23; mean over seeds)."""
    results = res['results']
    gamma2 = res['gamma2']
    N, K = res['N'], res['K']
    x = results['girs']['particles']
    lo, hi = float(x.min()) - 0.3, float(x.max()) + 0.3
    xgrid = np.linspace(lo, hi, 800)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _density_panel(axes[0], results, xgrid, posterior_pdf(xgrid, gamma2))
    axes[0].set_title('Density (weighted KDE)')

    steps = np.arange(1, K + 1)
    ax = axes[1]
    for w in WEIGHTS:
        ax.plot(steps, results[w]['inc_hist'], color=COLOR[w], linewidth=1.0,
                label=WEIGHT_LABEL[w])
    ax.set_xlabel('step k'); ax.set_ylabel(r'mean $|G_k|$')
    ax.set_title('Per-step increment magnitude (eq. 33/34)')
    ax.set_yscale('log'); ax.legend(fontsize=7)

    ax = axes[2]
    t = np.arange(K + 1)                      # K per-step ESS + final (post-terminal-correction)
    for w in WEIGHTS:
        ax.plot(t, results[w]['ess_history'], color=COLOR[w], linewidth=1.0,
                label=WEIGHT_LABEL[w])
    ax.set_xlabel('step k'); ax.set_ylabel('ESS')
    ax.set_title('ESS over steps (final = after eq. 23)')
    ax.legend(fontsize=7)

    fig.suptitle(f'{title}  (N={N}, K={K})')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(FIGS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_sweep(xvals, series, path, title, xlabel, xscale='log', yscale='log', ylabel='W1'):
    """One panel per weighting; solid = EM, dashed = Heun.  Mean over seeds, no error bars.
    Used by T3 (K, N) and T4 (gamma2)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w in WEIGHTS:
        for prop in PROPOSALS:
            ls = '-' if prop == 'em' else '--'
            ax.plot(xvals, series[prop][w], color=COLOR[w], marker=MARKER[w], markersize=4,
                    linestyle=ls, linewidth=1.2,
                    label=f'{WEIGHT_LABEL[w]} ({PROPOSAL_LABEL[prop]})')
    ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


# ----------------------------------------------------------------- T1: validity (twist x weighting)
def run_t1(N, K, gamma2=1.0, kappa=KAPPA):
    pm, ps = posterior_params(gamma2)[-2:]
    print(f'\n=== T1: Validity  (twist x weighting, EM, N={N}, K={K}, '
          f'gamma2={gamma2}, kappa={kappa}, seeds={N_SEEDS}) ===')

    twists = {'exact': exact_twist(gamma2),
              'surrogate': surrogate_twist(gamma2, kappa),
              'consistent': consistent_twist(gamma2, kappa),
              'plug_in': plug_in_twist(gamma2),
              'plug_in_corrected': plug_in_corrected_twist(gamma2)}
    allres = {}
    for tname, tw in twists.items():
        res = run_cell(N, K, SEEDS, tw, 'em', gamma2)
        allres[tname] = res
        _print_metrics(f'T1 twist={TWIST_LABEL[tname]}',
                       dict(N=N, K=K, proposal='em', results=res), pm, ps, gamma2)
        write_metrics_table(dict(N=N, K=K, proposal='em', gamma2=gamma2, results=res),
                            os.path.join(TABLES_DIR, f'toy_t1_{tname}.tex'),
                            f'T1 ({TWIST_LABEL[tname]} twist, $\\kappa={kappa}$, $N={N}$, $K={K}$); '
                            'W1 (eq.~35) and weighted posterior mean/std error vs the analytic '
                            'posterior.',
                            f'tab:toy_t1_{tname}')
        plot_cell(dict(N=N, K=K, proposal='em', gamma2=gamma2, results=res),
                  f'T1: {TWIST_LABEL[tname]} twist', f'toy_t1_{tname}.pdf')

    # compact W1 matrix: rows = twist, cols = weighting
    rows = [[TWIST_LABEL[t]] + [_f4pm(allres[t][w]['w1'], allres[t][w]['w1_std'])
                                for w in WEIGHTS] for t in twists]
    write_latex_table(os.path.join(TABLES_DIR, 'toy_t1.tex'),
                      'T1 validity: Wasserstein-1 (eq.~35) by twist and weighting '
                      f'($N={N}$, $K={K}$, $\\kappa={kappa}$, {N_SEEDS} seeds, mean$\\pm$std).  '
                      'The corrected weightings (Girs, Pot, Pot-tr) sit at the posterior for '
                      'every twist; PBS is biased away from the exact twist and increasingly so '
                      'as the surrogate is misspecified.  The plug-in twists are the realistic '
                      'surrogates ($N(y; D(x,\\sigma), r^2)$, terminally consistent): the naive '
                      'width $r^2=\\gamma^2$ ignores the residual uncertainty and is overconfident, '
                      'while plug-in corr uses the residual-corrected width '
                      '$r^2(\\sigma)=\\gamma^2+\\bar c(\\sigma)$.',
                      'tab:toy_t1', ['twist'] + [WEIGHT_LABEL[w] for w in WEIGHTS], rows)

    _t1_diagnostics(allres, N, K, gamma2, kappa)
    return allres


def _t1_diagnostics(allres, N, K, gamma2, kappa):
    res = allres['exact']
    fp = res['pbs']['particles']
    fT = loglik(fp, 0.0, gamma2)
    diff = res['pbs']['pre_terminal'] - fT
    eq34 = float(np.abs(diff - diff.mean()).mean())          # eq. 34 telescoping spread
    bridge_rect = float(res['girs']['bridge_rect'].mean())    # Girs vs Pot (rect quadrature err)
    bridge_trap = float(res['girs']['bridge_trap'].mean())    # Girs vs Pot-tr (trap quadrature err)
    meanG = {w: float(res[w]['inc_hist'].mean()) for w in WEIGHTS}

    rb = allres['surrogate']['girs']                          # terminally inconsistent
    rc = allres['consistent']['girs']                         # terminally consistent (idealised)
    rp = allres['plug_in']['girs']                            # realistic plug-in (naive width)
    rk = allres['plug_in_corrected']['girs']                  # plug-in, residual-corrected width
    tb = rb['terminal_corr']
    tc = rc['terminal_corr']
    tp = rp['terminal_corr']
    tk = rk['terminal_corr']

    print('\nT1 diagnostics (mean over seeds):')
    print(f'  PBS telescoping spread (eq. 34)       = {eq34:.3e}   (expected ~0)')
    print(f'  mean|G_girs - G_pot_rect| (quadrature) = {bridge_rect:.3e}')
    print(f'  mean|G_girs - G_pot_trap| (quadrature) = {bridge_trap:.3e}')
    print(f'  mean|G| over steps:  PBS={meanG["pbs"]:.3e}  Girs={meanG["girs"]:.3e}  '
          f'Pot={meanG["pot"]:.3e}  Pot-tr={meanG["pot_trap"]:.3e}   (eq. 33/34)')
    print(f'  mean|terminal corr (23)|:  surrogate={tb:.3e}   consistent={tc:.3e}   '
          f'plug-in={tp:.3e}   plug-in corr={tk:.3e}')
    print(f'  final ESS (Girs):                   surrogate={rb["ess"]:.0f}   '
          f'consistent={rc["ess"]:.0f}   plug-in={rp["ess"]:.0f}   plug-in corr={rk["ess"]:.0f}')

    write_latex_table(
        os.path.join(TABLES_DIR, 'toy_t1_diag.tex'),
        'T1 diagnostics ($N={N}$, $K={K}$, $\\kappa={kappa}$, {N_SEEDS} seeds): under the '
        'exact twist the corrected increments vanish (eq.~33) and the pseudo-bootstrap weight '
        'telescopes to $f_T$ (eq.~34); the Girs-vs-Pot bridge gap (eq.~16) is the Pot '
        'quadrature error, smaller for the trapezoidal Pot-tr.  Terminally-consistent '
        'surrogates (consistent, and both plug-in widths) zero the terminal correction '
        '(eq.~23) and remove the last-step ESS collapse; the surrogate twist does '
        'not.  Plug-in corr ($r^2(\\sigma)=\\gamma^2+\\bar c(\\sigma)$) additionally removes '
        'the naive plug-in\'s intermediate-$\\sigma$ overconfidence.'.format(
            N=N, K=K, kappa=kappa, N_SEEDS=N_SEEDS),
        'tab:toy_t1_diag',
        ['quantity', 'value', 'expected'],
        [['PBS telescoping spread $|\\sum\\Delta f_k - f_T|$ (eq.~34)', _fe(eq34), r'$\approx 0$'],
         ['mean per-step $|G_{\\mathrm{Girs}} - G_{\\mathrm{Pot}}|$', _fe(bridge_rect), r'$\approx 0$'],
         ['mean per-step $|G_{\\mathrm{Girs}} - G_{\\mathrm{Pot\\text{-}tr}}|$', _fe(bridge_trap), r'$\approx 0$'],
         ['mean $|G_{\\mathrm{Girs}}|$ (eq.~33)', _fe(meanG['girs']), r'$\approx 0$'],
         ['mean $|G_{\\mathrm{Pot}}|$ (eq.~33)', _fe(meanG['pot']), r'$\approx 0$'],
         ['mean $|G_{\\mathrm{Pot\\text{-}tr}}|$ (eq.~33)', _fe(meanG['pot_trap']), r'$\approx 0$'],
         ['mean $|G_{\\mathrm{PBS}}|$ (eq.~34)', _fe(meanG['pbs']), '>$0$'],
         ['mean $|$terminal corr. (23)$|$, surrogate twist', _fe(tb), '>$0$'],
         ['mean $|$terminal corr. (23)$|$, consistent twist', _fe(tc), r'$\approx 0$'],
         ['mean $|$terminal corr. (23)$|$, plug-in twist', _fe(tp), r'$\approx 0$'],
         ['mean $|$terminal corr. (23)$|$, plug-in corr twist', _fe(tk), r'$\approx 0$'],
         ['final ESS (Girs), sur. / cons. / plug-in / plug-in corr',
          f'{rb["ess"]:.0f} / {rc["ess"]:.0f} / {rp["ess"]:.0f} / {rk["ess"]:.0f}', '--']])


# ----------------------------------------------------------------- T2: base grid (proposal x weighting)
def run_t2(N, K, gamma2=1.0):
    pm, ps = posterior_params(gamma2)[-2:]
    ov = plug_in_corrected_twist(gamma2)
    print(f'\n=== T2: Base grid  (proposal x weighting, plug-in corr twist, N={N}, K={K}, '
          f'gamma2={gamma2}, seeds={N_SEEDS}) ===')
    allres = {}
    for prop in PROPOSALS:
        res = run_cell(N, K, SEEDS, ov, prop, gamma2)
        allres[prop] = res
        _print_metrics(f'T2 proposal={PROPOSAL_LABEL[prop]}',
                       dict(N=N, K=K, proposal=prop, results=res), pm, ps, gamma2)
        write_metrics_table(dict(N=N, K=K, proposal=prop, gamma2=gamma2, results=res),
                            os.path.join(TABLES_DIR, f'toy_t2_{prop}.tex'),
                            f'T2 ({PROPOSAL_LABEL[prop]} proposal, plug-in corr twist '
                            f'($r^2(\\sigma)=\\gamma^2+\\bar c(\\sigma)$), $N={N}$, $K={K}$); '
                            'W1 (eq.~35) and weighted posterior mean/std error.',
                            f'tab:toy_t2_{prop}')
        plot_cell(dict(N=N, K=K, proposal=prop, gamma2=gamma2, results=res),
                  f'T2: {PROPOSAL_LABEL[prop]} proposal, plug-in corr twist',
                  f'toy_t2_{prop}.pdf')

    # compact W1/ESS/resamp: rows = proposal x weighting
    headers = ['cell', 'W1', 'ESS', 'resamp']
    rows = []
    for prop in PROPOSALS:
        for w in WEIGHTS:
            r = allres[prop][w]
            rows.append([f'{PROPOSAL_LABEL[prop]} {WEIGHT_LABEL[w]}',
                         _f4pm(r['w1'], r['w1_std']), _f0pm(r['ess'], r['ess_std']),
                         _f0pm(r['resamples'], r['resamples_std'])])
    write_latex_table(os.path.join(TABLES_DIR, 'toy_t2.tex'),
                      'T2 base grid: proposal $\\times$ weighting at the canonical plug-in corr '
                      f'twist ($r^2(\\sigma)=\\gamma^2+\\bar c(\\sigma)$, $N={N}$, $K={K}$, '
                      f'$\\gamma^2={gamma2}$, {N_SEEDS} seeds, mean$\\pm$std). The EM-vs-Heun '
                      'row gap is the integrator-order effect; the Pot-vs-Pot-tr column gap is '
                      'the quadrature-order effect.',
                      'tab:toy_t2', headers, rows)
    return allres


# ----------------------------------------------------------------- T3: convergence (K- and N-sweeps)
def run_t3(N, K, gamma2=1.0):
    ov = plug_in_corrected_twist(gamma2)
    print(f'\n=== T3: Convergence  (K- and N-sweeps over the base grid, plug-in corr twist, '
          f'gamma2={gamma2}, seeds={N_SEEDS}) ===')

    # K-sweep (discretisation error)
    print(f'\n-- K sweep  (N={N})')
    print(f'  {"cell":>16}  ' + ''.join(f'{Kk:>14}' for Kk in K_SWEEP))
    print('  ' + '-' * (17 + 14 * len(K_SWEEP)))
    series_k = {prop: {w: [] for w in WEIGHTS} for prop in PROPOSALS}
    std_k = {prop: {w: [] for w in WEIGHTS} for prop in PROPOSALS}
    for Kk in K_SWEEP:
        for prop in PROPOSALS:
            res = run_cell(N, Kk, SEEDS, ov, prop, gamma2)
            for w in WEIGHTS:
                series_k[prop][w].append(res[w]['w1'])
                std_k[prop][w].append(res[w]['w1_std'])
    for prop in PROPOSALS:
        for w in WEIGHTS:
            print(f'  {PROPOSAL_LABEL[prop]} {WEIGHT_LABEL[w]:>9}  ' +
                  ''.join(f'{_f4pm(v, s):>14}' for v, s in zip(series_k[prop][w], std_k[prop][w])))
    rows = [[f'{PROPOSAL_LABEL[prop]} {WEIGHT_LABEL[w]}'] +
            [_f4pm(v, s) for v, s in zip(series_k[prop][w], std_k[prop][w])]
            for prop in PROPOSALS for w in WEIGHTS]
    write_latex_table(os.path.join(TABLES_DIR, 'toy_t3_k.tex'),
                      f'T3 K-sweep (plug-in corr twist '
                      f'$r^2(\\sigma)=\\gamma^2+\\bar c(\\sigma)$, $N={N}$, '
                      f'$\\gamma^2={gamma2}$, {N_SEEDS} seeds, mean$\\pm$std): corrected W1 is '
                      'elevated at coarse $K$ (discretisation error) and drops to the '
                      'ESS/Monte-Carlo floor as $K$ grows (eq.~33); PBS retains a structural '
                      'bias (eq.~34) independent of $K$.',
                      'tab:toy_t3_k', ['cell'] + [f'$K={Kk}$' for Kk in K_SWEEP], rows)
    plot_sweep(K_SWEEP, series_k, os.path.join(FIGS_DIR, 'toy_t3_k.pdf'),
               f'T3: step-count sweep  (N={N}, plug-in corr twist)', 'K')

    # N-sweep (Monte-Carlo error)
    print(f'\n-- N sweep  (K={K})')
    print(f'  {"cell":>16}  ' + ''.join(f'{Nn:>14}' for Nn in N_SWEEP))
    print('  ' + '-' * (17 + 14 * len(N_SWEEP)))
    series_n = {prop: {w: [] for w in WEIGHTS} for prop in PROPOSALS}
    std_n = {prop: {w: [] for w in WEIGHTS} for prop in PROPOSALS}
    for Nn in N_SWEEP:
        for prop in PROPOSALS:
            res = run_cell(Nn, K, SEEDS, ov, prop, gamma2)
            for w in WEIGHTS:
                series_n[prop][w].append(res[w]['w1'])
                std_n[prop][w].append(res[w]['w1_std'])
    for prop in PROPOSALS:
        for w in WEIGHTS:
            print(f'  {PROPOSAL_LABEL[prop]} {WEIGHT_LABEL[w]:>9}  ' +
                  ''.join(f'{_f4pm(v, s):>14}' for v, s in zip(series_n[prop][w], std_n[prop][w])))
    rows = [[f'{PROPOSAL_LABEL[prop]} {WEIGHT_LABEL[w]}'] +
            [_f4pm(v, s) for v, s in zip(series_n[prop][w], std_n[prop][w])]
            for prop in PROPOSALS for w in WEIGHTS]
    write_latex_table(os.path.join(TABLES_DIR, 'toy_t3_n.tex'),
                      f'T3 N-sweep (plug-in corr twist '
                      f'$r^2(\\sigma)=\\gamma^2+\\bar c(\\sigma)$, $K={K}$, '
                      f'$\\gamma^2={gamma2}$, {N_SEEDS} seeds, mean$\\pm$std): corrected W1 '
                      f'follows $1/\\sqrt{{N}}\\to 0$ as $N$ grows (no unremovable bias); PBS is '
                      'flat (structural bias, eq.~34), independent of $N$.',
                      'tab:toy_t3_n', ['cell'] + [f'$N={Nn}$' for Nn in N_SWEEP], rows)
    plot_sweep(N_SWEEP, series_n, os.path.join(FIGS_DIR, 'toy_t3_n.pdf'),
               f'T3: particle-count sweep  (K={K}, plug-in corr twist)', 'N')
    return series_k, series_n


# ----------------------------------------------------------------- T4: regime (gamma2-sweep)
def run_t4(N, K):
    print(f'\n=== T4: Regime  (gamma2-sweep over the base grid, plug-in corr twist, N={N}, '
          f'K={K}, seeds={N_SEEDS}) ===')
    series = {prop: {w: [] for w in WEIGHTS} for prop in PROPOSALS}
    std = {prop: {w: [] for w in WEIGHTS} for prop in PROPOSALS}
    ess = {prop: {w: [] for w in WEIGHTS} for prop in PROPOSALS}
    ess_std = {prop: {w: [] for w in WEIGHTS} for prop in PROPOSALS}
    resamp = {prop: {w: [] for w in WEIGHTS} for prop in PROPOSALS}
    resamp_std = {prop: {w: [] for w in WEIGHTS} for prop in PROPOSALS}
    print(f'  {"cell":>16}  ' + ''.join(f'{g2:>14}' for g2 in GAMMA2_SWEEP))
    print('  ' + '-' * (17 + 14 * len(GAMMA2_SWEEP)))
    for g2 in GAMMA2_SWEEP:
        ov = plug_in_corrected_twist(g2)
        for prop in PROPOSALS:
            res = run_cell(N, K, SEEDS, ov, prop, g2)
            for w in WEIGHTS:
                series[prop][w].append(res[w]['w1'])
                std[prop][w].append(res[w]['w1_std'])
                ess[prop][w].append(res[w]['ess'])
                ess_std[prop][w].append(res[w]['ess_std'])
                resamp[prop][w].append(res[w]['resamples'])
                resamp_std[prop][w].append(res[w]['resamples_std'])
    for prop in PROPOSALS:
        for w in WEIGHTS:
            print(f'  {PROPOSAL_LABEL[prop]} {WEIGHT_LABEL[w]:>9}  ' +
                  ''.join(f'{_f4pm(v, s):>14}' for v, s in zip(series[prop][w], std[prop][w])))
    print('  -- resampling events --')
    for prop in PROPOSALS:
        for w in WEIGHTS:
            print(f'  {PROPOSAL_LABEL[prop]} {WEIGHT_LABEL[w]:>9}  ' +
                  ''.join(f'{_f0pm(v, s):>14}' for v, s in zip(resamp[prop][w], resamp_std[prop][w])))
    rows = [[f'{PROPOSAL_LABEL[prop]} {WEIGHT_LABEL[w]}'] +
            [_f4pm(v, s) for v, s in zip(series[prop][w], std[prop][w])]
            for prop in PROPOSALS for w in WEIGHTS]
    write_latex_table(os.path.join(TABLES_DIR, 'toy_t4.tex'),
                      f'T4 regime: exact observation variance sweep with the plug-in corr twist '
                      f'($r^2(\\sigma)=\\gamma^2+\\bar c(\\sigma)$, $N={N}$, $K={K}$, '
                      f'{N_SEEDS} seeds, mean$\\pm$std).  As $\\gamma^2$ shrinks the posterior '
                      'sharpens relative to the prior and the corrected weightings degenerate '
                      '(observation-informativeness limit, not surrogate misspecification -- '
                      'the residual-corrected width removes the naive plug-in\'s '
                      'overconfidence); this locates the observational regime in which guided '
                      'SMC works vs fails.',
                      'tab:toy_t4', ['cell'] + [f'$\\gamma^2={g2}$' for g2 in GAMMA2_SWEEP], rows)
    plot_sweep(GAMMA2_SWEEP, series, os.path.join(FIGS_DIR, 'toy_t4.pdf'),
               f'T4: observation-variance sweep  (N={N}, K={K}, plug-in corr twist)',
               r'$\gamma^2$', xscale='log', yscale='log')

    # ESS across the sweep: where does degeneracy set in?
    ess_rows = [[f'{PROPOSAL_LABEL[prop]} {WEIGHT_LABEL[w]}'] +
                [_f0pm(v, s) for v, s in zip(ess[prop][w], ess_std[prop][w])]
                for prop in PROPOSALS for w in WEIGHTS]
    write_latex_table(os.path.join(TABLES_DIR, 'toy_t4_ess.tex'),
                      f'T4 final effective sample size by cell '
                      f'($N={N}$, $K={K}$, plug-in corr twist, {N_SEEDS} seeds, mean$\\pm$std).',
                      'tab:toy_t4_ess', ['cell'] + [f'$\\gamma^2={g2}$' for g2 in GAMMA2_SWEEP],
                      ess_rows)

    # resampling count across the sweep: the regime boundary (weight degeneracy)
    resamp_rows = [[f'{PROPOSAL_LABEL[prop]} {WEIGHT_LABEL[w]}'] +
                   [_f0pm(v, s) for v, s in zip(resamp[prop][w], resamp_std[prop][w])]
                   for prop in PROPOSALS for w in WEIGHTS]
    write_latex_table(os.path.join(TABLES_DIR, 'toy_t4_resamp.tex'),
                      f'T4 resampling events by cell '
                      f'($N={N}$, $K={K}$, plug-in corr twist, {N_SEEDS} seeds, mean$\\pm$std).  '
                      'With the residual-corrected plug-in surrogate the corrected weightings '
                      'degenerate as $\\gamma^2$ shrinks purely from posterior sharpness '
                      '(observation-informativeness limit): the resampling count rises with '
                      'shrinking $\\gamma^2$, without the surrogate-induced overconfidence of '
                      'the naive plug-in.',
                      'tab:toy_t4_resamp',
                      ['cell'] + [f'$\\gamma^2={g2}$' for g2 in GAMMA2_SWEEP], resamp_rows)
    return series, ess


# ----------------------------------------------------------------- main
def main():
    N, K = 2048, 500
    gamma2, kappa = 1.0, KAPPA
    t0 = time.time()
    _setup_style()

    eb, eH, es = selfcheck()
    print(f'closed-form self-check  max|d(b) - FD(loglik)|={eb:.2e}  '
          f'max|d(H) - FD(b)|={eH:.2e}  max|score - FD(logpdf)|={es:.2e}')

    run_t1(N, K, gamma2, kappa)
    run_t2(N, K, gamma2)
    run_t3(N, K, gamma2)
    run_t4(N, K)

    print(f'\nDone ({time.time() - t0:.0f}s).')


if __name__ == '__main__':
    main()