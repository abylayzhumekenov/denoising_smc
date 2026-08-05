"""Minimal toy SMC demo: 1D Gaussian-mixture prior with known posterior.

Validates the lambda-rho weighting machinery (docs/recipe.md) against an analytic
posterior, across four variants:
  1. ODE          - deterministic guided reverse ODE, no SMC (repo baseline)
  2. SOSaG+pBS    - Millard et al., lam=0 (pseudo-bootstrap)
  3. GEM+Girsanov - first-order, lam=1
  4. HeunSDE+Girs - second-order stochastic Heun, lam=1

Run: python3 scripts/toy_smc.py
"""

import resource
import time

import numpy as np
from scipy.special import erf

resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024))  # safety net: never exceed ~4GB

# ----------------------------------------------------------------- model
W = np.array([0.6, 0.4])
MU = np.array([-1.5, 2.0])
VAR = np.array([0.7, 1.3])
Y = 0.8
OBS_VAR = 0.5


def score(x, s2):
    """d/dx log p_t(x) of the noised mixture: p_t = sum_m w_m N(mu_m, var_m + s2)."""
    v = VAR + s2
    logc = np.log(W) - 0.5 * np.log(2 * np.pi * v) - (x - MU) ** 2 / (2 * v)
    logc = logc - logc.max(axis=-1, keepdims=True)
    pi = np.exp(logc)
    pi = pi / pi.sum(axis=-1, keepdims=True)
    return (pi * (MU - x) / v).sum(axis=-1, keepdims=True)


def loglik(x):
    """log p_tilde(y|x), Gaussian observation model."""
    return -(x - Y) ** 2 / (2 * OBS_VAR)


def grad_loglik(x):
    """b(x) = d/dx log p_tilde(y|x)."""
    return (Y - x) / OBS_VAR


def posterior_params():
    """Posterior mixture params (Kalman update per component): abar, mu_p, var_p, mean, std."""
    S = VAR + OBS_VAR
    q = W * np.exp(-0.5 * (Y - MU) ** 2 / S) / np.sqrt(2 * np.pi * S)
    abar = q / q.sum()
    var_p = 1 / (1 / VAR + 1 / OBS_VAR)
    mu_p = var_p * (MU / VAR + Y / OBS_VAR)
    pm = (abar * mu_p).sum()
    pv = (abar * (var_p + mu_p ** 2)).sum() - pm ** 2
    return abar, mu_p, var_p, float(pm), float(np.sqrt(max(pv, 0)))


def posterior_stats():
    """Analytic posterior mean/std (Kalman update per component)."""
    _, _, _, pm, ps = posterior_params()
    return pm, ps


# ----------------------------------------------------------------- schedule
SIGMA_MAX, SIGMA_MIN, K, RHO = 6.0, 1e-3, 500, 7.0
idx = np.arange(K)
sigma = (SIGMA_MAX ** (1 / RHO) + idx / (K - 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
sigma = np.concatenate([sigma, [0.0]])  # final step ends at sigma = 0


# ----------------------------------------------------------------- proposals
def gem(x, sk, skm1, rng):
    """First-order Euler-Maruyama with guidance drift."""
    sk2, skm12 = sk ** 2, skm1 ** 2
    delta = sk2 - skm12
    b = grad_loglik(x)
    z = rng.standard_normal(x.shape)
    x_km1 = x + delta * (score(x, sk2) + b) + np.sqrt(delta) * z
    return x_km1, {'z': z, 'grad': b, 'delta': delta}


def heunsde(x, sk, skm1, rng):
    """Second-order stochastic Heun, same Brownian increment throughout."""
    sk2, skm12 = sk ** 2, skm1 ** 2
    delta = sk2 - skm12
    z = rng.standard_normal(x.shape)
    s_k = score(x, sk2)
    b_k = grad_loglik(x)
    x_pred = x + delta * (s_k + b_k) + np.sqrt(delta) * z
    s_p = score(x_pred, skm12)
    b_p = grad_loglik(x_pred)
    x_km1 = x + 0.5 * delta * (s_k + b_k + s_p + b_p) + np.sqrt(delta) * z
    return x_km1, {'z': z, 'grad': b_k, 'delta': delta}


def sosag(x, sk, skm1, rng, gamma=0.2):
    """Jittering + Heun ODE + guidance correction (Millard et al.)."""
    hat_sigma = sk + gamma * sk
    psi = rng.standard_normal(x.shape)
    hat_x = x + np.sqrt(hat_sigma ** 2 - sk ** 2) * psi
    d_k = -hat_sigma * score(hat_x, hat_sigma ** 2)
    x_pred = hat_x + (skm1 - hat_sigma) * d_k
    d_km1 = -skm1 * score(x_pred, skm1 ** 2)
    x_den = hat_x + (skm1 - hat_sigma) * 0.5 * (d_k + d_km1)
    delta = sk ** 2 - skm1 ** 2
    b = grad_loglik(hat_x)
    x_km1 = x_den + delta * b
    return x_km1, {'grad': b, 'delta': delta}


# ----------------------------------------------------------------- weight
def weight(dll, aux, lam, rho):
    """Unified lambda-rho weight: rho*Delta_loglik + lam*C_k."""
    if lam == 0:
        return rho * dll
    delta = aux['delta']
    g, z = aux['grad'], aux['z']
    C = -np.sqrt(delta) * (g * z).sum(axis=-1) - 0.5 * delta * (g ** 2).sum(axis=-1)
    return rho * dll + lam * C


# ----------------------------------------------------------------- filter
def exact_init(N, rng):
    """Sample from p_{sigma_max} (the exact noised marginal), not from Gaussian."""
    m = rng.choice(len(W), size=N, p=W)
    return (MU[m] + np.sqrt(VAR[m] + SIGMA_MAX ** 2) * rng.standard_normal(N))[:, None]


def run_filter(proposal, lam, rho, N, seed):
    rng = np.random.default_rng(seed)
    x = exact_init(N, rng)
    log_w = np.zeros(N)
    n_resample = 0
    for step in range(K):
        sk, skm1 = sigma[step], sigma[step + 1]
        x_km1, aux = proposal(x, sk, skm1, rng)
        dll = (loglik(x_km1) - loglik(x))[:, 0]
        log_w = log_w + weight(dll, aux, lam, rho)
        log_w = log_w - np.log(np.sum(np.exp(log_w)))
        if 1.0 / np.sum(np.exp(2 * log_w)) < 0.5 * N:
            idx = rng.choice(N, size=N, p=np.exp(log_w))
            x_km1 = x_km1[idx]
            log_w = np.zeros(N)
            n_resample += 1
        x = x_km1
    ess = 1.0 / np.sum(np.exp(2 * log_w))
    return x, log_w, float(ess), n_resample


def run_ode(N, seed):
    """Deterministic Heun on the guided drift score + b (no weights/resampling)."""
    rng = np.random.default_rng(seed)
    x = exact_init(N, rng)
    for step in range(K):
        sk, skm1 = sigma[step], sigma[step + 1]
        delta = sk ** 2 - skm1 ** 2
        k1 = score(x, sk ** 2) + grad_loglik(x)
        x_pred = x + delta * k1
        k2 = score(x_pred, skm1 ** 2) + grad_loglik(x_pred)
        x = x + 0.5 * delta * (k1 + k2)
    return x


def weighted_stats(x, log_w):
    """Weighted posterior mean/std from the final normalized log-weights."""
    l = log_w - np.max(log_w)
    w = np.exp(l)
    w = w / w.sum()
    m = float((w * x[:, 0]).sum())
    v = float((w * x[:, 0] ** 2).sum()) - m ** 2
    return m, float(np.sqrt(max(v, 0)))


def posterior_cdf(xx):
    """Analytic posterior mixture CDF."""
    abar, mu_p, var_p, _, _ = posterior_params()
    return (abar[:, None] * 0.5 * (1 + erf((xx[None, :] - mu_p[:, None]) / np.sqrt(2 * var_p[:, None])))).sum(axis=0)


def wasserstein1(samples, lo=-6, hi=6, grid=8000):
    """W1 between the particle cloud and the analytic posterior (1D closed form)."""
    g = np.linspace(lo, hi, grid)
    Fp = posterior_cdf(g)
    Fn = np.searchsorted(np.sort(samples[:, 0]), g) / len(samples)
    return float(np.trapezoid(np.abs(Fn - Fp), g))


def verify_gem_weight(seed=0):
    """Check GEM Girsanov C_k against the closed-form Gaussian log density ratio
    log(p_P/p_Q) = -(1/2 delta)[|x_km1-mu_p|^2 - |x_km1-mu_g|^2] (docs/idea.md Sec 4)."""
    rng = np.random.default_rng(seed)
    x = exact_init(8, rng)
    max_err = 0.0
    for step in range(K):
        sk, skm1 = sigma[step], sigma[step + 1]
        sk2, skm12 = sk ** 2, skm1 ** 2
        delta = sk2 - skm12
        b = grad_loglik(x)
        s = score(x, sk2)
        mu_p = x + delta * s
        mu_g = mu_p + delta * b
        z = rng.standard_normal(x.shape)
        x_km1 = x + delta * (s + b) + np.sqrt(delta) * z
        C_girs = -np.sqrt(delta) * (b * z).sum(axis=-1) - 0.5 * delta * (b ** 2).sum(axis=-1)
        C_closed = -(1 / (2 * delta)) * (((x_km1 - mu_p) ** 2 - (x_km1 - mu_g) ** 2)).sum(axis=-1)
        max_err = max(max_err, float(np.max(np.abs(C_girs - C_closed))))
        x = x_km1
    return max_err


# ----------------------------------------------------------------- main
def main():
    pm, ps = posterior_stats()
    print(f'analytic posterior: mean={pm:.4f}  std={ps:.4f}')
    print(f'GEM C_k vs closed-form Gaussian ratio: max|diff| = {verify_gem_weight():.2e}')

    N = 128
    variants = [
        ('ODE', lambda s: (run_ode(N, s), None, 0, 0)),
        ('SOSaG+pBS', lambda s: run_filter(sosag, 0.0, 1.0, N, s)),
        ('GEM+pBS', lambda s: run_filter(gem, 0.0, 1.0, N, s)),
        ('HeunSDE+pBS', lambda s: run_filter(heunsde, 0.0, 1.0, N, s)),
        ('GEM+Girs', lambda s: run_filter(gem, 1.0, 1.0, N, s)),
        ('HeunSDE+Girs', lambda s: run_filter(heunsde, 1.0, 1.0, N, s)),
    ]
    seeds = 8
    fmt = lambda a, s: f'{a:.3f}+-{s:.3f}'
    print(f'\nmain table (N={N}, K={K}, {seeds} seeds, mean+-SEM)')
    print(f"{'variant':<14}{'W1':>16}{'|dmean|':>15}{'|dstd|':>15}{'ESS':>9}{'resamp':>8}{'sec':>7}")
    for name, fn in variants:
        t0 = time.time()
        rows = []
        for s in range(seeds):
            x, log_w, ess, nr = fn(s)
            wmean, wstd = weighted_stats(x, log_w) if log_w is not None else (float(x.mean()), float(x.std()))
            rows.append((wasserstein1(x), abs(wmean - pm), abs(wstd - ps), ess, nr))
        dt = (time.time() - t0) / seeds
        cols = [np.array([r[i] for r in rows]) for i in range(5)]
        sem = lambda a: a.std(ddof=1) / np.sqrt(seeds)
        print(f'{name:<14}{fmt(cols[0].mean(), sem(cols[0])):>16}'
              f'{fmt(cols[1].mean(), sem(cols[1])):>15}{fmt(cols[2].mean(), sem(cols[2])):>15}'
              f'{cols[3].mean():>9.0f}{cols[4].mean():>8.1f}{dt:>7.1f}')

    print(f'\nN-sweep (K={K}, {seeds} seeds, lam=1 arms, mean+-SEM):')
    print(f"{'N':>6}{'GEM W1':>16}{'GEM |dstd|':>15}{'GEM ESS':>10}{'Heun W1':>16}{'Heun |dstd|':>15}{'Heun ESS':>10}")
    for Nn in (128, 512, 2048):
        cells = [f'{Nn:>6}']
        for prop in (gem, heunsde):
            acc = []
            for s in range(seeds):
                x, log_w, ess, _ = run_filter(prop, 1.0, 1.0, Nn, s)
                _, wstd = weighted_stats(x, log_w)
                acc.append((wasserstein1(x), abs(wstd - ps), ess))
            c = [np.array([r[i] for r in acc]) for i in range(3)]
            cells.append(fmt(c[0].mean(), sem(c[0])).rjust(16))
            cells.append(fmt(c[1].mean(), sem(c[1])).rjust(15))
            cells.append(f'{c[2].mean():.0f}'.rjust(10))
        print(''.join(cells))


if __name__ == '__main__':
    main()
