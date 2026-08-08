"""Validation toy: 1D Gaussian-mixture SMC with exact intermediate likelihood.

Implements the λ-ρ unified weight (note_2.tex eq 18):
  log w_{k-1} = log w_k + ρ·[log p̃(y|x_{k-1}) - log p̃(y|x_k)] + λ·C_k

Samplers:
  run_smc — GEM proposal + left-endpoint Girsanov (:= TDS when λ=1)
  run_ode — deterministic guided Heun ODE (DiffusionPDE baseline)

Experiments (each callable independently):
  sweep_lambda  — W1 / |dstd| / ESS vs λ, returns data dict
  sweep_K       — W1 / |dstd| / ESS vs K, returns data dict
  plot_lambda_experiment — 2×2 figure from λ-sweep data

Run: venv/bin/python smc/toy_smc.py
"""

import os, resource, time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, logsumexp
from scipy.stats import gaussian_kde, norm

resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024))

# ----------------------------------------------------------------- model
W = np.array([0.6, 0.4])
MU = np.array([-1.5, 2.0])
VAR = np.array([0.7, 1.3])
obs_y = 0.0
obs_var = 1.0

# ----------------------------------------------------------------- schedule
SIGMA_MAX, SIGMA_MIN, RHO_SCHED = 6.0, 1e-3, 7.0


def build_sigma(k):
    idx = np.arange(k)
    s = (SIGMA_MAX ** (1 / RHO_SCHED) + idx / (k - 1) *
         (SIGMA_MIN ** (1 / RHO_SCHED) - SIGMA_MAX ** (1 / RHO_SCHED))) ** RHO_SCHED
    return np.concatenate([s, [0.0]])


# ----------------------------------------------------------------- posterior
def posterior_params():
    S = VAR + obs_var
    q = W * np.exp(-0.5 * (obs_y - MU) ** 2 / S) / np.sqrt(2 * np.pi * S)
    abar = q / q.sum()
    var_p = 1 / (1 / VAR + 1 / obs_var)
    mu_p = var_p * (MU / VAR + obs_y / obs_var)
    pm = (abar * mu_p).sum()
    pv = (abar * (var_p + mu_p ** 2)).sum() - pm ** 2
    return abar, mu_p, var_p, float(pm), float(np.sqrt(max(pv, 0)))


def posterior_cdf(xx):
    abar, mu_p, var_p, _, _ = posterior_params()
    return (abar[:, None] *
            0.5 * (1 + erf((xx[None, :] - mu_p[:, None]) /
                           np.sqrt(2 * var_p[:, None])))).sum(axis=0)


def posterior_pdf(xx):
    abar, mu_p, var_p, _, _ = posterior_params()
    return (abar[:, None] *
            norm.pdf(xx[None, :], mu_p[:, None], np.sqrt(var_p[:, None]))).sum(axis=0)


def wasserstein1(samples, lo=-6, hi=6, grid=8000):
    g = np.linspace(lo, hi, grid)
    Fp = posterior_cdf(g)
    Fn = np.searchsorted(np.sort(samples[:, 0]), g) / len(samples)
    return float(np.trapezoid(np.abs(Fn - Fp), g))


# ----------------------------------------------------------------- score
def noised_logpdf(x, sigma2):
    v = VAR + sigma2
    return np.log(W) - 0.5 * np.log(2 * np.pi * v) - 0.5 * (x - MU) ** 2 / v


def score(x, sigma2):
    v = VAR + sigma2
    logc = noised_logpdf(x, sigma2)
    logc = logc - logc.max(axis=-1, keepdims=True)
    pi = np.exp(logc)
    pi = pi / pi.sum(axis=-1, keepdims=True)
    return (pi * (MU - x) / v).sum(axis=-1, keepdims=True)


# ----------------------------------------------------------------- exact intermediate likelihood and guidance
def _lik_stats(x, sigma):
    sigma2 = sigma ** 2
    A = VAR + sigma2
    r_coef = VAR / A
    c = VAR * sigma2 / A
    s2 = obs_var + c

    logb_raw = noised_logpdf(x, sigma2)
    logb_max = logb_raw.max(axis=-1, keepdims=True)
    logb = logb_raw - logb_max
    b = np.exp(logb)
    Z = b.sum(axis=-1, keepdims=True)
    pi = b / Z
    log_pi = logb - np.log(Z)
    s = (pi * (MU - x) / A).sum(axis=-1, keepdims=True)

    m_m = MU + r_coef * (x - MU)
    log_phi = -0.5 * np.log(2 * np.pi * s2) - 0.5 * (obs_y - m_m) ** 2 / s2

    log_prod = log_pi + log_phi
    log_pyx = logsumexp(log_prod, axis=-1)
    xi = np.exp(log_prod - log_pyx[:, None])

    term1 = -(x - MU) / A
    term2 = (obs_y - m_m) * r_coef / s2
    grad = (xi * (term1 + term2)).sum(axis=-1, keepdims=True) - s

    return {'log_lik': log_pyx, 'grad': grad}


def exact_loglik(x, sigma):
    return _lik_stats(x, sigma)['log_lik']


def guidance(x, sigma):
    return _lik_stats(x, sigma)['grad']


# ----------------------------------------------------------------- proposals (kept for future use)
def gem(x, sk, skm1, rng):
    delta = sk ** 2 - skm1 ** 2
    z = rng.standard_normal(x.shape)
    b_k = guidance(x, sk)
    s_k = score(x, sk ** 2)
    x_km1 = x + delta * (s_k + b_k) + np.sqrt(delta) * z
    aux = {'z': z, 'b_k': b_k, 'delta': delta}
    return x_km1, aux


def heunsde(x, sk, skm1, rng):
    delta = sk ** 2 - skm1 ** 2
    sd = np.sqrt(delta)
    z = rng.standard_normal(x.shape)
    s_k = score(x, sk ** 2)
    b_k = guidance(x, sk)
    x_pred = x + delta * (s_k + b_k) + sd * z
    s_p = score(x_pred, skm1 ** 2)
    b_p = guidance(x_pred, skm1)
    x_km1 = x + 0.5 * delta * (s_k + b_k + s_p + b_p) + sd * z
    aux = {'z': z, 'b_k': b_k, 'delta': delta}
    return x_km1, aux


# ----------------------------------------------------------------- core samplers
def init_particles(N, rng):
    m = rng.choice(len(W), size=N, p=W)
    return (MU[m] + np.sqrt(VAR[m] + SIGMA_MAX ** 2) *
            rng.standard_normal(N))[:, None]


def run_smc(N, K, seed, lam=1.0, rho=1.0):
    """GEM proposal + λ-ρ unified weight. Returns dict."""
    sig = build_sigma(K)
    rng = np.random.default_rng(seed)
    x = init_particles(N, rng)

    log_w = rho * exact_loglik(x, SIGMA_MAX)
    n_resample = 0
    ess_history = []

    for step in range(K):
        sk, skm1 = sig[step], sig[step + 1]

        log_lik_k = exact_loglik(x, sk)

        x_km1, aux = gem(x, sk, skm1, rng)

        log_lik_km1 = exact_loglik(x_km1, skm1)
        dll = log_lik_km1 - log_lik_k

        b = aux['b_k'][:, 0]
        z = aux['z'][:, 0]
        delta = aux['delta']
        sd = np.sqrt(delta)
        C = -sd * b * z - 0.5 * delta * b ** 2

        log_w = log_w + rho * dll + lam * C
        log_w = log_w - logsumexp(log_w)
        ess = 1.0 / np.sum(np.exp(2 * log_w))
        ess_history.append(float(ess))

        if ess < 0.5 * N:
            idx = rng.choice(N, size=N, p=np.exp(log_w))
            x_km1 = x_km1[idx]
            log_w = np.zeros(N)
            n_resample += 1

        x = x_km1

    final_ess = 1.0 / np.sum(np.exp(2 * log_w))
    return {'particles': x, 'weights': log_w, 'ess': float(final_ess),
            'ess_history': np.array(ess_history), 'resamples': n_resample}


def run_ode(N, K, seed):
    """Deterministic guided Heun ODE. Returns particles array."""
    sig = build_sigma(K)
    rng = np.random.default_rng(seed)
    x = init_particles(N, rng)

    for step in range(K):
        sk, skm1 = sig[step], sig[step + 1]
        delta = sk ** 2 - skm1 ** 2
        s_k = score(x, sk ** 2)
        b_k = guidance(x, sk)
        x_pred = x + delta * (s_k + b_k)
        s_p = score(x_pred, skm1 ** 2)
        b_p = guidance(x_pred, skm1)
        x = x + 0.5 * delta * (s_k + b_k + s_p + b_p)

    return x


def weighted_stats(x, log_w):
    lw = log_w - np.max(log_w)
    w = np.exp(lw)
    w = w / w.sum()
    m = float((w * x[:, 0]).sum())
    v = float((w * x[:, 0] ** 2).sum()) - m ** 2
    return m, float(np.sqrt(max(v, 0)))


# ----------------------------------------------------------------- sweeps
FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)


def sweep_lambda(lambdas, K=2000, N=128, seeds=8, rho=1.0):
    """λ sweep: print table, return {lam: {w1, dstd, ess, ...}}."""
    pm, ps_ref = posterior_params()[-2:]
    print(f'analytic posterior: mean={pm:.4f}  std={ps_ref:.4f}')
    print(f'λ-sweep  K={K}  N={N}  seeds={seeds}\n')

    header = (f'{"λ":>6}  {"W1":>12}  {"|dmean|":>11}  {"|dstd|":>12}  '
              f'{"ESS":>8}  {"resamp":>7}')
    print(header)
    print('-' * len(header))

    data = {}
    for lam in lambdas:
        w1_list, dm_list, ds_list, ess_list = [], [], [], []
        for s in range(seeds):
            r = run_smc(N, K, s, lam=lam, rho=rho)
            wm, ws = weighted_stats(r['particles'], r['weights'])
            w1_list.append(wasserstein1(r['particles']))
            dm_list.append(abs(wm - pm))
            ds_list.append(abs(ws - ps_ref))
            ess_list.append(r['ess'])

        sem = lambda a: np.std(a, ddof=1) / np.sqrt(seeds)
        data[lam] = {
            'w1': (np.mean(w1_list), sem(w1_list)),
            'dmean': (np.mean(dm_list), sem(dm_list)),
            'dstd': (np.mean(ds_list), sem(ds_list)),
            'ess': (np.mean(ess_list), sem(ess_list)),
        }
        print(f'{lam:>6.2f}  {np.mean(w1_list):>12.4f}  {np.mean(dm_list):>11.4f}  '
              f'{np.mean(ds_list):>12.4f}  {np.mean(ess_list):>8.0f}  {0:>7}')
    return data


def sweep_K(K_values, N=128, seeds=8, lam=1.0, rho=1.0):
    """K sweep: print table, return {K: {w1, dstd, ess, ...}}."""
    pm, ps_ref = posterior_params()[-2:]
    print(f'analytic posterior: mean={pm:.4f}  std={ps_ref:.4f}')
    print(f'K-sweep  N={N}  seeds={seeds}  λ={lam}  ρ={rho}\n')

    header = (f'{"K":>6}  {"W1":>12}  {"|dmean|":>11}  {"|dstd|":>12}  '
              f'{"ESS":>8}  {"resamp":>7}')
    print(header)
    print('-' * len(header))

    data = {}
    for K in K_values:
        w1_list, dm_list, ds_list, ess_list = [], [], [], []
        for s in range(seeds):
            r = run_smc(N, K, s, lam=lam, rho=rho)
            wm, ws = weighted_stats(r['particles'], r['weights'])
            w1_list.append(wasserstein1(r['particles']))
            dm_list.append(abs(wm - pm))
            ds_list.append(abs(ws - ps_ref))
            ess_list.append(r['ess'])

        sem = lambda a: np.std(a, ddof=1) / np.sqrt(seeds)
        data[K] = {
            'w1': (np.mean(w1_list), sem(w1_list)),
            'dmean': (np.mean(dm_list), sem(dm_list)),
            'dstd': (np.mean(ds_list), sem(ds_list)),
            'ess': (np.mean(ess_list), sem(ess_list)),
        }
        print(f'{K:>6}  {np.mean(w1_list):>12.4f}  {np.mean(dm_list):>11.4f}  '
              f'{np.mean(ds_list):>12.4f}  {np.mean(ess_list):>8.0f}  {0:>7}')
    return data


# ----------------------------------------------------------------- plots
def plot_lambda_experiment(data, K, N, seed, lambdas, rho=1.0):
    """2×2 figure: W1/|dstd|/ESS vs λ (top row) + density comparison (bottom row)."""
    xs = sorted(data.keys())
    colors_smc = plt.cm.Blues(np.linspace(0.4, 0.9, len(xs)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # -- top row: sweep panels (single blue line, no legend)
    def _panel(ax, metric, ylabel):
        ys = [data[x][metric][0] for x in xs]
        ses = [data[x][metric][1] for x in xs]
        ax.errorbar(xs, ys, yerr=ses, color='#1f77b4', marker='o',
                    ecolor='#7eb8da', markersize=6, capsize=3, linewidth=1.2)
        ax.set_xlabel('λ')
        ax.set_ylabel(ylabel)

    _panel(axes[0, 0], 'w1', 'W1')
    _panel(axes[0, 1], 'dstd', '|dstd|')
    _panel(axes[1, 0], 'ess', 'ESS')

    # -- bottom-right: density comparison
    ax = axes[1, 1]
    arm_data = [('ODE', run_ode(N, K, seed), None)]
    for lam in lambdas:
        r = run_smc(N, K, seed, lam=lam, rho=rho)
        arm_data.append((f'λ={lam:.2f}', r['particles'], r['weights']))

    all_x = np.concatenate([x[:, 0] for _, x, _ in arm_data])
    lo, hi = float(all_x.min()) - 0.3, float(all_x.max()) + 0.3
    xgrid = np.linspace(lo, hi, 800)

    ax.plot(xgrid, posterior_pdf(xgrid), 'k--', linewidth=2.0,
            label='Analytic posterior', zorder=10)
    ax.plot([], [], '|', color='gray', markersize=8, markeredgewidth=1.0,
            label='ODE')

    for label, x, lw in arm_data:
        xf = x[:, 0]
        if label == 'ODE':
            ax.plot(xf, np.full_like(xf, -0.02), '|', color='gray',
                    markersize=8, markeredgewidth=1.0, alpha=0.4)
        else:
            w = np.exp(lw - np.max(lw)); w = w / w.sum()
            lam_idx = lambdas.index(float(label.split('=')[1]))
            kde = gaussian_kde(xf, weights=w, bw_method='scott')
            ax.plot(xgrid, kde(xgrid), color=colors_smc[lam_idx], alpha=0.8,
                    linewidth=1.2, label=label)

    ax.set_xlabel('x'); ax.set_ylabel('Density')
    ax.legend(fontsize=7, loc='upper left')

    fig.suptitle(f'λ experiment  (K={K}, N={N})', fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIGS_DIR, 'lambda_experiment.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


# ----------------------------------------------------------------- main
def main():
    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    K, N, seeds = 2000, 512, 8
    t_start = time.time()

    data = sweep_lambda(lambdas, K=K, N=N, seeds=seeds)
    print(f'  ({time.time() - t_start:.0f}s)\n')

    plot_lambda_experiment(data, K, N=N, seed=0, lambdas=lambdas)
    print(f'\nDone ({time.time() - t_start:.0f}s total).')


if __name__ == '__main__':
    main()
