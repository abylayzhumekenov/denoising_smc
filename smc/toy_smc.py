"""Validation toy: 1D Gaussian-mixture SMC with exact intermediate likelihood.

Implements the λ-ρ unified weight (note_2.tex eq 18):
  log w_{k-1} = log w_k + ρ·[log p̃(y|x_{k-1}) - log p̃(y|x_k)] + λ·C_k

Sweeps (each callable independently):
  sweep_lambda  — W1 / |dstd| / ESS vs λ
  sweep_K       — W1 / |dstd| / ESS vs K

Plots (2×2: W1, |dstd|, ESS panels + weighted KDE density):
  plot_lambda_experiment  — plots λ-sweep
  plot_K_experiment       — plots K-sweep

Internals: _sweep (shared loop), _plot_experiment (shared layout)

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


def wasserstein1(samples, log_w=None, lo=-6, hi=6, grid=8000):
    g = np.linspace(lo, hi, grid)
    Fp = posterior_cdf(g)
    x = samples[:, 0]
    order = np.argsort(x)
    if log_w is not None:
        w = np.exp(log_w - np.max(log_w))
        w = w / w.sum()
        Fn = np.interp(g, x[order], np.cumsum(w[order]), left=0.0, right=1.0)
    else:
        Fn = np.searchsorted(x[order], g) / len(samples)
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


def run_smc(N, K, seed, lam=1.0, rho=1.0, resample_threshold=0.5):
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

        if ess < resample_threshold * N:
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


def _sweep(runs, N, pm, ps_ref, xlabel, key_fmt, resample_threshold=0.5):
    header = (f'{xlabel:>6}  {"W1":>12}  {"|dmean|":>11}  {"|dstd|":>12}  '
              f'{"ESS":>8}  {"resamp":>7}')
    print(header)
    print('-' * len(header))

    data = {}
    for key, kw in runs:
        r = run_smc(N, kw['K'], 0, lam=kw.get('lam', 1.0),
                     rho=kw.get('rho', 1.0), resample_threshold=resample_threshold)
        wm, ws = weighted_stats(r['particles'], r['weights'])
        w1 = wasserstein1(r['particles'], r['weights'])
        dm = abs(wm - pm)
        ds = abs(ws - ps_ref)
        data[key] = {'w1': w1, 'dmean': dm, 'dstd': ds, 'ess': r['ess'],
                     'particles': r['particles'], 'weights': r['weights']}
        print(f'{key_fmt(key):>6}  {w1:>12.4f}  {dm:>11.4f}  '
              f'{ds:>12.4f}  {r["ess"]:>8.0f}  {r["resamples"]:>7}')
    return data


def sweep_lambda(lambdas, K=2000, N=128, rho=1.0, resample_threshold=0.5):
    pm, ps_ref = posterior_params()[-2:]
    print(f'analytic posterior: mean={pm:.4f}  std={ps_ref:.4f}')
    print(f'λ-sweep  K={K}  N={N}\n')
    runs = [(lam, dict(K=K, lam=lam, rho=rho)) for lam in lambdas]
    return _sweep(runs, N, pm, ps_ref, 'λ', lambda x: f'{x:.2f}', resample_threshold)


def sweep_K(K_values, N=128, lam=1.0, rho=1.0, resample_threshold=0.5):
    pm, ps_ref = posterior_params()[-2:]
    print(f'analytic posterior: mean={pm:.4f}  std={ps_ref:.4f}')
    print(f'K-sweep  N={N}  λ={lam}  ρ={rho}\n')
    runs = [(K, dict(K=K, lam=lam, rho=rho)) for K in K_values]
    return _sweep(runs, N, pm, ps_ref, 'K', str, resample_threshold)


# ----------------------------------------------------------------- plots
def _plot_experiment(data, xs, N, xlabel, ode_K, title, filename, label_fn,
                     data2=None, label2_fn=None):
    colors_smc = plt.cm.Blues(np.linspace(0.4, 0.9, len(xs)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    def _panel(ax, metric, ylabel):
        ys = [data[x][metric] for x in xs]
        ax.plot(xs, ys, color='#1f77b4', marker='o', markersize=6, linewidth=1.2,
                label='λ=1' if data2 is not None else None)
        if data2 is not None:
            ys2 = [data2[x][metric] for x in xs]
            ax.plot(xs, ys2, color='#d62728', marker='s', markersize=6, linewidth=1.2,
                    label='λ=0')
            ax.legend(fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    _panel(axes[0, 0], 'w1', 'W1')
    _panel(axes[0, 1], 'dstd', '|dstd|')
    _panel(axes[1, 0], 'ess', 'ESS')

    ax = axes[1, 1]
    if data2 is not None:
        k_max = xs[-1]
        arm_data = [
            ('ODE', run_ode(N, ode_K, 0), None),
            (label_fn(k_max), data[k_max]['particles'], data[k_max]['weights']),
            (label2_fn(k_max), data2[k_max]['particles'], data2[k_max]['weights']),
        ]
    else:
        arm_data = [('ODE', run_ode(N, ode_K, 0), None)]
        for _, x in enumerate(xs):
            arm_data.append((label_fn(x), data[x]['particles'], data[x]['weights']))

    all_x = np.concatenate([x[:, 0] for _, x, _ in arm_data])
    lo, hi = float(all_x.min()) - 0.3, float(all_x.max()) + 0.3
    xgrid = np.linspace(lo, hi, 800)

    ax.plot(xgrid, posterior_pdf(xgrid), 'k--', linewidth=2.0,
            label='Analytic posterior', zorder=10)
    ax.plot([], [], '|', color='gray', markersize=8, markeredgewidth=1.0,
            label='ODE')

    for idx, (label, x, lw) in enumerate(arm_data):
        xf = x[:, 0]
        if label == 'ODE':
            ax.plot(xf, np.full_like(xf, -0.02), '|', color='gray',
                    markersize=8, markeredgewidth=1.0, alpha=0.4)
        else:
            w = np.exp(lw - np.max(lw)); w = w / w.sum()
            kde = gaussian_kde(xf, weights=w, bw_method=0.3)
            if data2 is not None:
                color = '#1f77b4' if idx == 1 else '#d62728'
            else:
                color = colors_smc[idx - 1]
            ax.plot(xgrid, kde(xgrid), color=color, alpha=0.8,
                    linewidth=1.2, label=label)

    ax.set_xlabel('x'); ax.set_ylabel('Density')
    ax.legend(fontsize=7, loc='upper left')

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIGS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_lambda_experiment(data, K, N, lambdas, rho=1.0):
    xs = sorted(data.keys())
    _plot_experiment(data, xs, N, 'λ', K,
                     f'λ experiment  (K={K}, N={N})',
                     'lambda_experiment.pdf',
                     lambda x: f'λ={x:.2f}')


def plot_K_experiment(data, N, lam=1.0, rho=1.0):
    xs = sorted(data.keys())
    _plot_experiment(data, xs, N, 'K', xs[-1],
                     f'K experiment  (N={N}, λ={lam}, ρ={rho})',
                     'K_experiment.pdf',
                     lambda x: f'SMC (K={x})')


def sweep_K_lambda_comparison(K_values, N=1024):
    print('K-sweep λ-comparison')
    print('─' * 50)
    print('λ = 0 (pseudo-bootstrap):')
    data_0 = sweep_K(K_values, N=N, lam=0.0)
    print()
    print('λ = 1 (TDS):')
    data_1 = sweep_K(K_values, N=N, lam=1.0)
    return data_0, data_1


def plot_K_lambda_comparison(data_0, data_1, N=1024):
    xs = sorted(data_0.keys())
    _plot_experiment(data_1, xs, N, 'K', xs[-1],
                     f'K-sweep: λ=1 (blue) vs λ=0 (red)  (N={N})',
                     'K_lambda_comparison.pdf',
                     lambda x: f'λ=1 (K={x})',
                     data2=data_0,
                     label2_fn=lambda x: f'λ=0 (K={x})')


# ----------------------------------------------------------------- main
def main():
    N = 1024
    t_start = time.time()

    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    data_l = sweep_lambda(lambdas, K=5000, N=N)
    print(f'  ({time.time() - t_start:.0f}s)\n')
    plot_lambda_experiment(data_l, K=5000, N=N, lambdas=lambdas)

    K_values = [100, 200, 500, 1000, 2000, 5000]
    data_k = sweep_K(K_values, N=N)
    print(f'  ({time.time() - t_start:.0f}s)\n')
    plot_K_experiment(data_k, N=N)

    data_k0, data_k1 = sweep_K_lambda_comparison(K_values, N=N)
    print(f'  ({time.time() - t_start:.0f}s)\n')
    plot_K_lambda_comparison(data_k0, data_k1, N=N)

    print(f'\nDone ({time.time() - t_start:.0f}s total).')


if __name__ == '__main__':
    main()
