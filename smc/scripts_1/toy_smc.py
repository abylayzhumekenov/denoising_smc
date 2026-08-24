"""Closed-form 1D Gaussian-mixture validation of the Girsanov-corrected SMC weight.

Faithful to `docs/note_1.pdf` Sec. 3 (toy model) and Sec. 2.4-2.5 (discretisation and
sequential weights).  Every ingredient -- prior, noised marginal, score, exact
intermediate likelihood, guidance drift, Hessian, and analytic posterior -- is available
in closed form, so any deviation from the analytic posterior is algorithm error.

The three weightings of Sec. 3.2 (eq. 30-32), each used as the whole per-step increment
`G_k` of eq. (19), are compared on two twists:

  * Experiment A (exact twist, gamma_tilde = gamma): Sec. 3.3.  The exact intermediate
    likelihood solves the backward Kolmogorov equation, so the potential V_tau = 0 (eq. 11)
    and the corrected increments vanish: G^Girs_k = Delta f_k + C_k = 0, G^pot_k = V_k = 0
    (eq. 33); the corrected weight reduces to the boundary factor f0(xi_0).  The
    pseudo-bootstrap increment does not vanish; by the telescoping (14) its path weight is
    log W^PBS = fT(xi_T) (eq. 34), retaining the full variance of Delta f_k.

  * Experiment 2 (surrogate twist, gamma_tilde != gamma): Sec. 3.4.  Guidance drift is
    suboptimal but the terminal correction (23) is still evaluated with the exact
    likelihood, so the target remains the posterior of (29).  The three weightings
    genuinely differ: PBS is biased, the corrected two target the posterior up to
    discretisation error.

  * Experiment 3 (terminally-consistent surrogate): the twist's observation variance is
    annealed gamma_tilde(sigma) -> gamma as sigma -> 0 (Sec. 2.5 remedy: a terminally
    consistent surrogate), so fT(xi_T) = log p(y|xi_T), the terminal correction (23)
    vanishes, and the last-step ESS collapse is removed.

  * Experiment 4 (K sweep, terminally-consistent surrogate): step count K in
    {25, 50, 100, 200, 500} at gamma_tilde = 0.5 annealed to gamma, run in the coarse regime
    where the discretisation error is visible: the corrected W1 is elevated at K = 25-50 and
    drops to the ESS/Monte-Carlo floor as K grows (eq. 33), while PBS keeps a structural bias
    (eq. 34).  Experiments 1-3 use K = 500; all experiments use N = 2048.

Working convention (reverse-time, per Sec. 2.1/3.1): sigma decreases sigma_max -> sigma_min
as tau goes 0 -> T, so the accumulated diffusion matrix of (12) is the positive scalar
Sigma_k = s^2_{tau_{k-1}} - s^2_{tau_k} (eq. 27).  The chain rule d/dtau = -d/dsigma fixes
the sign of the potential term.

Proposal: guided Euler-Maruyama (Appendix B), the discretisation for which the left-endpoint
correction C_hat_k of (18) equals the exact kernel ratio at every step size.

Metrics (Sec. 3.5): Wasserstein-1 (eq. 35), weighted mean/std error, ESS, resampling count.

Outputs: figures in `figures/` and LaTeX tables in `tables/`, systematically named
`toy_1..toy_4` (figures and tables share the base name; labels `fig:toy_N`, `tab:toy_N`).

Run: .venv/bin/python smc/scripts_1/toy_smc.py
"""

import os
import resource
import time

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
GAMMA2 = 1.0                  # exact observation variance gamma^2 (eq. 26)
GAMMA_TILDE2 = 0.25           # misspecified observation variance gamma~^2 (Exp 2, Sec. 3.4)
K_SWEEP = [25, 50, 100, 200, 500]                 # Exp 4: step count (coarse regime)


def make_annealed_obs_var(gt):
    """Terminally-consistent surrogate twist: misspecified scale `gt` at sigma_max, healed to the
    exact gamma^2 as sigma -> 0 (Sec. 2.5 remedy).  This is the practically-relevant surrogate: a
    plug-in likelihood N(y; D_theta(x_sigma, sigma), r^2) with D_theta(x_0, 0) = x_0 is
    automatically terminally consistent."""
    return lambda sigma: GAMMA2 + (gt ** 2 - GAMMA2) * (sigma / SIGMA_MAX)


def annealed_obs_var(sigma):
    """Terminally-consistent surrogate for Exp 3 (gamma_tilde = 0.5)."""
    return make_annealed_obs_var(GAMMA_TILDE2)(sigma)

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
def posterior_params():
    S = VAR + GAMMA2
    q = C * np.exp(-0.5 * (OBS_Y - MU) ** 2 / S) / np.sqrt(2 * np.pi * S)
    abar = q / q.sum()
    var_p = 1 / (1 / VAR + 1 / GAMMA2)
    mu_p = var_p * (MU / VAR + OBS_Y / GAMMA2)
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


def _twist_obs(obs_var, sigma):
    """Observation variance of the twist at noise level sigma (scalar or callable)."""
    return obs_var(sigma) if callable(obs_var) else obs_var


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
def run_smc(N, K, seed, obs_var, weighting, resample_threshold=0.5):
    """Guided EM proposal + one of the three weightings.  Returns result dict.

    Pipeline (eq. 19-23): initial f0(xi_0); per-step G_k; terminal correction
    log p(y|xi_T) - fT(xi_T) (exact gamma always).  Systematic resampling at ESS < N/2.
    """
    sig = build_sigma(K)
    rng = np.random.default_rng(seed)
    x = init_particles(N, rng)

    log_w = loglik(x, SIGMA_MAX, _twist_obs(obs_var, SIGMA_MAX))  # initial weight f0(xi_0) (eq. 21)
    inc_hist = {w: [] for w in ('pbs', 'girs', 'pot')}
    bridge = []
    n_resample = 0
    ess_history = []

    for step in range(K):
        sk, skm1 = sig[step], sig[step + 1]
        delta = sk ** 2 - skm1 ** 2                # Sigma_k = s^2_{tau_{k-1}} - s^2_{tau_k} (eq. 27)
        st = lik_stats(x, sk, _twist_obs(obs_var, sk))     # f, b, H, score at (xi_{k-1}, sigma_{k-1})

        eps = rng.standard_normal(N)
        z = np.sqrt(delta) * eps                    # z_k ~ N(0, Sigma_k) (eq. 12)
        x_next = x + delta * (st['score'] + st['grad']) + z    # EM proposal (App. B)

        f_km1 = loglik(x_next, skm1, _twist_obs(obs_var, skm1))  # f(xi_k; sigma_k)
        f_k = st['log_lik']                         # f(xi_{k-1}; sigma_{k-1})
        dll = f_km1 - f_k                           # Delta f_k (eq. 13)

        # Girsanov left-endpoint correction (eq. 18)
        C = -st['grad'] * z - 0.5 * st['grad'] ** 2 * delta
        # Potential increment (eq. 17, 11): frozen-state telescoping of d_tau f + left-endpoint
        # quadrature of the remaining integrand 2*sigma[score*b + 1/2 b^2 + 1/2 H].
        f_frozen = loglik(x, skm1, _twist_obs(obs_var, skm1))
        Vpot = (f_frozen - f_k) + (sk - skm1) * 2.0 * sk * (
            st['score'] * st['grad'] + 0.5 * st['grad'] ** 2 + 0.5 * st['hess'])

        inc_hist['pbs'].append(np.mean(np.abs(dll)))
        inc_hist['girs'].append(np.mean(np.abs(dll + C)))
        inc_hist['pot'].append(np.mean(np.abs(Vpot)))
        bridge.append(np.mean(np.abs((dll + C) - Vpot)))

        G = {'pbs': dll, 'girs': dll + C, 'pot': Vpot}[weighting]
        log_w += G
        log_w -= logsumexp(log_w)
        ess_history.append(float(1.0 / np.sum(np.exp(2 * log_w))))

        if float(1.0 / np.sum(np.exp(2 * log_w))) < resample_threshold * N:
            idx = systematic_resample(log_w, rng)
            x_next = x_next[idx]
            log_w = np.zeros(N)
            n_resample += 1

        x = x_next

    pre_terminal = log_w.copy()
    # terminal correction (eq. 23), always with the exact likelihood
    terminal_corr = loglik(x, 0.0, GAMMA2) - loglik(x, 0.0, _twist_obs(obs_var, 0.0))
    log_w += terminal_corr
    log_w -= logsumexp(log_w)
    ess_final = float(1.0 / np.sum(np.exp(2 * log_w)))
    ess_history.append(ess_final)   # includes the post-terminal-correction collapse (eq. 23)

    return {'particles': x, 'weights': log_w, 'pre_terminal': pre_terminal,
            'ess': ess_final, 'terminal_corr': terminal_corr,
            'ess_history': np.array(ess_history), 'resamples': n_resample,
            'inc_hist': {w: np.array(v) for w, v in inc_hist.items()},
            'bridge': np.array(bridge)}


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
    for obs_var in (GAMMA2, GAMMA_TILDE2):
        for x in xs:
            for sig in sigs:
                st = lik_stats(x, sig, obs_var)
                err_b = max(err_b, float(np.abs(st['grad'] -
                                       (loglik(x + h, sig, obs_var) - loglik(x - h, sig, obs_var)) / (2 * h))[0]))
                err_H = max(err_H, float(np.abs(st['hess'] -
                                       (lik_stats(x + h, sig, obs_var)['grad'] -
                                        lik_stats(x - h, sig, obs_var)['grad']) / (2 * h))[0]))
                err_score = max(err_score, float(np.abs(st['score'] -
                                               (noised_logpdf(x + h, sig) - noised_logpdf(x - h, sig)) / (2 * h))[0]))
    return float(err_b), float(err_H), float(err_score)


# ----------------------------------------------------------------- experiments
WEIGHTS = ('pbs', 'girs', 'pot')
WEIGHT_LABEL = {'pbs': 'PBS', 'girs': 'Girs', 'pot': 'Pot'}
COLOR = {'pbs': '#d62728', 'girs': '#1f77b4', 'pot': '#2ca02c'}


def _run_metrics(N, K, seed, obs_var):
    """Run the three weightings at one setting; return results dict (no printing)."""
    pm, ps = posterior_params()[-2:]
    results = {}
    for w in WEIGHTS:
        r = run_smc(N, K, seed, obs_var, w)
        wm, ws = weighted_stats(r['particles'], r['weights'])
        w1 = wasserstein1(r['particles'], r['weights'])
        dm = abs(wm - pm)
        ds = abs(ws - ps)
        results[w] = dict(w1=w1, dmean=dm, dstd=ds, ess=r['ess'],
                          particles=r['particles'], weights=r['weights'],
                          inc_hist=r['inc_hist'], bridge=r['bridge'],
                          ess_history=r['ess_history'],
                          pre_terminal=r['pre_terminal'], resamples=r['resamples'])
    return results


def _report(title, obs_var, N, K, seed):
    pm, ps = posterior_params()[-2:]
    twist = (obs_var if not callable(obs_var)
             else f'annealed({GAMMA_TILDE2}->{GAMMA2})')
    print(f'analytic posterior: mean={pm:.4f}  std={ps:.4f}')
    print(f'{title}  (N={N}, K={K}, obs_var_twist={twist})')
    print(f'{"weight":>6}  {"W1":>12}  {"|dmean|":>11}  {"|dstd|":>12}  {"ESS":>8}  {"resamp":>7}')
    print('-' * 66)
    results = _run_metrics(N, K, seed, obs_var)
    for w in WEIGHTS:
        r = results[w]
        print(f'{WEIGHT_LABEL[w]:>6}  {r["w1"]:>12.4f}  {r["dmean"]:>11.4f}  {r["dstd"]:>12.4f}  '
              f'{r["ess"]:>8.0f}  {r["resamples"]:>7}')
    return results


def _exp1_diagnostics(N, K, seed, obs_var):
    r_girs = run_smc(N, K, seed, obs_var, 'girs')
    r_pbs = run_smc(N, K, seed, obs_var, 'pbs')
    fT = loglik(r_pbs['particles'], 0.0, obs_var)   # fT(xi_T) = log p~(y|xi_T)
    diff = r_pbs['pre_terminal'] - fT
    eq34 = np.abs(diff - diff.mean()).mean()  # log W^PBS = f0 + sum Delta f_k = fT (eq. 34)
    bridge = r_girs['bridge'].mean()
    mg, mp, mb = (r_girs['inc_hist'][w].mean() for w in ('girs', 'pot', 'pbs'))
    print('\nExp 1 diagnostics (exact twist):')
    print(f'  spread of [pre-terminal PBS weight - fT(xi_T)] = {eq34:.3e}   (eq. 34, telescoping)')
    print(f'  mean per-step |G_girs - G_pot|          = {bridge:.3e}   (bridge id. eq. 16)')
    print(f'  mean |G_girs| over steps                = {mg:.3e}   (eq. 33)')
    print(f'  mean |G_pot|  over steps                = {mp:.3e}   (eq. 33)')
    print(f'  mean |G_pbs|  over steps                = {mb:.3e}   (eq. 34)')
    write_latex_table(
        os.path.join(TABLES_DIR, 'toy_1_diag.tex'),
        'Exp 1 diagnostics: the corrected increments vanish (eq.~33), the pseudo-bootstrap '
        'weight telescopes to $f_T$ (eq.~34).',
        'tab:toy_1_diag',
        ['quantity', 'value', 'expected'],
        [['PBS telescoping spread $|\\sum\\Delta f_k - f_T|$', _fe(eq34), r'$\approx 0$'],
         ['mean per-step $|G_{\\mathrm{Girs}} - G_{\\mathrm{pot}}|$ (eq.~16)', _fe(bridge), r'$\approx 0$'],
         ['mean $|G_{\\mathrm{Girs}}|$ (eq.~33)', _fe(mg), r'$\approx 0$'],
         ['mean $|G_{\\mathrm{pot}}|$ (eq.~33)', _fe(mp), r'$\approx 0$'],
         ['mean $|G_{\\mathrm{PBS}}|$ (eq.~34)', _fe(mb), '>$0$']])
    return eq34


def _exp3_diagnostics(N, K, seed):
    rb = run_smc(N, K, seed, GAMMA_TILDE2, 'girs')     # Exp 2 (terminally inconsistent)
    rc = run_smc(N, K, seed, annealed_obs_var, 'girs')  # Exp 3 (terminally consistent)
    tb = np.abs(rb['terminal_corr']).mean()
    tc = np.abs(rc['terminal_corr']).mean()
    print('\nExp 3 diagnostics (terminal consistency):')
    print(f'  mean|terminal correction (eq. 23)| : Exp 2 (const gamma~)={tb:.3e}   '
          f'Exp 3 (annealed)={tc:.3e}')
    print(f'  final ESS (girs):                    Exp 2={rb["ess"]:.0f}   Exp 3={rc["ess"]:.0f}')
    write_latex_table(
        os.path.join(TABLES_DIR, 'toy_3_diag.tex'),
        'Exp 3 diagnostics: a terminally-consistent surrogate (annealed $\\tilde\\gamma\\to\\gamma$) '
        'zeroes the terminal correction (eq.~23) and removes the last-step ESS collapse.',
        'tab:toy_3_diag',
        ['quantity', 'Exp 2 (const $\\tilde\\gamma$)', 'Exp 3 (annealed)'],
        [['mean $|$terminal correction (eq.~23)$|$', _fe(tb), _fe(tc)],
         ['final ESS (Girs)', _f0(rb['ess']), _f0(rc['ess'])]])
    return tc


# ----------------------------------------------------------------- outputs: figures + tables
FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables')
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

MARKER = {'pbs': 'o', 'girs': 's', 'pot': '^'}


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


def write_metrics_table(results, path, caption, label):
    """Standard 3-row metrics table (PBS/Girs/Pot x W1, |dmean|, |dstd|, ESS, resamp)."""
    pm, ps = posterior_params()[-2:]
    headers = ['weight', 'W1', '|dmean|', '|dstd|', 'ESS', 'resamp']
    rows = [[WEIGHT_LABEL[w], _f4(results[w]['w1']), _f4(results[w]['dmean']),
             _f4(results[w]['dstd']), _f0(results[w]['ess']),
             _f0(results[w]['resamples'])] for w in WEIGHTS]
    write_latex_table(path, caption + f' (analytic posterior: $\\mu={pm:.3f}$, $\\sigma={ps:.3f}$).',
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


def plot_experiment(results, N, K, title, filename):
    """Shared 1x3 layout used identically by every experiment so they are directly comparable.

    Panels: weighted density vs analytic posterior (shape), per-step increment magnitude
    (mechanism, eq. 33/34), ESS over steps (terminal-correction collapse, eq. 23).  W1 etc. are
    reported in the accompanying LaTeX table.
    """
    x = results['girs']['particles']
    lo, hi = float(x.min()) - 0.3, float(x.max()) + 0.3
    xgrid = np.linspace(lo, hi, 800)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _density_panel(axes[0], results, xgrid, posterior_pdf(xgrid))
    axes[0].set_title('Density (weighted KDE)')

    steps = np.arange(1, K + 1)
    ax = axes[1]
    for w in WEIGHTS:
        ax.plot(steps, results[w]['inc_hist'][w], color=COLOR[w], linewidth=1.0,
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


def plot_exp1(results, N, K):
    plot_experiment(results, N, K, 'Experiment 1: exact twist', 'toy_1.pdf')


def plot_exp2(results, N, K):
    plot_experiment(results, N, K, r'Experiment 2: surrogate twist $\tilde\gamma=0.5$',
                    'toy_2.pdf')


def plot_exp3(results, N, K):
    plot_experiment(results, N, K,
                    r'Experiment 3: terminally-consistent surrogate ($\tilde\gamma\to\gamma$)',
                    'toy_3.pdf')


def plot_exp4(data, N):
    """Single-panel W1-vs-K (log-x): corrected elevated at coarse K (discretisation error),
    dropping to the MC floor as K grows; PBS retains a structural bias."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w in WEIGHTS:
        vals = [data[K][w]['w1'] for K in K_SWEEP]
        ax.plot(K_SWEEP, vals, color=COLOR[w], marker=MARKER[w], markersize=5,
                linewidth=1.3, label=WEIGHT_LABEL[w])
    ax.set_xscale('log'); ax.set_xlabel('K'); ax.set_ylabel('W1')
    ax.legend(fontsize=8)
    fig.suptitle(f'Experiment 4: step-count sweep  (N={N}, '
                 f'$\\tilde\\gamma=0.5$ annealed to $\\gamma$)')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(FIGS_DIR, 'toy_4.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def run_exp4(N, seed):
    """Exp 4: K sweep (discretisation convergence) with a terminally-consistent surrogate."""
    ov = make_annealed_obs_var(GAMMA_TILDE2)
    print(f'\n=== Experiment 4: K sweep  (N={N}, terminally-consistent surrogate) ===')
    print(f'{"K":>7}  {"W1(PBS)":>10}  {"W1(Girs)":>10}  {"W1(Pot)":>10}  {"ESS(Girs)":>9}')
    print('-' * 50)
    data, rows = {}, []
    for K in K_SWEEP:
        res = _run_metrics(N, K, seed, ov)
        data[K] = res
        rows.append([f'{K}', _f4(res['pbs']['w1']), _f4(res['girs']['w1']),
                     _f4(res['pot']['w1']), _f0(res['girs']['ess'])])
        print(f'{K:>7}  {res["pbs"]["w1"]:>10.4f}  {res["girs"]["w1"]:>10.4f}  '
              f'{res["pot"]["w1"]:>10.4f}  {res["girs"]["ess"]:>9.0f}')
    write_latex_table(os.path.join(TABLES_DIR, 'toy_4.tex'),
                      f'Exp 4: terminally-consistent surrogate '
                      f'($\\tilde\\gamma(\\sigma)\\to\\gamma$; $N={N}$), coarse regime. '
                      f'Corrected W1 is elevated at coarse $K$ (discretisation error) and '
                      f'drops to the ESS/Monte-Carlo floor as $K$ grows (eq.~33); PBS retains '
                      f'a structural bias (eq.~34) independent of $K$.',
                      'tab:toy_4',
                      ['K', 'W1 (PBS)', 'W1 (Girs)', 'W1 (Pot)', 'ESS (Girs)'], rows)
    plot_exp4(data, N)


# ----------------------------------------------------------------- main
def main():
    N, K, seed = 2048, 500, 0
    t0 = time.time()
    _setup_style()

    eb, eH, es = selfcheck()
    print(f'closed-form self-check  max|d(b) - FD(loglik)|={eb:.2e}  '
          f'max|d(H) - FD(b)|={eH:.2e}  max|score - FD(logpdf)|={es:.2e}')

    print('\n=== Experiment 1: exact twist (gamma~ = gamma) ===')
    res_1 = _report('Exp 1', GAMMA2, N, K, seed)
    write_metrics_table(res_1, os.path.join(TABLES_DIR, 'toy_1.tex'),
                        'Exp 1: exact twist ($\\tilde\\gamma = \\gamma$); W1 (eq.~35) and '
                        'weighted posterior mean/std error vs the analytic posterior.',
                        'tab:toy_1')
    _exp1_diagnostics(N, K, seed, GAMMA2)
    plot_exp1(res_1, N, K)

    print(f'\n=== Experiment 2: surrogate twist (gamma~^2 = {GAMMA_TILDE2}) ===')
    res_2 = _report('Exp 2', GAMMA_TILDE2, N, K, seed)
    write_metrics_table(res_2, os.path.join(TABLES_DIR, 'toy_2.tex'),
                        'Exp 2: misspecified surrogate twist ($\\tilde\\gamma = 0.5$); '
                        'target remains the posterior of (29).',
                        'tab:toy_2')
    plot_exp2(res_2, N, K)

    print('\n=== Experiment 3: terminally-consistent surrogate (annealed) ===')
    res_3 = _report('Exp 3', annealed_obs_var, N, K, seed)
    write_metrics_table(res_3, os.path.join(TABLES_DIR, 'toy_3.tex'),
                        'Exp 3: terminally-consistent surrogate '
                        '($\\tilde\\gamma(\\sigma)\\to\\gamma$ as $\\sigma\\to 0$); '
                        'terminal correction (23) vanishes.',
                        'tab:toy_3')
    _exp3_diagnostics(N, K, seed)
    plot_exp3(res_3, N, K)

    run_exp4(N=N, seed=seed)

    print(f'\nDone ({time.time() - t0:.0f}s).')


if __name__ == '__main__':
    main()
