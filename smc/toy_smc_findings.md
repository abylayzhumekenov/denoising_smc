# Validating the λ-ρ Unified Weight on a 1D Gaussian-Mixture Toy

Self-contained validation of the SMC weighting machinery (`docs/note_2.tex` §2)
against a closed-form ground truth.

Reproduce: `venv/bin/python smc/toy_smc.py`

## Setting

**Prior.** Two-component Gaussian mixture:
$$p_0(x) = w_1\,\mathcal{N}(x;\mu_1,v_1) + w_2\,\mathcal{N}(x;\mu_2,v_2),$$
with $w=(0.6,0.4)$, $\mu=(-1.5,2.0)$, $v=(0.7,1.3)$.

**Noising.** VE/EDM: $x_\sigma = x_0 + \sigma z$, $z\sim\mathcal{N}(0,1)$.  The noised
marginal is the same mixture with widened variances $v_m+\sigma^2$, and the score
$\nabla_x\log p_\sigma(x)$ is exact (no learned network).

**Observation.** $y = x_0 + \varepsilon$, $\varepsilon\sim\mathcal{N}(0,\gamma^2)$,
with $y=0.0$ and $\gamma^2=1.0$ (wide enough to see both prior modes).

**Posterior.** Analytic (mixture × Gaussian = mixture, per-component Kalman update).
For this instance: **mean −0.33, std 1.06**, visibly bimodal.

**Intermediate likelihood.** The exact $p(y\mid x_\sigma)$ is available in closed form
via the mixture-in-$y$ identity (`note_2.tex` eq 27).  Guidance uses $\nabla_x\log p(y\mid x_\sigma)$
exactly — no approximation, so every deviation from the analytic posterior is
algorithm error.

## Samplers

| Sampler | Proposal | Weight | Returns |
|---------|----------|--------|---------|
| `run_smc` | GEM (Euler–Maruyama, guided) | $\lambda$–$\rho$ unified (eq 18) | Particles + log-weights + ESS |
| `run_ode` | Deterministic guided Heun (EDM) | — | Particles only (no weights) |

`run_smc` always uses the left-endpoint Girsanov correction $C_k = -\sqrt{\delta_k}\,b_k z_k - \frac12\delta_k\|b_k\|^2$,
which equals the exact Gaussian kernel ratio for GEM (= TDS when $\lambda=1$).

## λ-sweep experiment

Sweeps $\lambda\in\{0,0.25,0.5,0.75,1\}$ at $\rho=1$, $K=2000$, $N=512$, $8$ seeds.
Metrics: W1, |dmean|, |dstd| (weighted), final ESS (SEM across seeds).

**Output:** one 2×2 PDF figure (`smc/figs/lambda_experiment.pdf`):
- Top row: W1 / |dstd| / ESS vs $\lambda$ (single blue line, error bars = SEM).
- Bottom row: weighted KDE (SMC, $\lambda$ gradient) + ODE rug ticks + analytic PDF.

### Results ($N=512$, $K=2000$, $y=0.0$, $\gamma^2=1.0$)

| λ | W1 | |dmean| | |dstd| | ESS |
|---|---:|---:|---:|---:|
| 0.00 (pBS) | 0.065 | 0.087 | 0.276 | 432 |
| 0.25 | 0.065 | 0.073 | 0.227 | 456 |
| 0.50 | 0.065 | 0.062 | 0.169 | 480 |
| 0.75 | 0.065 | 0.055 | 0.100 | 500 |
| 1.00 (TDS) | 0.065 | 0.047 | 0.020 | 510 |

### Interpretation

**λ controls posterior variance recovery.**  |dstd| drops monotonically from 0.276
(pBS) to 0.020 (TDS) — a 14× improvement.  Fractional λ interpolates cleanly,
confirming the unified-weight formula (eq 18).

**W1 is identical across λ.**  W1 depends only on unweighted particle positions.
Since `run_smc` uses the same GEM proposal with the same seed for every λ, the
particle cloud is identical — only the weights differ.  The residual W1 ≈ 0.065
at $K=2000$ is the finite-step trajectory error, equal for all arms.  More steps or
more particles would reduce it further.

**ESS is healthy for all λ.**  432–510 out of $N=512$, no resampling triggered.
The Girsanov correction cancels the dominant stochastic component of the
likelihood ratio, keeping weight variance small.  Even pBS (λ=0) stays at 84% of N
because the telescoping likelihood-ratio sum collapses to the bounded random
variable $\log p(y\mid x_0)$.

**ODE → point estimate.**  The deterministic guided Heun ODE produces a rug of
nearly identical point estimates, confirming the core limitation of the
DiffusionPDE baseline: zero uncertainty quantification.

## Code structure

```
sweep_lambda(lambdas, K, N, seeds)        → table + data dict
sweep_K(K_values, N, seeds, lam, rho)     → table + data dict  (future use)
plot_lambda_experiment(data, K, N, ...)   → 2×2 PDF
```

Each sweep/plot is independently callable.  `main()` picks the current experiment.

## Next steps

- **K sweep** — check W1 → 0 as K → ∞ (trajectory error vanishes).
- **ρ sweep** — confirm ρ directly controls posterior tempering.
- **Heun-SDE** — higher-order proposal with `run_smc` adapted to use `heunsde`.
- **PDE port** — Burgers/Darcy with learned score + surrogate likelihood.
