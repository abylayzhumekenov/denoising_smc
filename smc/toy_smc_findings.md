# Validating Girsanov-Corrected SMC for Diffusion-Guided Inference on a Tractable Toy

A self-contained note summarizing the validation of the λ-ρ Sequential Monte Carlo (SMC)
weighting machinery (`docs/idea.md`) on a one-dimensional Gaussian-mixture model with known
ground truth. Implementation and reproduction: `smc/toy_smc.py` (`venv/bin/python smc/toy_smc.py`).

## 1. Motivation

Before applying the SMC extension to the (expensive, network-based) PDE pipeline, we validate
the weighting machinery on a model whose every ingredient is available in closed form:

- the **score** of the noised distribution is exact (a "perfect model"),
- the **posterior** `p(x₀|y)` is known analytically.

The toy is therefore a *perfect model*: any deviation of the SMC output from the analytic
posterior is algorithm error, not model error.

## 2. Setting

**Prior.** A one-dimensional Gaussian mixture

$$p_0(x) = \sum_{m=1}^{M} w_m \, \mathcal{N}(x; \mu_m, \sigma_m^2),$$

with $M=2$, $w=(0.6, 0.4)$, $\mu=(-1.5, 2.0)$, $\sigma^2=(0.7, 1.3)$.

**Noising.** VE/EDM forward process $x_t = x_0 + \sigma_t z$, $z \sim \mathcal{N}(0,1)$. Because
Gaussians are closed under convolution, the noised marginal is the same mixture with widened
variances, and the score

$$\nabla_x \log p_t(x) = \sum_m \pi_m(x)\, \frac{\mu_m - x}{\sigma_m^2 + \sigma_t^2}$$

is a cheap weighted average (a "perfect score").

**Observation.** $y = x + n$, $n \sim \mathcal{N}(0, 0.5)$, with $y = 0.8$.

**Posterior.** Mixture × Gaussian likelihood = another mixture (per-component Kalman update).
For this instance the analytic posterior has **mean 0.833, std 0.802**, dominated by the
component near the observation.

## 3. Guided Reverse Dynamics and the λ-ρ Weight

The reverse process runs from $\sigma_{\max}$ down to $\sigma_{\min}$ in $K$ steps with increments
$\delta_k = \sigma_k^2 - \sigma_{k-1}^2$. The guided proposal adds the likelihood gradient
$b(x) = \nabla_x \log \tilde{p}(y|x)$ to the score drift.

The unified incremental weight is

$$\log w_{k-1}^{\text{inc}} = \rho \cdot \bigl[\log\tilde{p}(y\mid x_{k-1}) - \log\tilde{p}(y\mid x_k)\bigr] \;+\; \lambda \cdot C_k,$$

with the Girsanov correction

$$C_k = -\sqrt{\delta_k}\, b_k^\top z \;-\; \frac{1}{2}\,\delta_k\, \|b_k\|^2,$$

where $z$ is the (already sampled) Brownian increment. Two operating points:

- **λ = 1 (Girsanov/TDS):** exact target $p(x_0\mid y)$ up to discretization. For a first-order
  Euler (GEM) step the transition is Gaussian, so $C_k$ coincides with the closed-form density
  ratio — the Twisted Diffusion Sampler (TDS) weight. We verify this identity numerically to
  $\sim 3\times10^{-13}$ at every step.
- **λ = 0 (pseudo-bootstrap, pBS):** drops the correction, targeting a tempered/inconsistent
  distribution rather than the true posterior.

## 4. Proposals and Variants

| Arm | Proposal | Weight | Role |
|-----|----------|--------|------|
| ODE | deterministic guided flow (Heun) | — | repo baseline; no SMC |
| SOSaG+pBS | SOSaG (Millard et al.) | λ=0 | literature baseline |
| GEM+pBS | GEM | λ=0 | weight-controlled pBS |
| HeunSDE+pBS | Heun-SDE | λ=0 | weight-controlled pBS |
| GEM+Girs | GEM | λ=1 | **= TDS** |
| HeunSDE+Girs | Heun-SDE | λ=1 | primary result |

The GEM+pBS / HeunSDE+pBS arms isolate the *weight* effect from the *proposal* effect.

## 5. Experiments

Schedule: $\sigma_{\max}=6$, $\sigma_{\min}=10^{-3}$, $K=500$, $\rho=7$. Exact initialization from
the noised marginal $p_{\sigma_{\max}}$ (not Gaussian noise). Default $N=128$ particles; metrics
averaged over 8 seeds and reported as mean ± SEM:

- **W1** — Wasserstein distance between the particle cloud and the analytic posterior
  (closed form in 1D),
- **|dmean|, |dstd|** — absolute error of the weighted posterior mean/std vs the analytic values,
- **ESS** — effective sample size (final) and resampling count.

## 6. Results

### 6.1 Main comparison (N=128, K=500, 8 seeds)

| variant | W1 | \|dmean\| | \|dstd\| | ESS | resamp |
|---|---|---|---|---|---|
| ODE | 0.644 ± 0.000 | 0.058 | 0.802 | — | — |
| SOSaG+pBS | 0.268 ± 0.057 | 0.149 | 0.122 | 91 | 86 |
| GEM+pBS | 0.207 ± 0.011 | 0.030 | 0.354 | 90 | 6 |
| HeunSDE+pBS | 0.215 ± 0.015 | 0.045 | 0.357 | 98 | 5 |
| GEM+Girs | 0.191 ± 0.011 | 0.129 | 0.169 | 110 | 37 |
| **HeunSDE+Girs** | **0.176 ± 0.011** | 0.094 | 0.144 | 100 | 31 |

**Interpretation.**

- **W1 ranks the arms exactly as the theory predicts**: HeunSDE+Girs < GEM+Girs < pBS arms < ODE.
  Both λ=1 arms are closest to the true posterior in full distribution.
- **The pBS inconsistency is the weight, not the proposal.** The proposal-controlled arms
  (GEM+pBS, HeunSDE+pBS) behave like SOSaG+pBS — worse than either λ=1 arm. Dropping the
  Girsanov correction (λ=0) while proposing from the guided dynamics biases the target, as
  `docs/idea.md` §1 states.
- **ODE collapses to a point estimate** (std = 0): the deterministic guided flow yields a single
  field with no uncertainty — the core limitation of the current repo pipeline that the SMC
  extension addresses.
- `|dmean|` is too noisy here (final-log-weight estimator after resampling) to discriminate arms;
  W1 and `|dstd|` are the reliable metrics.

### 6.2 N-sweep (λ=1 arms, K=500, 8 seeds)

| N | GEM W1 | GEM \|dstd\| | GEM ESS | Heun W1 | Heun \|dstd\| | Heun ESS |
|---|---|---|---|---|---|---|
| 128 | 0.191 ± 0.011 | 0.169 ± 0.022 | 110 | 0.176 ± 0.011 | 0.144 ± 0.016 | 100 |
| 512 | 0.178 ± 0.008 | 0.105 ± 0.022 | 400 | 0.225 ± 0.042 | 0.112 ± 0.016 | 432 |
| 2048 | 0.147 ± 0.013 | 0.102 ± 0.016 | 1626 | 0.151 ± 0.018 | 0.112 ± 0.011 | 1693 |

- **Underdispersion is not purely finite-sample**: `|dstd|` shrinks with N but plateaus around
  0.10–0.11 at N=2048. The K-sweep below rules out discretization as the cause.
- **ESS scales ~linearly with N** (110 → 400 → 1626) — the filter is healthy.

### 6.3 K-sweep (λ=1 arms + ODE, N=512, 8 seeds)

Sweeps the number of reverse steps K at fixed σ_max/σ_min (only step size changes). `w|dstd|` =
weighted std error (final log-weights); `u|dstd|` = unweighted std error.

| K | arm | W1 | \|dmean\| | w\|dstd\| | u\|dstd\| | ESS | resamp |
|---|---|---|---|---|---|---|---|
| 100 | ODE | 5.46 ± 0.02 | 975 ± 833 | 6498 ± 1048 | 6498 ± 1048 | — | — |
| 100 | GEM+Girs | 0.156 ± 0.010 | 0.067 | 0.118 ± 0.016 | 0.181 | 428 | 24 |
| 100 | Heun+Girs | 0.179 ± 0.011 | 0.100 | 0.118 ± 0.019 | 0.177 | 412 | 20 |
| 250 | ODE | 0.644 ± 0.000 | 0.058 | 0.802 | 0.802 | — | — |
| 250 | GEM+Girs | 0.183 ± 0.006 | 0.096 | 0.125 ± 0.013 | 0.203 | 379 | 38 |
| 250 | Heun+Girs | 0.157 ± 0.015 | 0.093 | 0.079 ± 0.018 | 0.168 | 411 | 29 |
| 500 | ODE | 0.644 ± 0.000 | 0.058 | 0.802 | 0.802 | — | — |
| 500 | GEM+Girs | 0.178 ± 0.008 | 0.118 | 0.105 ± 0.022 | 0.175 | 400 | 43 |
| 500 | Heun+Girs | 0.225 ± 0.042 | 0.160 | 0.112 ± 0.016 | 0.166 | 432 | 39 |
| 1000 | ODE | 0.644 ± 0.000 | 0.058 | 0.802 | 0.802 | — | — |
| 1000 | GEM+Girs | 0.194 ± 0.038 | 0.145 | 0.115 ± 0.021 | 0.172 | 409 | 50 |
| 1000 | Heun+Girs | 0.180 ± 0.020 | 0.115 | 0.134 ± 0.024 | 0.180 | 398 | 47 |
| 2000 | ODE | 0.644 ± 0.000 | 0.058 | 0.802 | 0.802 | — | — |
| 2000 | GEM+Girs | 0.178 ± 0.007 | 0.061 | 0.139 ± 0.013 | 0.210 | 383 | 58 |
| 2000 | Heun+Girs | 0.176 ± 0.014 | 0.114 | 0.118 ± 0.018 | 0.190 | 408 | 55 |

**Interpretation.**

- **GEM's weight is exact for any step size.** GEM+Girs W1 is flat across K
  (0.156–0.194, within noise), as expected: for GEM the Girsanov correction coincides with the
  closed-form Gaussian ratio at *every* δ, so the target is K-independent and only the variance
  changes. This is direct confirmation of the `docs/idea.md` §4 identity.
- **Heun-SDE's approximate weight introduces no detectable bias.** Heun+Girs is statistically
  indistinguishable from GEM at every K (including K=100, where δ is large) — the
  constant-interpolation error is below the noise floor at N=512.
- **The underdispersion is not discretization.** `w|dstd|` stays ~0.10–0.14 and `u|dstd|` ~0.17–0.21
  even at K=2000, with no trend. Combined with the N-sweep plateau, the residual is a systematic
  effect — the λ=1 cloud modestly under-represents the secondary posterior mode (weight ~0.23),
  which inflates the true std (0.80 vs the dominant mode's ~0.60). It is neither a weight error
  (GEM's is exact) nor a discretization effect.
- **ODE is not a posterior sampler**: at K=100 the coarse deterministic guidance diverges
  numerically; at K≥250 it collapses to the same point estimate (W1 0.644, std≈0).

## 7. Discussion and Next Steps

- The **λ=1 (Girsanov/TDS) machinery is validated against ground truth**: both GEM and Heun-SDE
  proposals with the exact weight recover the analytic posterior well, with W1 strictly below the
  pBS baselines. GEM's weight is exact for any discretization; Heun-SDE's is unbiased to within
  noise.
- **pBS is confirmed inconsistent** as theory predicts, and the new weight-controlled arms show
  the effect is attributable to λ, not to the proposal choice.
- **Open question — modest std understatement**: both the N- and K-sweeps show a residual
  |dstd| ~0.10–0.20 that neither particle count nor step count removes, consistent with
  under-sampling the secondary posterior mode. Worth checking whether a stricter resampling
  scheme or a different proposal changes it.
- **Next steps**: (i) a noise-regime (σ_n) sweep, (ii) examining the mode-sampling residual,
  (iii) porting to Burgers/Darcy — which requires first defining an explicit observation
  likelihood, since the current `zeta`-scaled guidance does not correspond to a consistent SMC
  weight.
