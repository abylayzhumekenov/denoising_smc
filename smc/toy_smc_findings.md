# Toy SMC Validation Findings (1D Gaussian Mixture)

Findings from validating the λ-ρ SMC weighting machinery (`docs/idea.md`/`docs/recipe.md`) against an
analytic posterior, using a cheap 1D Gaussian-mixture model instead of the expensive
network-based PDE pipeline. Implementation: `smc/toy_smc.py`. Run:

```bash
venv/bin/python smc/toy_smc.py
```

## Setup

- Prior: mixture `w=[0.6,0.4]`, `μ=[-1.5,2.0]`, `σ²=[0.7,1.3]`
- VE/EDM forward noising `x_t = x_0 + σ_t z`; noised marginals and score are closed form
  (mixtures are closed under Gaussian convolution — a "perfect score" model)
- Observation: `y = x + n`, `n ~ N(0, 0.5)`, with `y = 0.8` (full observation)
- EDM power-law schedule: `σ_max=6`, `σ_min=1e-3`, `K=500`, `rho=7`
- Exact initialization from the noised marginal `p_{σ_max}` (not from Gaussian noise)
- Analytic posterior via Kalman update per component: **mean = 0.8334, std = 0.8015**
- numpy `float64`; a `RLIMIT_AS` guard caps memory as insurance

## Variants

| Arm | Proposal | Weight | Notes |
|-----|----------|--------|-------|
| ODE | deterministic Heun (guided drift) | — | repo baseline; no SMC |
| SOSaG+pBS | SOSaG (Millard et al.) | λ=0 | baseline reproduction |
| GEM+pBS | GEM | λ=0 | proposal-controlled pBS |
| HeunSDE+pBS | Heun-SDE | λ=0 | proposal-controlled pBS |
| GEM+Girs | GEM | λ=1 | **= TDS exactly** (`docs/idea.md` §4 identity) |
| HeunSDE+Girs | Heun-SDE | λ=1 | novel arm |

## Key Findings

### 1. Conceptual error found: sign of the Girsanov Itô term

Re-deriving the weight exposed a genuine sign error in `docs/idea.md` §2.2/§2.3/§3.3/§4 and
`docs/recipe.md` §3.4/§4.3. The weight needs `dP/dQ` (prior transition / guided transition):

```
log(p_P/p_Q) = -√δ · bᵀz - ½δ‖b‖²
```

The docs (and the original toy) used `+√δ · bᵀz - ½δ‖b‖²`. The error was confirmed two ways:
direct closed-form Gaussian ratio (`log(p_P/p_Q) = -(1/2δ)[‖x-μ_p‖² - ‖x-μ_g‖²]`) and a 1D
numeric Girsanov sanity check. Because `E[bᵀz] = 0`, the *expected* weight is identical either
way — so posterior *means* looked correct while ESS and the spread were corrupted. Fixed in the
toy and both docs. Verification now passes:

```
GEM C_k vs closed-form Gaussian ratio: max|diff| = 2.97e-13
```

### 2. Broadcasting bugs (memory + statistics)

Two instances of the same `(N,)` vs `(N,1)` broadcasting trap:

- `score` used `x[:, None]`, returning `(N,1,1)`; broadcasting against `(N,1)` produced
  `(N,N,1)`, then `(N,N,N,1)`… an **exponential blowup** (2 GiB at 4 dims, 256 GiB next).
  This was the original OOM that killed the host machine, not a system-side memory issue.
  Fixed by dropping the extra axis.
- `weighted_stats` computed `(w * x).sum()` with `w` shape `(N,)` and `x` shape `(N,1)`,
  silently creating an `(N,N)` matrix (weighted mean = `Σw·Σx` ≈ 113 instead of ~0.9).
  Fixed by indexing `x[:, 0]`.

### 3. Main results (N=128, K=500, 8 seeds, mean ± SEM)

| variant | W1 | \|dmean\| | \|dstd\| | ESS | resamp |
|---|---|---|---|---|---|
| ODE | 0.644 ± 0.000 | 0.058 | 0.802 | — | — |
| SOSaG+pBS | 0.268 ± 0.057 | 0.149 | 0.122 | 91 | 86 |
| GEM+pBS | 0.207 ± 0.011 | 0.030 | 0.354 | 90 | 6 |
| HeunSDE+pBS | 0.215 ± 0.015 | 0.045 | 0.357 | 98 | 5 |
| GEM+Girs | 0.191 ± 0.011 | 0.129 | 0.169 | 110 | 37 |
| **HeunSDE+Girs** | **0.176 ± 0.011** | 0.094 | 0.144 | 100 | 31 |

- **W1 ranks the arms correctly**: HeunSDE+Girs < GEM+Girs < pBS arms < ODE. Both λ=1 arms are
  closest to the true posterior in full distribution.
- **`|dstd|` cleanly separates weights**: λ=1 arms ~0.14–0.17 (weighted std ≈ 0.63–0.66) vs
  pBS arms ~0.35 (weighted std ≈ 0.45). The pBS inconsistency (`docs/idea.md` §1) is real and is
  driven by the **weight (λ=0)**, not the proposal.
- **ODE collapses to a point estimate** (std=0): deterministic guided flow gives no uncertainty,
  mirroring the current repo pipeline.
- `|dmean|` is too noisy here (final-log-weight estimator after resampling) to discriminate
  arms; W1 and `|dstd|` are the reliable metrics.

### 4. N-sweep (λ=1 arms, K=500, 8 seeds)

| N | GEM W1 | GEM \|dstd\| | GEM ESS | Heun W1 | Heun \|dstd\| | Heun ESS |
|---|---|---|---|---|---|---|
| 128 | 0.191 ± 0.011 | 0.169 ± 0.022 | 110 | 0.176 ± 0.011 | 0.144 ± 0.016 | 100 |
| 512 | 0.178 ± 0.008 | 0.105 ± 0.022 | 400 | 0.225 ± 0.042 | 0.112 ± 0.016 | 432 |
| 2048 | 0.147 ± 0.013 | 0.102 ± 0.016 | 1626 | 0.151 ± 0.018 | 0.112 ± 0.011 | 1693 |

- `|dstd|` shrinks from ~0.17 to ~0.10 as N grows but **plateaus at ~0.10–0.11**, so the
  residual underdispersion is *not purely* a finite-N artifact.
- ESS scales roughly linearly with N (110 → 400 → 1626) — the filter behaves healthily.

## Caveats

- Single toy instance: one observation `y`, one mixture, `σ_max=6`. Term magnitudes and ESS
  behavior could differ in other regimes.
- The residual `|dstd| ~ 0.10` plateau likely reflects the Girsanov constant-interpolation
  discretization (Heun-SDE has no closed-form weight) — a **K-sweep is the deferred test**.
- `σ_n` (likelihood sharpness), ρ-tempering, Heun linear-interpolation Girsanov, and
  exact-vs-Gaussian init comparisons are all deferred.
- **Real-pipeline port**: the current repo guidance (`zeta_obs`-scaled loss gradient, two-phase)
  is not a proper likelihood with a defined `σ_n`, so it has no consistent SMC weight. The
  PDE port will require redefining an explicit observation model before `smc/toy_smc.py`'s
  machinery can be applied to Burgers/Darcy.
