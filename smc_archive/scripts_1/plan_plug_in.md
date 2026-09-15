# Plug-in surrogate reliability — experiment plan and result

## Context
`smc/scripts_1/toy_smc.py` validates the Girsanov-corrected SMC weight on a closed-form
1D Gaussian-mixture posterior.  The experiment grid is now:

- **4 twists:** `exact`, `plug_in`, `plug_in_annealed`, `plug_in_corrected`.
- **2 proposals:** EM, Heun.
- **4 weightings:** PBS, Girs, Pot, Pot-tr.
- **4-seed** mean±std, tables T1–T4.

The old `surrogate` (constant misspecified `γ̃ = 0.5γ`) and `consistent` (annealed exact-form)
twists have been removed; they were pedagogical holdovers from the PDF note and no longer
match the paper's focus.

## Twists

| name | width `r²(σ)` | role |
|---|---|---|
| `exact` | exact intermediate likelihood | validates the machinery (zero-variance ideal) |
| `plug_in` | `γ²` | realistic naive baseline; ignores residual uncertainty |
| `plug_in_annealed` | `γ² (1 + 2σ/σ_max)` | heuristic, oracle-free compromise |
| `plug_in_corrected` | `γ² + c̄(σ)` | oracle residual-corrected target |

`plug_in_annealed` uses `η = 3`, chosen to sit between the naive plug-in (`η = 1`) and the
corrected surrogate (`η ≈ 4.5` at `σ_max` in the toy).  It is terminally consistent and
requires no oracle beyond the denoiser already used for guidance.

## Finding: the naive plug-in surrogate is heavy-tailed at moderate N
- The plug-in surrogate `p̃(y|x_σ) = N(y; D(x,σ), γ²)` is terminally consistent
  (`D(x,0)=x`), but it ignores the residual uncertainty `c̄(σ)`.
- The corrected SMC is **unbiased**, but its error distribution is heavy-tailed at moderate
  `N`: some seeds give W1 ~0.4–0.5 while ESS stays ~1600–2000.
- Mechanism: overconfident guidance occasionally herds the cloud toward the observation;
  resampling fixes weight concentration, not cloud location.  Rare escaped particles carry
  fat-tailed influence, so `N^{-1/2}` and ESS are not reliable diagnostics.
- Root cause is the **surrogate width** (through the proposal), not a bug.

## Result: residual correction removes the tail; annealing is a practical compromise
- `plug_in_corrected` gives W1 comparable to the exact twist (~0.035) with small std and
  ESS ~1970.
- `plug_in_annealed` is the oracle-free middle ground.  It removes the worst of the naive
  plug-in's overconfidence while remaining simple to deploy.

## Experiment grid

- **T1 Validity:** all four twists × four weightings (EM).  Shows that corrected weightings
  target the posterior for every terminally-consistent twist, while PBS is biased.
- **T2 Base grid:** proposal × weighting at the canonical `plug_in_annealed` twist.
- **T3 Convergence:** K-sweep and N-sweep over the `plug_in_annealed` baseline.
- **T4 Regime:** `γ²`-sweep over the `plug_in_annealed` baseline.

`plug_in_corrected` appears only in T1 as an oracle reference; T2–T4 use the realistic
annealed surrogate as the baseline.

## Implementation notes
- `PlugInTwist` accepts a callable `r2(sigma)`.
- `_cbar_schedule()` tabulates `c̄(σ)` by exact-marginal quadrature for `plug_in_corrected`.
- `plug_in_annealed(gamma2, eta=3.0)` uses a simple linear width schedule.
- `selfcheck` covers all four twists.
