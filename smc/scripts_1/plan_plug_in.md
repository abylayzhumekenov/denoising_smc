# Plug-in surrogate reliability — finding & next-experiment plan (T1 update)

## Context
`smc/scripts_1/toy_smc.py` validates the Girsanov-corrected SMC weight on a closed-form
1D Gaussian-mixture posterior as a uniform grid: 4 twists (exact, surrogate-const,
consistent-annealed, plug-in), 2 proposals (EM, Heun), 4 weightings (PBS, Girs, Pot,
Pot-tr), 4-seed mean±std, tables T1–T4.

## Finding: the naive plug-in surrogate is heavy-tailed at moderate N
- The plug-in twist `p~(y|x_sigma) = N(y; D(x,sigma), gamma^2)` is the realistic surrogate
  (terminally consistent: `D(x,0)=x`), but it ignores the residual uncertainty `c_k(sigma)`,
  so its guidance `b = (y-D)/r^2 * grad D` is 2–10× too strong at sigma ~ 1–3.
- The corrected SMC is **unbiased** (weight == brute-force discrete importance weight to
  3e-6; converges to the posterior, err 0.002 at N=32768), **but** its error distribution is
  heavy-tailed: at N=2048, ~6% of seeds give W1 ~0.4–0.5 while ESS stays ~1600–2000.
- Mechanism: overconfident guidance occasionally herds the whole cloud toward the
  observation y=0 (away from the prior-weighted posterior); the weights stay diffuse, so
  resampling cannot fix the misplacement (it prunes weight concentration, not cloud
  location); the rare escaped particles carry fat-tailed influence, so the CLT has not
  converged at N=2048 — n^{-1/2} and ESS are not reliable diagnostics here.
- Tail rate (W1 > 0.15) vs N: 19% @512, 6% @1024, 6% @2048, 0% @4096, 0% @8192.
- Root cause is the SURROGATE (through the proposal), not a bug: exact and consistent
  twists have no tails; only the overconfident plug-in does.

## Plan: T1 update (T2–T4 unchanged for now)
Add an annealed-width plug-in family to test a REALISTIC partial correction:

    r^2(sigma) = gamma^2 * [1 + (eta-1) * sigma/sigma_max],   eta in {1, 2, 3, 4}

- eta = 1 → naive plug-in (the failing case, already a T1 row).
- eta > 1 widens the surrogate at intermediate sigma (reduces the overconfidence);
  terminally consistent; realistic hyperparameter (zeta_obs analogue in the real pipeline).
- At eta ~ 3–4 the width over the action range matches the (unrealistic) full correction
  `r^2 = gamma^2 + c_bar(sigma)`, so the annealed family is a deployable proxy.
- Expected: locate where the catastrophic tail disappears. If eta ~ 2–3 fixes it, the
  practical guidance is "widen the guidance variance schedule" — a usable conclusion.
- Optional: include the fully-corrected plug-in (`gamma^2 + c_bar(sigma)`) as a diagnostic
  upper bound (unrealistic — a trained denoiser does not expose c_bar — mechanism only).

New T1 rows: exact, surrogate, consistent, plug-in(eta=1), plug-in(eta=2), plug-in(eta=3),
plug-in(eta=4). Same metrics (mean±std) and plots, 4 seeds.

## Open decisions
- [ ] family sweep (eta 1–4) vs a single realistic cell (e.g. eta=3)
- [ ] include the fully-corrected cell or drop it as unrealistic
- [ ] median vs mean reporting for heavy-tail robustness
- [ ] commit scope

## Implementation notes
- Generalize `PlugInTwist` to accept a callable `r2(sigma)`; resolve it in `plug_in_stats`.
- Add `plug_in_annealed_twist(gamma2, eta)` and optionally `plug_in_corrected_twist(gamma2)`.
- Extend `selfcheck` to the new twists; extend the T1 twists dict, captions, and the
  T1 diagnostics (terminal corr ~ 0, final ESS).