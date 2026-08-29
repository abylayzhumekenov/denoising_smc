# Plug-in surrogate reliability — finding, experiment, and result

## Context
`smc/scripts_1/toy_smc.py` validates the Girsanov-corrected SMC weight on a closed-form
1D Gaussian-mixture posterior as a uniform grid: 5 twists (exact, surrogate-const,
consistent-annealed, plug-in, plug-in corr), 2 proposals (EM, Heun), 4 weightings (PBS, Girs,
Pot, Pot-tr), 4-seed mean±std, tables T1–T4.

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

## Experiment: residual-corrected plug-in (T1 row + T2–T4 re-baseline)
Add ONE new twist row: the residual-corrected plug-in

    r^2(sigma) = gamma^2 + c_bar(sigma),   c_bar(sigma) = E_x[sigma^2 * gradD(x, sigma)]

- `c_bar` is the expected residual posterior variance, a prior expectation derived from the
  SAME oracle denoiser D the naive plug-in already uses (`gradD` is already computed inside
  `plug_in_stats`).  In the toy it is tabulated once by quadrature of `sigma^2 gradD` against
  the exact mixture marginal and interpolated in log-sigma (`_cbar_schedule`), so it is
  seed-independent and costs nothing per step.
- Terminally consistent (`c_bar -> 0` at sigma -> 0): terminal correction (23) vanishes,
  like the consistent twist.  It additionally removes the plug-in's intermediate-sigma
  overconfidence, which is the root cause of the heavy tail.
- No η sweep.  We checked the annealed-width family `gamma^2[1+(eta-1)sigma/sigma_max]`
  against the true width numerically: at the canonical gamma^2=1 the true width is an S-curve
  (≈ gamma^2+sigma^2 near 0, ~2.1-3.6x gamma^2 through sigma~1.3-2.8, saturating ~4.5x at
  sigma_max).  eta=3-4 matches it only near sigma~0.4-0.6 and still leaves 1.3-1.5x residual
  overconfidence in the action range; matching the action range needs eta~6-7.  So the
  corrected schedule, not a linear proxy, is the deployable target.
- T1 now has 5 rows: exact, surrogate, consistent, plug-in (naive r^2=gamma^2), plug-in corr.
- T2–T4 re-baselined to `plug_in_corrected_twist` (option A: replace only; T1 keeps both
  plug-in rows as the validity contrast, no naive series in T2–T4).

## Result (N=2048, K=500, gamma^2=1, 4 seeds)
T1 Girs W1 (eq. 35):  plug-in naive **0.1600 ± 0.1827** (heavy tail) vs plug-in corr
**0.0354 ± 0.0126** — the corrected row sits in the same band as the exact (0.0374) and
consistent (0.0385) twists.  Tail removed, final ESS 1974, terminal corr ≈ 5e-7.

- T2 (base grid, EM/Heun): canonical corrected cell matches the T1 row (Girs 0.0354/0.0351,
  ESS ~1974-1983); no tail stds.
- T3 (convergence): N-sweep follows ~1/sqrt(N) with small stds (0.209 @32 -> 0.035 @2048);
  K-sweep shows the coarse-K discretisation error dropping to the ESS/MC floor.
- T4 (gamma2-sweep): **zero resampling across the whole sweep** and ESS stays >=1555 even at
  gamma^2=0.0625, W1 *improves* to 0.010.  On the naive baseline resampling rose 0->1->13->~65
  as gamma^2 shrank; that "degeneracy regime" was entirely the naive surrogate's
  overconfidence — with the residual-corrected width there is no regime boundary, because the
  Girsanov-corrected weights absorb the guidance drift (near the eq. 33 zero-variance ideal).

## Lessons
1. An overconfident surrogate is invisible to the standard health checks: the corrected SMC is
   unbiased and n^{-1/2}-convergent, yet heavy-tailed at moderate N while ESS stays high —
   high ESS + unbiasedness does not certify reliability.
2. The obvious width (r^2 = gamma^2) is the wrong one; the right width is the residual-corrected
   marginal `gamma^2 + c_bar(sigma)`, and it is the deployable choice, not a diagnostic oracle
   (it derives from the same denoiser D the naive plug-in already uses).
3. Heuristic width schedules mislead: the assumed eta~3-4 anneal leaves 1.3-1.5x overconfidence
   in the action range; checking against the closed-form truth was decisive.
4. A "fundamental" SMC degeneracy regime (T4) disappeared once the surrogate was corrected.

## Open decisions
- [x] family sweep (eta 1–4) vs a single realistic cell: single corrected cell (no sweep)
- [x] include the fully-corrected cell or drop it as unrealistic: include — it is the
      deployable target, not mechanism-only (c_bar derives from the same oracle D)
- [x] median vs mean reporting for heavy-tail robustness: keep mean (tail removed by the fix)
- [x] commit scope: `smc/scripts_1/` only (toy_smc.py, plan_plug_in.md, tables/, figures/)

## Implementation notes
- `PlugInTwist` now accepts a callable `r2(sigma)` (`obs` resolves it per step, mirroring
  `ExactFormTwist`); `plug_in_stats` exposes `D` and `gradD`.
- `_cbar_schedule()` (lru-cached) tabulates `c_bar(sigma)` by exact-marginal quadrature;
  `plug_in_corrected_twist(gamma2)` = `PlugInTwist(r2=lambda sigma: gamma2 + c_bar(sigma))`.
- `selfcheck` extended to the corrected twist; T1 twists dict, captions, compact matrix, and
  `toy_t1_diag.tex` diagnostics extended with the plug-in-corr row; T2–T4 captions reworded to
  the corrected baseline.