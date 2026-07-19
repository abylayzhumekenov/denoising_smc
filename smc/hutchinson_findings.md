# Hutchinson-Laplacian Findings (Burgers, CPU sanity check)

Findings from validating whether the Hutchinson-estimated Laplacian term needed by the
Doob-transform / Ito's-formula weight derivation ("Consistent Particle Guided Denoising" draft,
not yet checked into this repo) is practical to compute -- as opposed to `idea.md`'s Girsanov
correction, which needs no second derivatives at all. Implementation: `smc/hutchinson.py`.

## Setup

- Real pretrained model: `pretrained-burgers.pkl` (208MB, from the project's Google Drive)
- Real test data: `data/testing/burgers.mat`, `offset=0`
- Real sparse-sensor mask: `random_sensor(5, 128, seed=0)` (matches `configs/burgers.yaml`)
- One representative point: `x_cur = randn(1,1,128,128) * sigma_t`, `sigma_t = 5.0`
- `ell(x) = -zeta_obs * L_obs(x)` using the repo's own `get_burger_loss`, `zeta_obs = 320`
- Hardware: CPU (Apple M1 laptop), float64 throughout (matches this repo's convention)

## 1. Architecture: double-backward works

`training/networks.py`'s custom `AttentionOp(torch.autograd.Function)` calls the internal
`torch._softmax_backward_data` in its `backward()`, which was a real risk for breaking
second-order autodiff (needed for any Hessian-vector product). Verified directly: double-backward
through `AttentionOp.apply` in isolation, and `torch.autograd.functional.vhp` through a real
`UNetBlock(attention=True)`, both succeed. Not a blocker.

## 2. Cost, at this resolution/precision, on this CPU

| Operation | Time |
|---|---|
| forward only | ~17-20s |
| forward + backward (today's guidance gradient) | ~23-45s |
| + 1 Hutchinson probe (reverse-over-reverse HVP) | +~85-95s marginal, per probe |

Complexity is O(M) in probe count, independent of state dimension d (~16k for this 128x128
1-channel field) -- the exact Laplacian would be O(d) HVPs, i.e. categorically infeasible
(days of compute per particle-step, extrapolated).

## 3. Variance: single probes are unreliable

M=16 independent Hutchinson probes at the point above:

```
[0.0048, -0.0092, -0.0057, -0.0040, 0.0034, -0.0099, -0.0366, -0.0217,
 -0.0064, -0.0072, 0.0011, -0.0076, -0.0016, -0.0326, -0.0121, -0.0104]
```

- Mean (M=16 estimate): **-0.009728**
- Sample std of a single probe: **0.011624**
- SEM of the M=16 mean: **0.002906**
- 95% CI: **[-0.01542, -0.00403]** (excludes zero)

Coefficient of variation for a *single* probe: `std/|mean| = 1.195` (~120% relative error) --
a single draw is dominated by noise, which is why the earlier M=1 (+0.0048) and M=2 (-0.0075,
sign-flipping) estimates were untrustworthy. Probes needed for a target relative error
(`rel_err = CV_1/sqrt(M)`), at ~90s/probe:

| Target rel. error | Probes needed | Time (this CPU) |
|---|---|---|
| 50% | 6 | ~9 min |
| 30% | 16 | ~24 min |
| 20% | 36 | ~54 min |
| 10% | 143 | ~3.6 hours |
| 5% | 572 | ~14.3 hours |

**Per particle, per diffusion step.** This is the binding constraint on using this weight in a
real SMC run, not raw per-probe cost.

## 4. Where the Laplacian sits among the other terms of V_tau

Full potential: `V_tau(x) = d(ell)/d(tau) + b^ap.grad_ell + (1/2)*a_bar*Laplacian(ell)
- (1/2)*a_bar*||grad_ell||^2`, with `a_bar(tau) = 2*sigma_t` (EDM "sigma-as-time" convention).
At the same point:

| Term | Value | Cost |
|---|---|---|
| `d(ell)/d(tau)` (finite difference, h=0.05*sigma_t) | +0.1134 | ~36s |
| `b^ap . grad_ell` (drift dot gradient) | **+0.3659** (dominant) | ~19s |
| `(1/2)*a_bar*Laplacian(ell)` | -0.0486 +/- 0.0145 | **~1470s for M=16** |
| `-(1/2)*a_bar*||grad_ell||^2` | -0.1107 | free (reuses grad_ell) |
| **V_tau(x) total** | **0.3200 +/- 0.0145** | |

The Laplacian term is the *smallest* of the four contributions, and the only one with any
Monte Carlo noise -- yet it costs ~40x longer than the other three terms combined.

## Caveats

- Single point on the trajectory (`sigma_t=5.0`); term magnitudes could look very different near
  `sigma -> 0` (likelihood sharpens) or `sigma -> sigma_max` (noise-dominated).
- `a_bar(tau) = 2*sigma` is a convention choice (identifying diffusion "time" with the noise
  level), not verified against the exact convention intended by the Doob-transform derivation.
- The finite-difference step for `d(ell)/d(tau)` (`h = 0.25` at `sigma_t=5`) has not been checked
  for convergence (e.g. by halving h).
- All measurements are CPU-only; a GPU would change absolute costs substantially but the O(M)
  vs. O(d) complexity argument, and the variance/probe-count relationship, should carry over.

## Practical takeaway

Before investing in a production implementation of this weighting scheme: check whether this
term-magnitude ranking (Laplacian smallest & noisiest) holds at other points in the trajectory,
and whether a deliberately crude/cheap estimate of it (or omitting it) changes SMC outcomes
enough to matter -- the cost here does not look proportional to its apparent contribution.
