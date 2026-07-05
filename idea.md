# Research Idea: Girsanov-Corrected SMC for Diffusion-Guided PDE Solvers

## 1. The Problem

In Millard et al. (arXiv:2601.23262), the SMC weight update requires the ratio of transition densities:

$$
G_{k-1}^{\text{TDS}}(x_k, x_{k-1}) = \frac{\tilde{p}_\theta(y \mid x_{k-1})}{\tilde{p}_\theta(y \mid x_k)} \cdot \frac{p_\theta(x_{k-1} \mid x_k)}{\tilde{p}_\theta(x_{k-1} \mid x_k, y)}
$$

For **first-order GEM** (Euler–Maruyama), the ratio $p_\theta / \tilde{p}_\theta$ is tractable (ratio of two Gaussians with the same covariance, different means). For **second-order SOSaG** (Heun with noise jittering), the transition is not Gaussian—the denoiser $D_\theta$ introduces nonlinearity in the Heun correction—so this ratio is intractable.

Millard et al. avoid this by switching to **pseudo-bootstrap (pBS)** weighting:

$$
G_{k-1}^{\text{pBS}}(x_k, x_{k-1}) = \frac{\tilde{p}_\theta(y \mid x_{k-1})}{\tilde{p}_\theta(y \mid x_k)}
$$

pBS drops the density ratio entirely, altering the target distribution. The result is asymptotically *inconsistent* (the limiting target is $\bar{p}_\theta(x_0 \mid y) \cdot \tilde{p}_\theta(y \mid x_0)^\rho$ rather than the true posterior). Empirically this works well, but the inconsistency is unsatisfying and leaves performance on the table.

---

## 2. The Proposed Idea

Replace the intractable density ratio with a **Girsanov (change-of-measure) correction** between the unguided and guided diffusion processes over each time interval $[t_k, t_{k-1}]$. Since both processes share the same diffusion coefficient $g(t)$ and differ only in drift, the Radon–Nikodym derivative depends only on the **drift difference**, which is proportional to $\nabla \log \tilde{p}_\theta(y \mid x)$—a quantity already being computed for guidance.

### 2.1 Reverse-Time SDE Parameterisation

Let $k$ index the reverse diffusion steps, and define the variance increment:

$$
\delta_k \coloneqq \sigma_k^2 - \sigma_{k-1}^2 > 0.
$$

The unguided reverse SDE (prior) is:

$$
dx_t = a(x_t, t)\,dt + g(t)\,dW_t^P,
$$

with $a(x_t, t) = -2\dot{\sigma}_t \sigma_t \nabla_{x_t} \log p_\theta(x_t, \sigma_t)$ and $g(t)^2 = 2\dot{\sigma}_t \sigma_t$. In discrete EM form, the prior drift is:

$$
a_k^{\text{EM}} \coloneqq \frac{D_\theta(x_k, \sigma_k) - x_k}{\sigma_k^2},
$$

so that the unguided step is $x_{k-1} = x_k + \delta_k a_k^{\text{EM}} + \sqrt{\delta_k}\,z$, with $z \sim \mathcal{N}(0, I)$.

The guided process adds the likelihood gradient as an extra drift term:

$$
b_k \coloneqq \nabla_{x_k} \log \tilde{p}_\theta(y \mid x_k).
$$

Thus the guided drift is $a_k^{\text{EM}} + b_k$, and the guided step is:

$$
x_{k-1} = x_k + \delta_k (a_k^{\text{EM}} + b_k) + \sqrt{\delta_k}\,z.
$$

### 2.2 Girsanov Correction in Discrete Time

Let $P$ be the path measure of the prior (unguided) process and $Q$ the path measure of the guided process. Since the diffusion coefficient is identical, Girsanov gives:

$$
\log \frac{dP_{[t_{k-1}, t_k]}}{dQ_{[t_{k-1}, t_k]}} =
\int_{t_{k-1}}^{t_k} b(x_s, s)^\top \, dW_s^Q
\;-\;
\int_{t_{k-1}}^{t_k} \dot{\sigma}_s \sigma_s \, \|b(x_s, s)\|^2 \, ds.
$$

The first term is an Itô integral; the second is the quadratic variation correction. Over the discrete step from $\sigma_k$ to $\sigma_{k-1}$, the Brownian increment is $\Delta W_k^Q = \sqrt{\delta_k}\,z$, and $\int \dot{\sigma}_s \sigma_s \, ds = \frac{1}{2}\delta_k$ (to first order in the discretisation).

Using **constant interpolation** of $b$ over the interval, the incremental Girsanov correction is:

$$
C_k \coloneqq \sqrt{\delta_k} \, b_k^\top z \;-\; \frac{1}{2}\,\delta_k \|b_k\|^2.
$$

### 2.3 Unified Weight Update

The full incremental log-weight combines the likelihood ratio with the Girsanov correction:

$$
\log w_{k-1}^{\text{inc}} =
\underbrace{\log \tilde{p}_\theta(y \mid x_{k-1}) - \log \tilde{p}_\theta(y \mid x_k)}_{\text{data / constraint fit}}
\;+\;
\underbrace{\lambda \cdot C_k}_{\text{Girsanov correction}},
$$

where $\lambda \in \{0, 1\}$ controls whether the path-measure correction is applied. More generally, the correction can be tempered:

$$
\log w_{k-1}^{\text{inc}} =
\rho \cdot \bigl[\log \tilde{p}(y\mid x_{k-1}) - \log \tilde{p}(y\mid x_k)\bigr]
\;+\;
\lambda \cdot C_k,
$$

with $\rho$ controlling the likelihood tempering (as in Millard’s pBS). This directly embeds the approach into the broader $(\lambda, \rho)$ parameter space.

**Key property**: Because $C_k$ depends only on $b_k$ and the noise $z$ (which is already sampled), no additional denoiser evaluations are required. The correction is a simple tensor operation vectorised over particles.

---

## 3. Proposals for the Guided Reverse SDE

Three numerical integrators are considered. The first two are from prior work; the third is the proposal of this work.

### 3.1 GEM: First-Order Euler–Maruyama [Existing]

The guided Euler–Maruyama (GEM) update is the direct Euler discretisation of the guided SDE:

$$
x_{k-1} =
x_k + \delta_k \frac{D_\theta(x_k, \sigma_k) - x_k}{\sigma_k^2}
+ \delta_k \nabla_{x_k}\log\tilde{p}_\theta(y\mid x_k)
+ \sqrt{\delta_k}\,z, \qquad z \sim \mathcal{N}(0,I).
$$

The transition $\tilde{p}_\theta^{\text{EM}}(x_{k-1}\mid x_k,y)$ is Gaussian with mean $\mu_{\text{guide}}$ and covariance $\delta_k I$. The unguided prior transition $p_\theta^{\text{EM}}(x_{k-1}\mid x_k)$ is also Gaussian with the same covariance but mean $\mu_{\text{prior}}$. The density ratio is closed-form, yielding the **Twisted Diffusion Sampler (TDS)** weight (Wu et al.).

### 3.2 SOSaG: Jittering + ODE + Guidance [Existing, Millard et al.]

The Second-Order Stochastic Guided (SOSaG) proposal is a heuristic hybrid that does not correspond to a clean SDE discretisation. It proceeds in three phases:

1. **Jittering** (stochastic): $\hat x_k = x_k + \sqrt{\hat\sigma_k^2 - \sigma_k^2}\,\psi$, $\psi\sim\mathcal{N}(0,I)$, $\hat\sigma_k = \sigma_k + \gamma\sigma_k$.
2. **ODE denoising** (deterministic): Heun's 2nd-order ODE step on $dx/d\sigma = (x-D_\theta)/\sigma$.
3. **Guidance correction** (deterministic): $x_{k-1} \leftarrow x_{k-1} - (\sigma_{k-1}^2 - \sigma_k^2)\,\nabla_{\hat x_k}\log\tilde{p}_\theta(y\mid\hat x_k)$.

Because the ODE step is deterministic and the guidance is applied as a post-hoc correction, the overall transition is not Gaussian and the density ratio is intractable. Millard et al. resort to **pBS weighting** ($\lambda=0$), which alters the target distribution.

### 3.3 Heun-SDE: Second-Order Stochastic Runge–Kutta [**New — This Work**]

We propose the **stochastic Heun method** (a standard 2nd-order SDE integrator for additive noise) as a proper SDE discretisation for the guided process. Unlike SOSaG, Heun-SDE:

- Discretises the **same SDE** as GEM (same drift, same diffusion) but to second order in the deterministic part.
- Uses the **same Brownian increment** $\Delta W = \sqrt{\delta_k}\,z$ throughout both stages.
- Therefore the Girsanov correction applies **exactly as in GEM**, with trivial Brownian increment recovery.

Let $a_k \coloneqq \frac{D_\theta(x_k, \sigma_k) - x_k}{\sigma_k^2}$ and $b_k \coloneqq \nabla_{x_k}\log\tilde{p}_\theta(y\mid x_k)$.

**Stage 1 (Euler prediction):**

$$
x_{\text{pred}} = x_k + \delta_k (a_k + b_k) + \sqrt{\delta_k}\,z.
$$

**Stage 2 (Heun correction):**

$$
\begin{aligned}
x_{k-1} &= x_k + \frac{\delta_k}{2}\bigl[ a_k + b_k + a(x_{\text{pred}}, \sigma_{k-1}) + b(x_{\text{pred}}, \sigma_{k-1}) \bigr] + \sqrt{\delta_k}\,z,
\end{aligned}
$$

where $a(x_{\text{pred}}, \sigma_{k-1}) = \frac{D_\theta(x_{\text{pred}}, \sigma_{k-1}) - x_{\text{pred}}}{\sigma_{k-1}^2}$, and $b(x_{\text{pred}}, \sigma_{k-1}) = \nabla_{x_{\text{pred}}}\log\tilde{p}_\theta(y\mid x_{\text{pred}})$.

**Key properties**:

- **Same noise structure as GEM**: The noise $z$ appears identically in both stages and is the **only** source of stochasticity.
- **Second-order deterministic drift**: The averaged drift gives $O(\delta^2)$ accuracy in the deterministic component.
- **Clean Girsanov**: Because the noise is identical to EM, $C_k = \sqrt{\delta_k}\,b_k^\top z - \frac{1}{2}\delta_k\|b_k\|^2$ remains valid. The midpoint $x_{\text{pred}}$ provides a **free evaluation point** for higher-order quadrature (e.g., trapezoidal rule) if desired, at no extra denoiser cost.

### 3.4 Proposal Comparison

| Property | GEM (1st order) | SOSaG (hybrid) | Heun-SDE [Ours] |
|----------|:---------------:|:--------------:|:----------------:|
| SDE discretisation | ✓ Euler–Maruyama | ✗ jitter+ODE+guide | ✓ Stochastic Heun |
| Drift accuracy | $O(\delta)$ | $O(\delta^2)$ (ODE part) | $O(\delta^2)$ |
| Density ratio tractable? | ✓ Gaussian | ✗ intractable | ✗ intractable |
| Girsanov $\Delta W$ recovery | ✓ exact from $z$ | ✗ approximate | ✓ exact from $z$ |
| Denoiser calls per step | 1 | 2 | 2 |
| Weighting method | TDS (Wu et al.) | pBS (Millard et al.) | **Girsanov [ours]** |

---

## 4. Analytic Verification: Constant Interpolation = EM Gaussian Ratio

For an Euler–Maruyama step, the constant-interpolation Girsanov correction exactly equals the closed-form Gaussian density ratio.

### Setup

Let $\delta = \sigma_k^2 - \sigma_{k-1}^2$, $a = \frac{D_\theta(x_k) - x_k}{\sigma_k^2}$, $b = \nabla\log\tilde{p}(y|x_k)$. Guided step:

$$
x_{k-1} = x_k + \delta(a + b) + \sqrt{\delta}\,z.
$$

Prior mean: $\mu_p = x_k + \delta a$. Guide mean: $\mu_g = x_k + \delta(a + b) = \mu_p + \delta b$.

### Closed-Form Density Ratio

The log-ratio of the two Gaussian transitions (both with covariance $\delta I$) is:

$$
\begin{aligned}
\log \frac{p_\theta^{\text{EM}}}{\tilde{p}_\theta^{\text{EM}}}
&= -\frac{1}{2\delta}\left(\|x_{k-1}-\mu_p\|^2 - \|x_{k-1}-\mu_g\|^2\right) \\
&= \sqrt{\delta}\, b^\top z - \frac{1}{2}\delta \|b\|^2.
\end{aligned}
$$

### Girsanov Correction (Constant Interpolation)

From Section 2.2, the Girsanov increment is $C_k = \sqrt{\delta}\, b^\top z - \frac{1}{2}\delta \|b\|^2$. This matches exactly. **Thus, for GEM, Girsanov with constant interpolation recovers the TDS weight.** For Heun-SDE, the same formula holds with the same $z$, so the correction remains exact in the SDE limit, while the proposal benefits from higher-order drift accuracy.

---

## 5. Experimental Design

### 5.1 Progression

1. **Burgers equation** (1D, 1-channel) — fast iteration.
2. **Darcy flow** (2D, 2-channel) — standard benchmark.
3. **Helmholtz / Navier-Stokes / Reaction-diffusion** — generalisation.

### 5.2 Primary Arms

| Arm | Proposal | Weight | Target | Status |
|-----|----------|--------|--------|--------|
| GEM-TDS | GEM | TDS ($\lambda{=}1$) | $p_\theta(x_0\mid y)$ | Existing (reproduction) |
| SOSaG-pBS | SOSaG | pBS ($\lambda{=}0$) | $\bar{p}_\theta(x_0\mid y)\,\tilde{p}_\theta(y\mid x_0)$ | Existing (reproduction) |
| **GEM-Girsanov** | GEM | Girsanov ($\lambda{=}1$) | $p_\theta(x_0\mid y)$ | **New — verification** |
| **Heun-SDE-Girsanov** | Heun-SDE | Girsanov ($\lambda{=}1$) | $p_\theta(x_0\mid y)$ | **New — primary result** |

GEM-Girsanov is expected to match GEM-TDS numerically (verification of Section 4). Heun-SDE-Girsanov is the main novel contribution: a second-order SDE proposal with consistent (exact-target) SMC weighting.

### 5.3 Ablation Studies

1. **Number of particles** ($N \in \{1,2,4,8,16\}$): ESS trajectory and final error.
2. **Number of steps** ($K \in \{500,1000,2000,4000\}$): Bias reduction in Girsanov approximation.
3. **Interpolation order** (constant vs. linear using the Heun midpoint): Convergence of the Girsanov integral.
4. **Girsanov correction strength** ($\lambda \in \{0, 0.5, 1.0\}$): Effect on ESS and accuracy (limited points due to compute).

### 5.4 Diagnostics

For every run, record:
- Effective Sample Size (ESS) trajectory.
- Per-step weight variance decomposition ($\mathrm{Var}[\log w_{k}^{\text{inc}}]$).
- Resampling times and number of unique particles.
- Final $L_2$ relative error.

### 5.5 Success Criteria

| Criterion | Interpretation |
|-----------|----------------|
| GEM-Girsanov matches GEM-TDS | Implementation verified (§4) |
| Heun-SDE-Girsanov error $\leq$ GEM-TDS error | Higher-order drift improves accuracy |
| Heun-SDE-Girsanov ESS $\geq$ GEM-TDS ESS | No penalty for 2nd-order proposal |
| SOSaG-pBS reproduces published results | Baseline correct |

---

## 6. Summary and Outlook

| Aspect | Assessment |
|--------|------------|
| **Novelty** | High. First application of Girsanov correction to diffusion-model SMC. First combination of a 2nd-order SDE integrator (Heun-SDE) with exact SMC weights. |
| **Mathematical soundness** | Verified: Girsanov with constant interpolation equals the closed-form EM ratio (§4). Heun-SDE is a proper SDE discretisation with identical noise structure to EM — the Girsanov correction applies without approximation. |
| **Practical impact** | Potentially high. If Heun-SDE+Girsanov beats or matches SOSaG-pBS, it provides a principled higher-order SMC method. If not, the comparison illuminates whether exact posterior weighting or proposal order matters more for PDE problems. |
| **Risk** | Low-medium. The mathematics is sound; the question is empirical. A negative result (Girsanov doesn't help) is still valuable, as it would support Millard’s evolutionary-algorithm interpretation of SMC. |

### Future Directions

The $(\lambda, \rho)$ parameter space defines a rich design space:

- $\lambda=0,\rho=1$: pBS (existing).
- $\lambda=1,\rho=1$: Girsanov / TDS (exact posterior).
- $\lambda=0,\rho\ne1$: tempered pBS (explored by Millard et al.).
- Full $(\lambda,\rho)$ grid: jointly varying correction and tempering.

A full sweep is computationally expensive and deferred. The primary experiments focus on the natural operating points $(\lambda,\rho) = (0,1)$ and $(1,1)$.
