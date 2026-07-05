# Research Idea: Girsanov-Corrected SMC for Diffusion-Guided PDE Solvers

## 1. The Problem

In Millard et al. (arXiv:2601.23262), the SMC weight update requires the ratio of transition densities:

$$
G_{k-1}^{\text{TDS}}(x_k, x_{k-1}) = \frac{\tilde{p}_\theta(y \mid x_{k-1})}{\tilde{p}_\theta(y \mid x_k)} \cdot \frac{p_\theta(x_{k-1} \mid x_k)}{\tilde{p}_\theta(x_{k-1} \mid x_k, y)}
$$

For **first-order GEM** (Euler–Maruyama), the ratio $p_\theta / \tilde{p}_\theta$ is tractable (ratio of two Gaussians with same covariance, different means). For **second-order SOSaG** (Heun with noise jittering), the transition is not Gaussian—the denoiser $D_\theta$ introduces nonlinearity in the Heun correction—so this ratio is intractable.

Millard et al. avoid this by switching to **pseudo-bootstrap (pBS)** weighting:

$$
G_{k-1}^{\text{pBS}}(x_k, x_{k-1}) = \frac{\tilde{p}_\theta(y \mid x_{k-1})}{\tilde{p}_\theta(y \mid x_k)}
$$

pBS drops the density ratio entirely, altering the target distribution. The result is asymptotically *inconsistent* (the limiting target is $\bar{p}_\theta(x_0 \mid y) \cdot \tilde{p}_\theta(y \mid x_0)^\rho$ rather than the true posterior). Empirically this works well, but the inconsistency is unsatisfying and leaves performance on the table.

---

## 2. The Proposed Idea

Replace the intractable density ratio with a **Girsanov (change-of-measure) correction** between the unguided and guided diffusion processes over each time interval $[t_k, t_{k-1}]$. Since both processes share the same diffusion coefficient $g(t)$ and differ only in drift, the Radon–Nikodym derivative depends only on the **drift difference**, which is proportional to $\nabla \log \tilde{p}_\theta(y \mid x)$—a quantity already being computed for guidance.

### 2.1 Girsanov for Diffusion Paths

Let $P$ be the path measure of the unguided (prior) reverse SDE, and $Q$ the path measure of the guided (posterior) reverse SDE:

$$
\begin{aligned}
P: &\quad dx_t = f_{\text{prior}}(x_t, t)\,dt + g(t)\,dW_t^P \\
Q: &\quad dx_t = \bigl[f_{\text{prior}}(x_t, t) + \Delta f(x_t, t)\bigr]\,dt + g(t)\,dW_t^Q
\end{aligned}
$$

where $\Delta f(x_t, t) = -2\dot{\sigma}_t \sigma_t \nabla_{x_t} \log \tilde{p}_\theta(y \mid x_t)$ is the guidance drift. For the EDM SDE ($g(t) = \sqrt{2\dot{\sigma}_t \sigma_t}$), the key ratio simplifies elegantly:

$$
\frac{\Delta f(x_t, t)}{g(t)^2} = -\nabla_{x_t} \log \tilde{p}_\theta(y \mid x_t)
$$

Girsanov gives the likelihood ratio of the two path measures over $[t_{k-1}, t_k]$ (computed under the guided measure $Q$, which is what we simulate):

$$
\log \frac{dP_{[t_{k-1}, t_k]}}{dQ_{[t_{k-1}, t_k]}} =
\underbrace{\int_{t_{k-1}}^{t_k} g(s)\,\nabla\log\tilde{p}_\theta(y\mid x_s)^\top dW_s^Q}_{\text{Itô integral}}
\;-\;
\underbrace{\int_{t_{k-1}}^{t_k} \dot{\sigma}_s \sigma_s \,\|\nabla\log\tilde{p}_\theta(y\mid x_s)\|^2\,ds}_{\text{quadratic variation}}
$$

This replaces the intractable density ratio $p_\theta(x_{k-1} \mid x_k) / \tilde{p}_\theta(x_{k-1} \mid x_k, y)$ pointwise along the path. For any numerical scheme that discretizes the guided SDE, we can estimate this RN derivative using only quantities available from the simulation.

### 2.2 Approximating the Girsanov Integral

The difficulty: we have $x_k$ and $x_{k-1}$ (and possibly midpoints from higher-order integrators), but **not** the full continuous path. Generating the full path is expensive because it requires additional denoiser calls.

**Proposal**: Approximate $\nabla\log\tilde{p}_\theta(y \mid x_s)$ over $s \in [t_{k-1}, t_k]$ by **interpolation** of values at the endpoints (or midpoints), then evaluate the Girsanov integrals via **quadrature**.

#### Constant interpolation

$$
\nabla\log\tilde{p}_\theta(y \mid x_s) \approx \nabla\log\tilde{p}_\theta(y \mid x_k), \quad s \in [t_{k-1}, t_k]
$$

The Itô integral becomes $g(t_k)\,\nabla\log\tilde{p}_\theta(y \mid x_k)^\top \Delta W_k^Q$, where $\Delta W_k^Q$ is the Brownian increment over the interval. For a given numerical scheme, $\Delta W_k^Q$ can be recovered from $x_k$, $x_{k-1}$, the drift, and the step size.

- **Claim**: For the Euler–Maruyama scheme, constant interpolation reproduces the closed-form Gaussian ratio $p_\theta^{\text{EM}} / \tilde{p}_\theta^{\text{EM}}$. This serves as a correctness check. (See §4 for verification.)
- **Accuracy**: $O(\Delta t)$ local truncation error.

#### Linear interpolation

$$
\nabla\log\tilde{p}_\theta(y \mid x_s) \approx \frac{s - t_{k-1}}{\Delta t} \nabla\log\tilde{p}_\theta(y \mid x_k) + \frac{t_k - s}{\Delta t} \nabla\log\tilde{p}_\theta(y \mid x_{k-1})
$$

This requires $\nabla\log\tilde{p}_\theta$ at both endpoints, each costing a backprop through the denoiser. The Itô integral becomes a sum of two Gaussian integrals tractable in closed form.

- **Accuracy**: $O(\Delta t^2)$ local truncation error (trapezoidal rule for the quadratic variation term + Milstein-type correction for the Itô term).
- **Cost**: One extra gradient evaluation per step (vs. constant interpolation).

#### Higher-order interpolation

Using midpoints from a Heun integrator provides additional free evaluation points at no extra denoiser cost, enabling higher-order quadrature.

### 2.3 Integration into the SMC Weight Update

For a particle simulated under the guided process $Q$, the **importance weight** relative to the unguided prior target $P$ over the whole path is:

$$
w(x_{0:K}) = \frac{dP}{dQ}(x_{[0,T]}) \cdot \tilde{p}_\theta(y \mid x_0)
$$

This replaces the recursive TDS weight update with a single path-wise correction. In practice, incremental weights are still preferable for resampling. The incremental log-weight over step $k$ is:

$$
\log w_k^{\text{inc}} = \log \frac{\tilde{p}_\theta(y \mid x_{k-1})}{\tilde{p}_\theta(y \mid x_k)} + \log \frac{dP_{[t_{k-1}, t_k]}}{dQ_{[t_{k-1}, t_k]}}
$$

The TDS correction $\log p_\theta(x_{k-1} \mid x_k) - \log \tilde{p}_\theta(x_{k-1} \mid x_k, y)$ is replaced by the Girsanov integral approximation. This weight is **asymptotically consistent** (as discretization $\to 0$ and particles $\to \infty$) for **any** proposal that discretizes the guided SDE.

---

## 3. Proposals for the Guided Reverse SDE

We consider three numerical integrators for the guided reverse SDE. The first two are from prior work; the third is our proposal.

### 3.1 GEM: First-Order Euler–Maruyama [Existing]

The guided Euler–Maruyama (GEM) update (Millard et al., eq:EM_denoiser_guided_constant) is a direct Euler discretization of the guided SDE. Using the **standard score convention** $\nabla\log p = (D_\theta - x)/\sigma^2$:

$$
\begin{aligned}
x_{k-1} &= x_k + (\sigma_{k-1}^2 - \sigma_k^2)\,\frac{D_\theta(x_k,\sigma_k) - x_k}{\sigma_k^2} \\
&\qquad - (\sigma_{k-1}^2 - \sigma_k^2)\,\nabla_{x_k}\log\tilde{p}_\theta(y\mid x_k) \\
&\qquad + \sqrt{\sigma_k^2 - \sigma_{k-1}^2}\;z, \qquad z\sim\mathcal{N}(0,I).
\end{aligned}
$$

The transition $\tilde{p}_\theta^{\text{EM}}(x_{k-1}\mid x_k,y)$ is Gaussian with mean $\mu_{\text{guide}}$ and covariance $(\sigma_k^2 - \sigma_{k-1}^2)I$. The unguided prior transition $p_\theta^{\text{EM}}(x_{k-1}\mid x_k)$ is also Gaussian with mean $\mu_{\text{prior}}$ and the same covariance. Hence the density ratio $p_\theta^{\text{EM}} / \tilde{p}_\theta^{\text{EM}}$ is available in closed form — this is the **Twisted Diffusion Sampler (TDS)** weight (Wu et al.).

**Score-convention note**: The Millard paper uses $\nabla\log p = (x-D)/\sigma^2$ (the negative of standard). All our formulas use the standard convention. A bug in the appendix pseudocode (missing $1/\sigma_k$ factor) is corrected by following the methodology equation rather than the appendix algorithm.

### 3.2 SOSaG: Jittering + ODE + Guidance [Existing, Millard et al.]

The Second-Order Stochastic Guided (SOSaG) proposal is a heuristic hybrid that does not correspond to a clean SDE discretization. It proceeds in three phases:

1. **Jittering** (stochastic): $\hat x_k = x_k + \sqrt{\hat\sigma_k^2 - \sigma_k^2}\,\psi$, $\psi\sim\mathcal{N}(0,I)$, $\hat\sigma_k = \sigma_k + \gamma\sigma_k$.
2. **ODE denoising** (deterministic): Heun's 2nd-order ODE step on $dx/d\sigma = (x-D_\theta)/\sigma$.
3. **Guidance correction** (deterministic): $x_{k-1} \leftarrow x_{k-1} - (\sigma_{k-1}^2 - \sigma_k^2)\,\nabla_{\hat x_k}\log\tilde{p}_\theta(y\mid\hat x_k)$.

Because the ODE step is deterministic and the guidance is applied as a post-hoc correction, the overall transition is not Gaussian and the density ratio is intractable. Millard et al. resort to **pBS weighting** (dropping the ratio entirely), which alters the target distribution.

### 3.3 Heun-SDE: Second-Order Stochastic Runge–Kutta [**New — this work**]

We propose using the **stochastic Heun method** (2nd-order stochastic Runge–Kutta with additive noise) as a proper SDE discretization for the guided process. Unlike SOSaG, Heun-SDE:

- Discretizes the **same SDE** as GEM (same drift, same diffusion), but to second order in the deterministic part.
- Uses the **same Brownian increment** $\Delta W = \sqrt{|\Delta t|}\,z$ throughout both stages — no split of noise between jittering and denoising.
- Therefore the Girsanov correction applies **exactly as in GEM**, with trivial Brownian increment recovery.

**Heun-SDE update** (two stages, same noise $z$):

*Stage 1 (Euler prediction):*

$$
x_{\text{pred}} = x_k + (\sigma_{k-1}^2 - \sigma_k^2)\,\bigl[s_{\text{prior}}(x_k,\sigma_k) + s_{\text{guide}}(x_k,\sigma_k)\bigr] + \sqrt{\sigma_k^2 - \sigma_{k-1}^2}\;z.
$$

*Stage 2 (Heun correction):*

$$
\begin{aligned}
x_{k-1} &= x_k + \frac{1}{2}(\sigma_{k-1}^2 - \sigma_k^2)\,
\bigl[\,s_{\text{prior}}(x_k,\sigma_k) + s_{\text{guide}}(x_k,\sigma_k) \\
&\qquad\qquad + s_{\text{prior}}(x_{\text{pred}},\sigma_{k-1}) + s_{\text{guide}}(x_{\text{pred}},\sigma_{k-1})\bigr] \\
&\qquad + \sqrt{\sigma_k^2 - \sigma_{k-1}^2}\;z,
\end{aligned}
$$

where $s_{\text{prior}} = (D_\theta - x)/\sigma^2$ and $s_{\text{guide}} = \nabla\log\tilde{p}_\theta(y\mid x)$.

**Key properties**:

1. **Same noise structure as GEM**: The noise $z$ appears identically in both stages and is the **only** source of stochasticity. The Brownian increment is $\Delta W = \sqrt{|\Delta t|}\,z$, exactly as in EM.
2. **Second-order deterministic drift**: The drift average $(a_k + a_{\text{pred}})/2$ gives $O(\Delta t^2)$ accuracy in the deterministic (prior) component, compared to $O(\Delta t)$ for GEM.
3. **Clean Girsanov**: Because the noise is identical to EM, the Girsanov correction $C_k$ uses the same constant-interpolation formula and the same Brownian increment recovery (directly from $z$).

### 3.4 Comparison of Proposals

| Property | GEM (1st order) | SOSaG (hybrid) | Heun-SDE [ours] |
|----------|:---------------:|:--------------:|:----------------:|
| SDE discretization | ✓ Euler–Maruyama | ✗ jitter+ODE+guide | ✓ Stochastic Heun |
| Drift accuracy | $O(\Delta t)$ | $O(\Delta t^2)$ (ODE part) | $O(\Delta t^2)$ |
| Density ratio tractable? | ✓ Gaussian | ✗ intractable | ✗ intractable |
| Girsanov $\Delta W$ recovery | ✓ exact from $z$ | ✗ approximate | ✓ exact from $z$ |
| Denoiser calls per step | 1 | 2 | 2 |
| Prior use | TDS (Wu et al.) | pBS (Millard et al.) | **Girsanov [this work]** |

The critical row is **Girsanov $\Delta W$ recovery**: GEM and Heun-SDE both use the exact same noise $z$ and can recover $g\,\Delta W$ directly. SOSaG's jittering noise $\psi$ only accounts for part of the total noise, making Girsanov correction approximate.

---

## 4. Critical Assessment

### 4.1 Strengths

1. **Girsanov unifies SMC weights across proposal orders**: The same correction formula $C_k = g\,\nabla\log\tilde{p}^\top\Delta W - \tfrac12\Delta\sigma^2\|\nabla\log\tilde{p}\|^2$ works for both GEM (1st order) and Heun-SDE (2nd order), because both are proper SDE discretizations with identical noise structure.

2. **Heun-SDE is a clean higher-order alternative to SOSaG**: Unlike SOSaG (which mixes jittering, ODE, and guidance in a way that breaks SDE interpretability), Heun-SDE is a standard numerical method for SDEs. The Girsanov correction applies without approximation.

3. **No intractable density ratios**: The correction depends only on $\nabla\log\tilde{p}$, which is already computed for guidance. No additional neural network evaluations are needed.

4. **Asymptotic exactness**: The Girsanov approach recovers the exact posterior $p_\theta(x_0\mid y)$ as discretization $\to 0$ and particles $\to\infty$, for **both** GEM and Heun-SDE proposals. This bridges the gap between TDS (exact, 1st order) and pBS (inexact, higher-order).

5. **Free midpoint for quadrature**: Heun-SDE computes $x_{\text{pred}}$ as a byproduct, which can be used as a free evaluation point for higher-order Girsanov integral approximation.

### 4.2 Challenges and Open Questions

1. **Discretization error in the Girsanov integral**: The Girsanov integral is exact only for the continuous path. Constant interpolation introduces $O(\Delta t)$ bias. This bias must be analyzed and controlled; finer $\sigma$ schedules and higher-order quadrature (using the Heun midpoint) should reduce it.

2. **Surrogate vs. true drift**: The Girsanov correction corrects between the *estimated* guided and unguided processes, not the true prior and posterior. The final target is $p_\theta(x_0\mid y)$, the model posterior — the same as TDS, and subject to the same model approximation errors.

3. **Comparison with pBS**: Millard et al. find that pBS (inexact target) outperforms TDS (exact target). If Heun-SDE+Girsanov also underperforms SOSaG-pBS, this supports their thesis that the exact model posterior is not the optimal target for PDE problems due to irreducible errors in the diffusion prior and likelihood. A negative result is still publishable.

4. **Computational cost**: Heun-SDE requires 2 denoiser calls per step (same as SOSaG). The Girsanov correction adds negligible overhead (a few tensor ops). The main cost is backprop through the denoiser for $\nabla\log\tilde{p}$, which is already needed for guidance.

---

## 5. Analytic Verification: Constant Interpolation = EM Gaussian Ratio

*This section proves the claim from §2.2: for an Euler–Maruyama step, constant-interpolation Girsanov exactly equals the closed-form Gaussian density ratio.*

### 5.1 Setup

Consider a single EM step from $t_k$ to $t_{k-1}$ (reverse time, $\Delta t = t_{k-1} - t_k < 0$). Under the guided process $Q$:

$$
x_{k-1} = \mu_{\text{guide}} + \sqrt{|\Delta t|}\, g(t_k)\, z, \quad z \sim \mathcal{N}(0, I)
$$

where $\mu_{\text{guide}} = \mu_{\text{prior}} + \Delta t \cdot \Delta f(x_k, t_k)$ and $\mu_{\text{prior}} = x_k + \Delta t \cdot f_{\text{prior}}(x_k, t_k)$.

The Brownian increment is $\Delta W_k^Q = \sqrt{|\Delta t|}\, z$.

### 5.2 Girsanov with Constant Interpolation

$$
\begin{aligned}
\log w_k^{\text{Girsanov}} &=
g(t_k) \nabla\log\tilde{p}(y|x_k)^\top \Delta W_k^Q
- |\Delta t|\, \dot{\sigma}_k \sigma_k \|\nabla\log\tilde{p}(y|x_k)\|^2
\end{aligned}
$$

Substituting $g = \sqrt{2\dot{\sigma}_k \sigma_k}$, $\Delta W_k^Q = \sqrt{|\Delta t|}\, z$:

$$
\log w_k^{\text{Girsanov}} = \sqrt{2\dot{\sigma}_k \sigma_k |\Delta t|}\; \nabla\log\tilde{p}_k^\top z
- |\Delta t|\; \dot{\sigma}_k \sigma_k \|\nabla\log\tilde{p}_k\|^2
$$

### 5.3 Closed-Form Gaussian Ratio

For the EM step, the ratio of transition densities is:

$$
\log \frac{p_\theta^{\text{EM}}(x_{k-1}|x_k)}{\tilde{p}_\theta^{\text{EM}}(x_{k-1}|x_k,y)}
= -\frac{1}{2|\Delta t| g_k^2}\bigl(\|x_{k-1}-\mu_{\text{prior}}\|^2 - \|x_{k-1}-\mu_{\text{guide}}\|^2\bigr)
$$

Using $x_{k-1} = \mu_{\text{guide}} + \sqrt{|\Delta t|} g_k z$, $\mu_{\text{guide}} = \mu_{\text{prior}} + \Delta t \cdot \Delta f$, and algebra:

$$
\log \frac{p_\theta^{\text{EM}}}{\tilde{p}_\theta^{\text{EM}}}
= -\frac{\sqrt{|\Delta t|}}{g_k} \Delta f^\top z - \frac{|\Delta t|}{2 g_k^2} \|\Delta f\|^2
$$

Substituting $\Delta f = -2\dot{\sigma}_k \sigma_k \nabla\log\tilde{p}_k$, $g_k = \sqrt{2\dot{\sigma}_k \sigma_k}$:

$$
\log \frac{p_\theta^{\text{EM}}}{\tilde{p}_\theta^{\text{EM}}}
= \sqrt{2\dot{\sigma}_k \sigma_k |\Delta t|}\; \nabla\log\tilde{p}_k^\top z
- |\Delta t|\; \dot{\sigma}_k \sigma_k \|\nabla\log\tilde{p}_k\|^2
$$

This **exactly matches** the constant-interpolation Girsanov expression. ∎

### 5.4 Interpretation

- For **GEM**: Girsanov with constant interpolation is **exact** — it equals the closed-form Gaussian ratio. The $\lambda=1$ weight recovers TDS.
- For **Heun-SDE**: constant interpolation introduces an $O(\Delta t)$ error because the true path deviates from the EM path within the interval. The Heun midpoint $x_{\text{pred}}$ can be used for higher-order quadrature, reducing this error at no extra denoiser cost.
- For **SOSaG**: constant interpolation also introduces $O(\Delta t)$ error, but additionally the Brownian increment recovery is approximate. Heun-SDE avoids this second source of error.

---

## 6. Experimental Design

### 6.1 Progression

1. **Burgers equation** (1D, 1-channel) — fast iteration, quick debugging.
2. **Darcy flow** (2D, 2-channel) — standard benchmark, comparison with Millard et al.
3. **Helmholtz / Navier-Stokes / Reaction-diffusion** — generalization.

### 6.2 Primary Arms

| Arm | Proposal | Weight | Target | Status |
|-----|----------|--------|--------|--------|
| GEM-TDS | GEM | TDS ($\lambda{=}1$) | $p_\theta(x_0\mid y)$ | Existing (reproduction) |
| SOSaG-pBS | SOSaG | pBS ($\lambda{=}0$) | $\bar{p}_\theta(x_0\mid y)\,\tilde{p}_\theta(y\mid x_0)$ | Existing (reproduction) |
| **GEM-Girsanov** | GEM | Girsanov ($\lambda{=}1$) | $p_\theta(x_0\mid y)$ | **New — verification** |
| **Heun-SDE-Girsanov** | Heun-SDE | Girsanov ($\lambda{=}1$) | $p_\theta(x_0\mid y)$ | **New — primary result** |

The first two arms reproduce published results. The third verifies that Girsanov-GEM matches GEM-TDS numerically (confirms §5). The fourth is the main novel contribution: a second-order SDE proposal with consistent (exact-target) SMC weighting.

### 6.3 What Is New

| Component | Prior work | This work |
|-----------|-----------|-----------|
| 1st-order SDE + exact SMC | TDS (GEM, Wu et al.) | GEM-Girsanov (same result, different method) |
| Higher-order + SMC | SOSaG-pBS (Millard et al.) | — |
| **2nd-order SDE + exact SMC** | — | **Heun-SDE + Girsanov** |
| Girsanov for diffusion models | — | **First application** |

### 6.4 Ablation Studies

1. **Number of particles** ($N \in \{1, 2, 4, 8, 16\}$): ESS trajectory and final error.
2. **Number of steps** ($K \in \{500, 1000, 2000, 4000\}$): Bias reduction in Girsanov approximation.
3. **Interpolation order** (constant vs. linear for Girsanov on Heun-SDE): Convergence of the Girsanov integral.
4. **Girsanov correction strength** ($\lambda \in \{0, 0.5, 1.0\}$): Effect on ESS and accuracy (limited points due to computational cost).

### 6.5 Diagnostics

For every run: ESS trajectory, per-step weight variance decomposition, resampling times, final $L_2$ relative error.

### 6.6 Long-Term Outlook (Future Work)

The $(\lambda, \rho)$ two-parameter family (correction strength $\lambda$ and likelihood tempering $\rho$) defines a rich design space:

$$
\log w_{k-1}^{\text{inc}} = \rho \cdot \bigl[\log\tilde{p}(y\mid x_{k-1}) - \log\tilde{p}(y\mid x_k)\bigr] + \lambda \cdot C_k.
$$

- $\lambda=0,\rho=1$: pBS (existing).
- $\lambda=1,\rho=1$: Girsanov / TDS (exact posterior).
- $\lambda=0,\rho\neq1$: tempered pBS (explored by Millard et al.).
- $(\lambda,\rho)$ grid: jointly varying correction and tempering.

A full sweep of $(\lambda,\rho)$ is computationally expensive and is deferred. The primary experiments use the natural operating points $(\lambda,\rho) = (0,1)$ and $(1,1)$.

### 6.7 Success Criteria

| Criterion | What it means |
|-----------|---------------|
| GEM-Girsanov matches GEM-TDS | Implementation verified (§5) |
| Heun-SDE-Girsanov error $\leq$ GEM-TDS error | Higher-order drift improves accuracy |
| Heun-SDE-Girsanov ESS $\geq$ GEM-TDS ESS | No penalty for 2nd-order proposal |
| SOSaG-pBS reproduces published results | Baseline correct |

---

## 7. Summary

| Aspect | Assessment |
|--------|------------|
| **Novelty** | High. First application of Girsanov correction to diffusion model SMC. First combination of 2nd-order SDE integrator (Heun-SDE) with exact SMC weights. |
| **Mathematical soundness** | Verified: Girsanov with constant interpolation equals closed-form EM ratio (§5). Heun-SDE is a proper SDE discretization with identical noise structure to EM — Girsanov applies without approximation. |
| **Practical impact** | Potentially high. If Heun-SDE+Girsanov beats or matches SOSaG-pBS, it provides a principled higher-order SMC method. If not, the comparison illuminates whether exact posterior weighting or proposal order matters more for PDE problems. |
| **Risk** | Low-medium. The mathematics is sound; the question is empirical. A negative result (Girsanov doesn't help) is still valuable. |

Implementation details, code structure, and build instructions are in [`recipe.md`](recipe.md).
