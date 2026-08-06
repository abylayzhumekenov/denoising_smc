# Literature Survey: Diffusion Models, SMC, and Posterior Sampling for PDEs

## Overview

This document summarizes the collection of papers in `literature/`, organized by methodology. The central theme is combining **denoising diffusion models** with **Sequential Monte Carlo (SMC)** for solving inverse problems, with particular focus on PDE-constrained settings.

---

## 1. Core Diffusion Foundations

### 1.1 DDPM — Denoising Diffusion Probabilistic Models
- **arXiv**: 2006.11239
- **Authors**: Jonathan Ho, Ajay Jain, Pieter Abbeel
- **Venue**: NeurIPS 2020
- **Key Idea**: Forward noising process gradually adds Gaussian noise to data via a Markov chain. Reverse process learns to denoise via a neural network trained with a simple MSE loss (denoising score matching).
- **Formulation**:
  - Forward: $q(x_k \mid x_{k-1}) = \mathcal{N}(\sqrt{\alpha_k} x_{k-1}, \beta_k I)$
  - Reverse: $p_\theta(x_{k-1} \mid x_k) = \mathcal{N}(\mu_\theta(x_k, k), \sigma_k^2 I)$
  - Loss: $L = \mathbb{E}_{k, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_k, k)\|^2]$
- **Relevance**: Foundational model; the score network $\epsilon_\theta$ is used in all downstream methods.

### 1.2 Score SDE — Score-Based Generative Modeling through SDEs
- **arXiv**: 2011.13456
- **Authors**: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole
- **Venue**: ICLR 2021
- **Key Idea**: Unifies DDPM and score matching with Langevin dynamics under a stochastic differential equation (SDE) framework. Introduces the probability flow ODE for deterministic sampling.
- **Formulation**:
  - Forward SDE: $dx = f(x, t) dt + g(t) dw$
  - Reverse SDE: $dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) d\bar{w}$
  - Probability flow ODE: $dx = [f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)] dt$
- **Relevance**: Provides the continuous-time foundation that all subsequent SMC and guidance methods build on.

---

## 2. Posterior Sampling from Diffusion Models (Non-SMC)

### 2.1 FPS — Filtering Posterior Sampling
- **File**: `literature/arXiv-0000.00000v/paper.md` (custom methodology writeup)
- **Key Idea**: Reformulates posterior sampling as a reverse-time Bayesian filtering problem. Constructs a *duplex forward diffusion* that jointly noises the data $x$ and measurements $y$, then derives an analytic posterior via Gaussian conditioning.
- **Formulation**:
  - Duplex forward: $x_k = \sqrt{\bar{\alpha}_k} x_0 + \sqrt{1 - \bar{\alpha}_k} z$, $y_k = \sqrt{\bar{\alpha}_k} y_0 + \sqrt{1 - \bar{\alpha}_k} A z$
  - Implies: $q(y_k \mid x_k) = \mathcal{N}(A x_k, \sigma^2 \bar{\alpha}_k I)$
  - Filtering posterior (DDIM-based):
    $$\Sigma_k^{\text{FPS}} = \left( (\Sigma_k^{\text{DDIM}})^{-1} + \frac{1}{\sigma^2 \bar{\alpha}_{k-1}} A^\top A \right)^{-1}$$
    $$\mu_k^{\text{FPS}} = \Sigma_k^{\text{FPS}} \left( (\Sigma_k^{\text{DDIM}})^{-1} \mu_k^{\text{DDIM}} + \frac{1}{\sigma^2 \bar{\alpha}_{k-1}} A^\top y_{k-1} \right)$$
- **Limitations**: Restricted to linear inverse problems $y = Ax + n$; assumes known linear forward operator $A$ and Gaussian noise.
- **Experimental Setting**: FFHQ-1k and ImageNet-1k images; tasks include super-resolution, inpainting, deblurring.

### 2.2 FPS-SMC — Sequential Monte Carlo Extension of FPS
- **File**: Same as FPS (Section "FPS-SMC")
- **Key Idea**: Uses $M$ particles with resampling to provide asymptotic consistency. Particles are propagated via the FPS filtering posterior and resampled according to likelihood weights.
- **Algorithm**:
  1. Generate measurement sequence $\{y_k\}_{k=0}^N$ via DDIM backward process
  2. Initialize $M$ particles $x_N^{(j)} \sim p_\theta(x_N \mid y_N)$
  3. For $k = N, \dots, 1$:
     - Sample $\bar{x}_{k-1}^{(j)} \sim p_\theta(x_{k-1} \mid x_k^{(j)}, y_{k-1})$
     - Resample with weights $\eta_j \propto p_\theta(y_{k-1} \mid \bar{x}_{k-1}^{(j)})$
  4. Output: uniform sample from $\{x_0^{(j)}\}$
- **Theoretical Guarantee**: Under perfect score estimation and zero discretization error, converges weakly to the true posterior as $M \to \infty$.
- **Hyperparameters**: $M = 20$ (default), $N = 1000$ diffusion steps, DDIM noise parameter $c$ task-specific (0.15--0.95).

---

## 3. SMC + Diffusion for Inverse Problems

### 3.1 TDS — Twisted Diffusion Sampler
- **arXiv**: 2306.17775
- **Authors**: Francisco Vargas, Will Grathwohl, Arnaud Doucet
- **Venue**: NeurIPS 2023
- **Key Idea**: Recognizes that the Markov structure of diffusion models permits factorization as an SMC target. Uses *twisting functions* $\tilde{p}(y \mid x_t) := p(y \mid \hat{x}_\theta(x_t))$ (likelihood evaluated at the denoised estimate) to approximate the optimal SMC proposal.
- **Formulation**:
  - Twisted proposal: $\tilde{r}_t(x_t \mid x_{t+1}, y) = \mathcal{N}(x_t; x_{t+1} + \sigma^2 \tilde{s}_\theta(x_{t+1}, y), \sigma^2 I)$
  - Conditional score approx: $\tilde{s}_\theta(x_{t+1}, y) = s_\theta(x_{t+1}) + \nabla_{x_{t+1}} \log \tilde{p}(y \mid x_{t+1})$
  - Weight update: $w_t \propto \frac{p_\theta(x_t \mid x_{t+1}) \tilde{p}(y \mid x_t)}{\tilde{p}(y \mid x_{t+1}) \tilde{r}_t(x_t \mid x_{t+1}, y)}$
- **Theoretical Guarantee**: Asymptotically exact — under regularity conditions, the particle approximation converges setwise to the true posterior as $K \to \infty$, regardless of approximation quality in the twisting function.
- **Applications**: MNIST inpainting, class-conditional generation, protein motif-scaffolding. Extends to Riemannian manifold diffusion models.
- **Limitations**: The twisting function requires the likelihood to be differentiable and strictly positive. For non-differentiable conditioning (e.g., inpainting with Dirac delta), alternative constructions are needed.

### 3.2 MCGDiff — Monte Carlo Guided Diffusion for Linear Inverse Problems
- **arXiv**: 2308.07983
- **Authors**: Gabriel Cardoso, Yazid Janati El Idrissi, Sylvain Le Corff, Eric Moulines
- **Venue**: ICLR 2024
- **Key Idea**: Designs a provably consistent SMC sampler specifically for Bayesian *linear* inverse problems with denoising diffusion priors. Proposes an auxiliary particle filter (APF) that leverages the analytic structure of the linear Gaussian observation model.
- **Formulation**:
  - For the noiseless case ($\sigma = 0$): constructs intermediate targets $\phi_t(x_t) \propto p_t(x_t) \cdot \mathcal{N}(\alpha_t^{1/2} y; \text{top}(x_t), (1 - \alpha_t) I)$ and uses analytic Gaussian conditioning for optimal proposals.
  - For the noisy case ($\sigma > 0$): diffuses the observation forward to time $\tau$ where $\sigma^2 = (1 - \alpha_\tau) / \alpha_\tau$, then solves a noiseless problem at $\tau$, and propagates back to $t = 0$.
  - Key insight: $\mathcal{N}(y; \text{top}(x_0), \sigma^2 I) \propto \mathcal{N}(\tilde{y}_\tau; \alpha_\tau^{1/2} x_0, (1 - \alpha_\tau) I)$, linking the likelihood to the forward diffusion kernel.
- **Theoretical Guarantee**: Provably consistent (first such guarantee for SGM-based posterior sampling). Empirical distribution converges to the true posterior as $N \to \infty$ under the assumption that the backward and forward processes reverse each other.
- **Key Result**: Shows empirically that DPS and DDRM produce samples inconsistent with the posterior (failing the Bayesian recovery task), while MCGDiff does not.
- **Limitations**: Restricted to *linear* Gaussian inverse problems; assumes forward operator $A$ is known and the noise variance is known.

### 3.3 PDDS — Particle Denoising Diffusion Sampler
- **arXiv**: 2402.06320
- **Authors**: Angus Phillips, Hai-Dang Dau, Michael John Hutchinson, Valentin De Bortoli, George Deligiannidis, Arnaud Doucet
- **Venue**: ICML 2024
- **Key Idea**: Adapts the guided diffusion framework to sampling from *unnormalized probability densities* (rather than generative modeling). Develops an SMC scheme to provide consistent estimates.
- **Formulation**: Represents the target as $\pi(x) \propto p_0(x) g_0(x)$ where $p_0 = \mathcal{N}(0, I)$ and $g_0(x) = \gamma(x)/p_0(x)$. Diffuses $p_0$ forward via an OU process, then uses guidance (via $g_0$) in the reverse SDE to target $\pi$. The SMC corrects for approximation errors in the score and guidance terms.
- **Theoretical Contribution**: Quantifies errors introduced by guided diffusions; establishes limit theorems for the SMC scheme; introduces a novel score matching loss to reduce variance.
- **Relevance**: Bridges the gap between denoising diffusion models and Monte Carlo sampling from unnormalized densities.

---

## 4. Millard et al. — Diffusion SMC for PDEs (Central Paper)

### 4.1 Particle-Guided Diffusion Models for PDEs
- **arXiv**: 2601.23262
- **Authors**: Andrew Millard, Fredrik Lindsten, Zheng Zhao (Linköping University)
- **Venue**: ICML 2026
- **Codebase**: Built on and extends the DiffusionPDE framework (this repository)

#### Problem Setting
Solve PDE systems from sparse observations using a pretrained diffusion model over the joint space of coefficient fields $a$ and solution fields $u$ (i.e., $x = (a, u) \in \mathcal{A} \times \mathcal{U}$). The diffusion model acts as a prior $p_\theta(x)$, and PDE residuals + sparse observations provide the conditioning likelihood $p(y \mid x)$.

#### PDE Residual Likelihood
The likelihood combines three terms:
$$\log p(y \mid x) = -\beta \cdot \text{MSE}(u_{\text{obs}} - \mathcal{M}_u \odot x) - \gamma \cdot \text{MSE}(a_{\text{obs}} - \mathcal{M}_a \odot x) - \omega \cdot \text{MSE}(f(c, \tau, x))$$

where $\mathcal{M}_{u,a}$ are binary observation masks and $f$ is the PDE residual.

Intermediate likelihoods are approximated via Tweedie's formula:
$$\tilde{p}_\theta(y \mid x_t) \approx p(y \mid \hat{x}_0 = D_\theta(x_t, \sigma_t))$$

#### SMC Framework
Uses the Feynman-Kac formulation with $N=4$ particles across the reverse diffusion. Two proposal variants:

1. **GEM (Guided Euler-Maruyama)**: First-order SDE integrator with score + guidance gradient:
   $$x_{k-1} = x_k - (\sigma_{k-1}^2 - \sigma_k^2) \frac{x_k - D_\theta(x_k, \sigma_k)}{\sigma_k^2} - (\sigma_{k-1}^2 - \sigma_k^2) \nabla_{x_k} \log \tilde{p}_\theta(y \mid x_k) + \sqrt{\sigma_{k-1}^2 - \sigma_k^2} \, z$$

2. **SOSaG (Second-Order Stochastic Guided)**: Uses EDM's 2nd-order Heun correction with stochastic noise jittering before the denoising step:
   - Jitter: $\hat{\sigma}_k = \sigma_k + \gamma_k \sigma_k$, $\hat{x}_k = x_k + \sqrt{\hat{\sigma}_k^2 - \sigma_k^2} \, \psi$
   - Denoise + guide: $x_{k-1} = \hat{x}_k + (\sigma_{k-1}^2 - \hat{\sigma}_k^2) \frac{\hat{x}_k - D_\theta(\hat{x}_k, \hat{\sigma}_k)}{\hat{\sigma}_k^2} - (\sigma_{k-1}^2 - \hat{\sigma}_k^2) \nabla_{\hat{x}_k} \log \tilde{p}_\theta(y \mid \hat{x}_k)$

#### Weighting Strategies

**TDS-style weighting** (GEM-TDS):
$$G_{k-1}(x_k, x_{k-1}) = \frac{\tilde{p}_\theta(y \mid x_{k-1})}{\tilde{p}_\theta(y \mid x_k)} \cdot \frac{p_\theta^{\text{EM}}(x_{k-1} \mid x_k)}{\tilde{p}_\theta(x_{k-1} \mid x_k, y)}$$

**Pseudo-Bootstrap (pBS) weighting** (GEM-pBS, SOSaG-pBS):
$$G_{k-1}(x_k, x_{k-1}) = \frac{\tilde{p}_\theta(y \mid x_{k-1})}{\tilde{p}_\theta(y \mid x_k)}$$

The pBS approach drops the ratio of proposal densities, trading asymptotic exactness for empirical performance. The resulting target is $\bar{\nu}_0(x_0) = \bar{p}_\theta(x_0 \mid y) \cdot \tilde{p}_\theta(y \mid x_0)^\rho$, where $\rho$ is a tempering parameter.

#### Key Insight: SMC as Evolutionary Algorithm
The authors argue that the empirical success of SMC in this context is *not* due to high effective sample size (which degenerates as $k \to 0$), but rather due to its interpretation as a *multiple-try* evolutionary algorithm:
- **Mutation** = proposal step (stochastic perturbation)
- **Fitness** = weighting by PDE/observation likelihood
- **Selection** = resampling

Tempering $\rho$ interpolates between MAP ($\rho$ small) and MLE ($\rho \to \infty$).

#### Experimental Results
- **Benchmark PDEs** (Darcy, Poisson, Helmholtz, Navier-Stokes bounded/unbounded): SOSaG-pBS achieves the lowest relative $L_2$ error on most tasks (e.g., Darcy forward: 3.96% vs DiffPDE 5.58%).
- **Multiphysics PDEs** (2-species Gray-Scott, 3-species Reaction-Diffusion): SMC methods consistently outperform ODE-based DiffPDE, especially under observation noise ($\sigma_O$ up to 0.02). SOSaG-pBS is most robust to noise.
- **Tempering**: Higher $\rho$ generally improves results (e.g., Darcy forward: 6.19% at $\rho=5$ to 3.19% at $\rho=500$), but tuning is problem-specific.

#### Relationship to This Repository
- Uses DiffusionPDE pretrained models (from Huang et al.)
- Adapts and extends the `scripts/generate_*.py` pipeline to add SMC particle proposals
- The pBS weighting simplifies to a form that can be implemented as a modification of the existing guidance loop

---

## 5. Ancillary Papers (Broader Context)

### 5.1 DDS — Denoising Diffusion Samplers
- **arXiv**: 2302.13834
- **Authors**: Francisco Vargas, Will Grathwohl, Arnaud Doucet
- **Venue**: ICLR 2023
- **Key Idea**: Uses denoising diffusion for Monte Carlo sampling from unnormalized densities (as opposed to generative modeling). Minimizes KL$(q^\theta \| p)$ where $p$ is the path measure of the forward diffusion and $q^\theta$ is the approximate time-reversal.
- **Relevance**: Provides the variational framework that underpins later SMC+diffusion methods.

### 5.2 Constrained Bayesian Filters
- **arXiv**: 2512.11012
- **Authors**: (SIAM journal paper)
- **Key Idea**: Studies Bayesian filtering under state-space constraints (compact subsets). Proves stability and approximation error rates for constrained filters. Proposes barrier-function modification of SDE drift for continuous-discrete filtering.
- **Relevance**: Provides theory for constrained SDE filtering, which relates to the PDE-constrained sampling setting.

### 5.3 Particle Filters for Low and Degenerate Observation Noise
- **arXiv**: 2601.08411
- **Key Idea**: Develops particle filters that remain robust when observation noise is low or degenerate (the likelihood is very informative). Proposes parameterizations that avoid weight collapse.
- **Relevance**: The observation noise regime in PDE-constrained problems is often effectively degenerate (deterministic PDE residuals), making this work relevant.

---

## 6. Methodology Comparison Matrix

| Method | Domain | SMC? | Exact Posterior? | PDE-Capable? | Key Limitation |
|--------|--------|------|-----------------|--------------|----------------|
| DDPM | Generative | No | N/A | No (unconditional) | Unconditional only |
| Score SDE | Generative | No | N/A | No (unconditional) | Unconditional only |
| DPS | Inverse problems | No | No (heuristic) | Via guidance | No consistency guarantee |
| FPS | Linear inverse | No (single particle) | Yes (linear Gaussian $A$) | No (linear $A$ only) | Linear $A$ only |
| FPS-SMC | Linear inverse | Yes | Yes (as $M \to \infty$) | No (linear $A$ only) | Linear $A$ only |
| TDS | General conditioning | Yes | Yes (as $K \to \infty$) | Via likelihood | Likelihood must be differentiable |
| MCGDiff | Linear inverse | Yes | Yes (as $N \to \infty$) | No | Linear $A$ only |
| PDDS | Unnormalized densities | Yes | Yes (as $N \to \infty$) | Via guidance | Designed for densities, not generative priors |
| **Millard et al. (SOSaG-pBS)** | **PDE-constrained** | **Yes** | **No (pBS trades exactness)** | **Yes** | **pBS alters target; tempering $\rho$ needs tuning** |

---

## 7. Key Open Questions for This Repo

1. **pBS vs TDS weighting**: Millard et al. show pBS outperforms TDS for PDEs, but at the cost of altering the target distribution. Can we characterize when this trade-off is beneficial?

2. **Tempering parameter $\rho$**: The optimal $\rho$ is problem-specific and varies across PDEs. Is there a principled way to select it (e.g., via cross-validation or marginal likelihood estimation)?

3. **Particle collapse**: ESS degenerates late in the diffusion. The evolutionary algorithm interpretation suggests this may not matter, but can we prove bounds?

4. **Score model vectorization**: Millard et al. note the score model is called sequentially per particle. Vectorizing this (batch inference) would give significant speedup.

5. **Extending to more PDEs**: The framework has been demonstrated on Darcy, Poisson, Helmholtz, Navier-Stokes, and Reaction-Diffusion. The PDE residual likelihood is plug-and-play.

6. **Beyond Heun's method**: The SOSaG proposal uses EDM's 2nd-order ODE integrator. Higher-order or adaptive integrators could improve accuracy.
