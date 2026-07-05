# Summary of Methodology and Setting for Filtering Posterior Sampling (FPS)

## Problem Setting

### Linear Inverse Problem

We aim to recover a signal $\mathbf{x} \in \mathbb{R}^D$ from noisy linear measurements:

$$\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{n}, \quad \mathbf{n} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$$

where:
- $\mathbf{A} \in \mathbb{R}^{d \times D}$ is the measurement matrix ($d < D$, ill-posed)
- $\mathbf{y} \in \mathbb{R}^d$ is the observed measurement
- $\sigma > 0$ is the known noise level

The goal is to sample from the posterior distribution:

$$p(\mathbf{x} \mid \mathbf{y}) \propto p(\mathbf{x}) \cdot p(\mathbf{y} \mid \mathbf{x}) = p(\mathbf{x}) \cdot \mathcal{N}(\mathbf{y} \mid \mathbf{A}\mathbf{x}, \sigma^2 \mathbf{I})$$

where $p(\mathbf{x})$ is the data prior learned by a diffusion model.

---

## Diffusion Model Background

### Forward Process (Discrete)

The forward noising process is a Markov chain:

$$q(\mathbf{x}_{1:N} \mid \mathbf{x}_0) = \prod_{k=1}^N q(\mathbf{x}_k \mid \mathbf{x}_{k-1})$$

with transition:

$$q(\mathbf{x}_k \mid \mathbf{x}_{k-1}) = \mathcal{N}(a_k \mathbf{x}_{k-1}, b_k^2 \mathbf{I})$$

The marginal distribution is Gaussian:

$$q(\mathbf{x}_k \mid \mathbf{x}_0) = \mathcal{N}(c_k \mathbf{x}_0, d_k^2 \mathbf{I})$$

where $c_k$ and $d_k$ are derived from $\{a_k\}$ and $\{b_k\}$.

### Backward Process

The learned reverse process is parameterized as:

$$p_{\theta}(\mathbf{x}_{k-1} \mid \mathbf{x}_k) = \mathcal{N}\left(u_k \tilde{\mathbf{x}}_0(\mathbf{x}_k) + v_k s_{\theta}(\mathbf{x}_k, t_k), w_k^2 \mathbf{I}\right)$$

where:
- $s_{\theta}(\mathbf{x}_k, t_k)$ is the learned score estimator
- $\tilde{\mathbf{x}}_0(\mathbf{x}_k) = \frac{\mathbf{x}_k + d_k^2 s_{\theta}(\mathbf{x}_k, t_k)}{c_k}$ is the Tweedie's formula estimate of $\mathbf{x}_0$

### Variance Preserving (VP) Diffusion (DDPM)

For DDPM:
$$a_k = \sqrt{\alpha_k}, \quad b_k = \sqrt{\beta_k}, \quad c_k = \sqrt{\bar{\alpha}_k}, \quad d_k = \sqrt{1 - \bar{\alpha}_k}$$

where $\alpha_k = 1 - \beta_k$ and $\bar{\alpha}_k = \prod_{j=1}^k \alpha_j$.

**DDPM parameters:**
$$u_k = \sqrt{\alpha_{k-1}}, \quad v_k = -\sqrt{\alpha_k}(1 - \bar{\alpha}_{k-1}), \quad w_k = \sqrt{\beta_k} \cdot \frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k}$$

**DDIM parameters:**
$$u_k = \sqrt{\alpha_{k-1}}, \quad v_k = -\sqrt{1 - \bar{\alpha}_{k-1} - \sigma_k^2 \cdot \sqrt{1 - \bar{\alpha}_k}}, \quad w_k = \sigma_k$$

with $\sigma_k^2 = c \cdot \frac{\beta_k(1 - \bar{\alpha}_{k-1})}{1 - \bar{\alpha}_k}$ where $c \in [0,1]$ is a tunable hyperparameter.

---

## Core Methodology: FPS (Filtering Posterior Sampling)

### Key Insight: Equivalence to Bayesian Filtering

Posterior sampling $p_{\theta}(\mathbf{x}_0 \mid \mathbf{y}_0)$ is reformulated as a reverse-time Bayesian filtering problem.

**Duplex Forward Diffusion Process:**

For data:
$$\mathbf{x}_k = \sqrt{\bar{\alpha}_k} \cdot \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_k} \cdot \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$$

For measurements:
$$\mathbf{y}_k = \sqrt{\bar{\alpha}_k} \cdot \mathbf{y}_0 + \sqrt{1 - \bar{\alpha}_k} \cdot \mathbf{A}\mathbf{z}$$

This implies:

$$q(\mathbf{y}_k \mid \mathbf{x}_k) = \mathcal{N}(\mathbf{A}\mathbf{x}_k, \sigma^2 \bar{\alpha}_k \cdot \mathbf{I})$$

### Step 1: Generating the Measurement Sequence $\{\mathbf{y}_k\}_{k=0}^N$

Using DDIM backward process (since $\mathbf{y}_0$ is known):

$$\mathbf{y}_{k-1} = \sqrt{\bar{\alpha}_{k-1}}\mathbf{y}_0 + \sqrt{\frac{(1-c)(1-\bar{\alpha}_{k-1})}{1-\bar{\alpha}_k}}(\mathbf{y}_k - \sqrt{\bar{\alpha}_k}\mathbf{y}_0) + \sqrt{c(1-\bar{\alpha}_{k-1})} \cdot \mathbf{A}\epsilon_k$$

where $\epsilon_k \sim \mathcal{N}(0, \mathbf{I})$.

Initialization:
$$\mathbf{y}_N = \mathbf{A}\epsilon_N, \quad \epsilon_N \sim \mathcal{N}(0, \mathbf{I})$$

### Step 2: Generating the Backward Sequence $\{\mathbf{x}_k\}_{k=0}^N$

For each step, we sample from:

$$p_{\theta}(\mathbf{x}_{k-1} \mid \mathbf{x}_k, \mathbf{y}_{k-1}) \propto p_{\theta}(\mathbf{x}_{k-1} \mid \mathbf{x}_k) \cdot p_{\theta}(\mathbf{y}_{k-1} \mid \mathbf{x}_{k-1})$$

**Explicit Gaussian Formulation (with DDIM):**

$$p_{\theta}(\mathbf{x}_{k-1} \mid \mathbf{x}_k, \mathbf{y}_{k-1}) = \mathcal{N}(\mu_k^{\text{FPS}}(\mathbf{x}_k, \mathbf{y}_{k-1}, \theta), \Sigma_k^{\text{FPS}})$$

where:

$$\Sigma_k^{\text{FPS}} = \left( (\Sigma_k^{\text{DDIM}})^{-1} + \frac{1}{\sigma^2 \cdot \bar{\alpha}_{k-1}} \mathbf{A}^\top \mathbf{A} \right)^{-1}$$

and

$$\mu_k^{\text{FPS}}(\mathbf{x}_k, \mathbf{y}_{k-1}, \theta) = \Sigma_k^{\text{FPS}} \cdot \left( (\Sigma_k^{\text{DDIM}})^{-1} \mu_k^{\text{DDIM}}(\mathbf{x}_k, \theta) + \frac{1}{\sigma^2 \cdot \bar{\alpha}_{k-1}} \mathbf{A}^\top \mathbf{y}_{k-1} \right)$$

**DDIM components:**
$$\mu_k^{\text{DDIM}}(\mathbf{x}_k, \theta) = \frac{1}{\alpha_k}(\mathbf{x}_k + (1-\bar{\alpha}_k)s_{\theta}(\mathbf{x}_k, t_k)) - \sqrt{(1-c)(1-\bar{\alpha}_{k-1})}(1-\bar{\alpha}_k)s_{\theta}(\mathbf{x}_k, t_k)$$

$$\Sigma_k^{\text{DDIM}} = c(1-\bar{\alpha}_{k-1}) \cdot \mathbf{I}$$

---

## FPS-SMC (Sequential Monte Carlo Extension)

For asymptotic consistency, FPS-SMC uses $M$ particles with resampling.

### Algorithm

1. Generate $\{\mathbf{y}_k\}_{k=0}^N$ sequence
2. Initialize $M$ i.i.d. particles:
   $$\mathbf{x}_N^{(j)} \sim p_{\theta}(\mathbf{x}_N \mid \mathbf{y}_N), \quad j \in [M]$$

3. For $k = N, N-1, \dots, 1$:
   - Sample:
     $$\bar{\mathbf{x}}_{k-1}^{(j)} \sim p_{\theta}(\mathbf{x}_{k-1} \mid \mathbf{x}_k^{(j)}, \mathbf{y}_{k-1}), \quad j \in [M]$$
   
   - Resample with weights:
     $$\eta_j = \frac{p_{\theta}(\mathbf{y}_{k-1} \mid \bar{\mathbf{x}}_{k-1}^{(j)})}{\sum_{i=1}^M p_{\theta}(\mathbf{y}_{k-1} \mid \bar{\mathbf{x}}_{k-1}^{(i)})}$$

4. Output: Uniformly sample $\mathbf{x}_0$ from $\{\mathbf{x}_0^{(j)}\}_{j=1}^M$

### Theoretical Consistency

**Proposition:** Under perfect score estimation and zero discretization error:

$$p_{\theta}(\mathbf{x}_0 \mid \mathbf{y}_0) \xrightarrow{w} p^*(\mathbf{x}_0 \mid \mathbf{y}_0) = q(\mathbf{x}_0 \mid \mathbf{y}_0) \quad \text{as } M \to \infty$$

where $\xrightarrow{w}$ denotes weak convergence.

---

## Continuous-Time Limit

As $\Delta t \to 0$, FPS approximates the SDE:

$$\mathrm{d}\mathbf{x}_t = \left[-\frac{\beta(t)}{2}\mathbf{x}_t - \beta(t)\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{y}_t)\right] \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}\bar{\mathbf{W}}_t$$

The approximation replaces $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{y})$ with $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{y}_t)$, where $\mathbf{y}_t$ follows the backward diffusion process rather than being a fixed noisy measurement.

---

## Experimental Setting

### Datasets
- **FFHQ-1k**: $256 \times 256 \times 3$ face images, normalized to $[0,1]$
- **ImageNet-1k**: $256 \times 256 \times 3$ natural images

### Linear Inverse Problems Tested

1. **Super Resolution**: Bicubic downsampling $256 \times 256 \to 64 \times 64$ (factor 4)
2. **Inpainting (Box)**: Random $128 \times 128$ masked region
3. **Inpainting (Random)**: $92\%$ pixels masked randomly
4. **Gaussian Deblurring**: $61 \times 61$ kernel, intensity 3.0
5. **Motion Deblurring**: $61 \times 61$ kernel, intensity 0.5

### Measurement Noise
$$\mathbf{n} \sim \mathcal{N}(0, \sigma^2 \mathbf{I}), \quad \sigma = 0.05$$

### Hyperparameters
- Number of diffusion steps: $N = 1000$
- Particle size: $M = 20$ (default)
- DDIM noise parameter $c$: Task-specific (see Table 8 in paper)
  - FFHQ: 0.3 (SR), 0.95 (box inpainting), 0.3 (deblur), 0.95 (random inpainting), 0.3 (motion)
  - ImageNet: 0.15 (SR), 0.25 (box), 0.3 (deblur), 0.25 (random), 0.3 (motion)

### Evaluation Metrics
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- FID (Fréchet Inception Distance)
- LPIPS (Learned Perceptual Image Patch Similarity)

### Baselines
- DPS (Diffusion Posterior Sampling)
- DDRM (Denoising Diffusion Restoration Models)
- MCG (Monte Carlo Guided diffusion)
- PnP-ADMM (Plug-and-Play ADMM)
- Score-SDE
- ADMM-TV

### Computational Resources
- Single A100 GPU
- Running time: ~33 seconds for FPS (box inpainting)
- FPS-SMC running time scales as $t(M) \propto \sqrt{M}$

---

## Key Equations Summary

**Measurement likelihood:**
$$p(\mathbf{y} \mid \mathbf{x}) = \mathcal{N}(\mathbf{y} \mid \mathbf{A}\mathbf{x}, \sigma^2 \mathbf{I})$$

**Posterior sampling via filtering:**
$$p(\mathbf{x}_0 \mid \mathbf{y}_0) = \int p(\mathbf{x}_0 \mid \mathbf{y}_{0:N}) \cdot p(\mathbf{y}_{1:N} \mid \mathbf{y}_0) \mathrm{d}\mathbf{y}_{1:N}$$

**Filtering posterior mean:**
$$\mu_k^{\text{FPS}} = \Sigma_k^{\text{FPS}} \left( (\Sigma_k^{\text{DDIM}})^{-1} \mu_k^{\text{DDIM}} + \frac{1}{\sigma^2 \bar{\alpha}_{k-1}} \mathbf{A}^\top \mathbf{y}_{k-1} \right)$$

**Filtering posterior covariance:**
$$\Sigma_k^{\text{FPS}} = \left( (\Sigma_k^{\text{DDIM}})^{-1} + \frac{1}{\sigma^2 \bar{\alpha}_{k-1}} \mathbf{A}^\top \mathbf{A} \right)^{-1}$$

**Resampling weight (FPS-SMC):**
$$\eta_j = \frac{p_{\theta}(\mathbf{y}_{k-1} \mid \bar{\mathbf{x}}_{k-1}^{(j)})}{\sum_{i=1}^M p_{\theta}(\mathbf{y}_{k-1} \mid \bar{\mathbf{x}}_{k-1}^{(i)})}$$
