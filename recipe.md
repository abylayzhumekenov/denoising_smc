# Implementation Recipe: Girsanov-Corrected SMC for Diffusion-Guided PDE Solvers

## 1. General Setting

We implement a modular SMC framework for diffusion-guided PDE solvers. Three proposals and weight updates are composed from interchangeable components:

### The $\lambda$-$\rho$ Weight Parameterization

All weight schedules in this family share a single formula:

$$
\log w_{k-1}^{\text{inc}} = \rho \cdot \bigl[\log\tilde{p}(y\mid x_{k-1}) - \log\tilde{p}(y\mid x_k)\bigr] \;+\; \lambda \cdot C_k,
$$

where $C_k$ is the Girsanov correction (or, equivalently for GEM, the closed-form density ratio). Special cases:

| Method | $\lambda$ | $\rho$ | Implementation |
|--------|:---------:|:------:|----------------|
| pBS | 0 | 1 | Likelihood ratio only |
| TDS / Girsanov | 1 | 1 | Full correction |
| Tempered pBS | 0 | $\rho$ | Tempered likelihood |

### What We Implement

| Module | Files |
|--------|-------|
| Noise schedule | `smc/schedule.py` |
| Proposals (GEM, Heun-SDE, SOSaG) | `smc/proposals.py` |
| Unified weight ($\lambda$-$\rho$) | `smc/weights.py` |
| SMC loop | `smc/core.py` |
| PDE runners | `scripts/generate_burgers_smc.py`, `generate_darcy_smc.py` |

---

## 2. Architecture

```
smc/
├── schedule.py     SigmaSchedule        noise schedule + derived quantities
├── proposals.py    Proposal (ABC)       GEM, HeunSDE, SOSaG
├── weights.py      UnifiedWeight        λ-ρ weight with Girsanov correction
└── core.py         ParticleFilter       propagate → weight → resample loop

scripts/
├── generate_burgers_smc.py    Burgers runner (first, simplest)
└── generate_darcy_smc.py      Darcy runner (standard benchmark)
```

### Data Flow

```
ParticleFilter
  │
  ├─ schedule.SigmaSchedule  →  sigma_k, sigma_km1, delta_sigma2, noise_coeff
  │
  ├─ proposal.propagate(x_k, sigma_k, sigma_km1, denoiser, grad_loglik_fn)
  │     │
  │     └─ returns x_km1, aux dict  (aux contains grad_loglik, mu_guide, z, ...)
  │
  ├─ weight.compute_log_inc(x_k, x_km1, sigma_k, sigma_km1, aux_k, ..., log_likelihood_fn)
  │     │
  │     └─ returns log_w_inc  (scalar per particle)
  │
  └─ resample if ESS < threshold
```

---

## 3. Pseudocode

### 3.1 GEM Proposal

```
function GEM.propagate(x_k, sigma_k, sigma_km1, denoiser, grad_loglik_fn):
    D_theta ← denoiser(x_k, sigma_k)
    score ← (D_theta - x_k) / sigma_k²                      # standard convention
    z ← random_normal()
    noise_coeff ← sqrt(sigma_k² - sigma_km1²)
    delta_sigma2 ← sigma_k² - sigma_km1²                     # > 0
    
    grad_loglik ← grad_loglik_fn(x_k)
    
    mu_prior ← x_k + delta_sigma2 * score
    mu_guide ← mu_prior + delta_sigma2 * grad_loglik
    x_km1 ← x_k + delta_sigma2 * (score + grad_loglik) + noise_coeff * z
    
    aux ← {mu_prior, mu_guide, grad_loglik, z, noise_coeff}
    return x_km1, aux
```

### 3.2 Heun-SDE Proposal

```
function HeunSDE.propagate(x_k, sigma_k, sigma_km1, denoiser, grad_loglik_fn):
    z ← random_normal()
    noise_coeff ← sqrt(sigma_k² - sigma_km1²)
    delta_sigma2 ← sigma_k² - sigma_km1²                     # > 0
    
    # Stage 1: Euler prediction
    s_prior_k ← (denoiser(x_k, sigma_k) - x_k) / sigma_k²
    s_guide_k ← grad_loglik_fn(x_k)
    x_pred ← x_k + delta_sigma2 * (s_prior_k + s_guide_k) + noise_coeff * z
    
    # Stage 2: Heun correction
    s_prior_pred ← (denoiser(x_pred, sigma_km1) - x_pred) / sigma_km1²
    s_guide_pred ← grad_loglik_fn(x_pred)
    x_km1 ← x_k + 0.5 * delta_sigma2 * (s_prior_k + s_guide_k + s_prior_pred + s_guide_pred)
              + noise_coeff * z
    
    # For Girsanov: the Brownian increment is g*dW = noise_coeff * z (same as GEM)
    aux ← {z, noise_coeff, grad_loglik_k: s_guide_k,
           grad_loglik_pred: s_guide_pred, x_pred,
           mu_guide: x_k + delta_sigma2*(s_prior_k + s_guide_k)}  # approximate
    return x_km1, aux
```

### 3.3 SOSaG Proposal (Baselines Only)

```
function SOSaG.propagate(x_k, sigma_k, sigma_km1, denoiser, grad_loglik_fn, gamma):
    # 1. Jittering
    hat_sigma ← sigma_k + gamma * sigma_k
    psi ← random_normal()
    hat_x ← x_k + sqrt(hat_sigma² - sigma_k²) * psi
    
    # 2. ODE denoising (Heun)
    d_k ← (hat_x - denoiser(hat_x, hat_sigma)) / hat_sigma
    x_pred ← hat_x + (sigma_km1 - hat_sigma) * d_k
    d_km1 ← (x_pred - denoiser(x_pred, sigma_km1)) / sigma_km1
    x_denoised ← hat_x + (sigma_km1 - hat_sigma) * 0.5 * (d_k + d_km1)
    
    # 3. Guidance
    grad_loglik ← grad_loglik_fn(hat_x)
    x_km1 ← x_denoised + (sigma_k² - sigma_km1²) * grad_loglik
    
    aux ← {psi, hat_x, hat_sigma, x_denoised, grad_loglik}
    return x_km1, aux
```

### 3.4 Unified Weight ($\lambda$-$\rho$)

```
function UnifiedWeight.compute_log_inc(x_k, x_km1, sigma_k, sigma_km1,
                                        aux_k, log_likelihood_fn):
    # Likelihood ratio (tempered by ρ)
    log_lik_km1 ← log_likelihood_fn(x_km1)
    log_lik_k   ← log_likelihood_fn(x_k)
    Δ_loglik ← log_lik_km1 - log_lik_k
    
    if λ == 0:
        return ρ * Δ_loglik
    
    # Girsanov correction
    delta_sigma2 ← sigma_k² - sigma_km1²          # > 0
    grad_loglik ← aux_k.grad_loglik
    
    # Quadratic variation
    quad_var ← 0.5 * delta_sigma2 * ||grad_loglik||²
    
    # Itô integral (g * ΔW recovered from aux)
    # For GEM / Heun-SDE: g*ΔW = noise_coeff * z = x_km1 - mu_guide
    g_dW ← aux_k.noise_coeff * aux_k.z             # exact for GEM and Heun-SDE
    ito ← Σ(grad_loglik * g_dW)
    
    C_k ← ito - quad_var
    return ρ * Δ_loglik + λ * C_k
```

### 3.5 ParticleFilter

```
function ParticleFilter.run(x_init, denoiser, log_likelihood_fn, grad_loglik_fn):
    x_k ← x_init
    log_w ← zeros(N)
    
    for step = 0, ..., K-1:
        sigma_k   ← schedule[step]
        sigma_km1 ← schedule[step+1]
        
        # Propagate
        x_km1, aux_k ← proposal.propagate(x_k, sigma_k, sigma_km1, denoiser, grad_loglik_fn)
        
        # Weight
        log_w_inc ← weight.compute_log_inc(x_k, x_km1, sigma_k, sigma_km1, aux_k, log_likelihood_fn)
        log_w ← log_w + log_w_inc
        log_w ← log_w - logsumexp(log_w)           # normalize
        
        # Resample (adaptive)
        ess ← 1 / Σ exp(2 * log_w)
        if ess < threshold * N:
            idx ← multinomial(exp(log_w))
            x_km1 ← x_km1[idx]
            log_w ← zeros(N)
        
        x_k ← x_km1
    
    return x_k, log_w
```

---

## 4. Python Implementations

### 4.1 `smc/schedule.py`

```python
import torch
from torch_utils.misc import auto_device

class SigmaSchedule:
    """Power-law sigma schedule for EDM-style diffusion.
    
    Key identity: g(t)^2 * |dt| = 2 * sigma_dot * sigma * |dt| = sigma_k^2 - sigma_{k-1}^2.
    """
    def __init__(self, sigma_min, sigma_max, num_steps, rho=7.0, device=None):
        if device is None:
            device = auto_device()
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_steps = num_steps
        self.rho = rho
        
        step_idx = torch.arange(num_steps, dtype=torch.float64, device=device)
        sigma_steps = (sigma_max ** (1/rho) + step_idx / (num_steps - 1)
                       * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        self.sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])
    
    def sigma_k(self, k):
        return self.sigma_steps[k]
    
    def delta_sigma2(self, k):
        """sigma_k^2 - sigma_{k-1}^2 > 0."""
        return self.sigma_steps[k]**2 - self.sigma_steps[k+1]**2
    
    def noise_coeff(self, k):
        """sqrt(sigma_k^2 - sigma_{k-1}^2) — the EM noise coefficient."""
        return torch.sqrt(self.delta_sigma2(k))
    
    def quad_var_coeff(self, k):
        """0.5 * (sigma_k^2 - sigma_{k-1}^2) — Girsanov quadratic variation coefficient."""
        return 0.5 * self.delta_sigma2(k)
```

### 4.2 `smc/proposals.py`

```python
import torch
from abc import ABC, abstractmethod

class Proposal(ABC):
    @abstractmethod
    def propagate(self, x_k, sigma_k, sigma_km1, denoiser, grad_loglik_fn, schedule, key):
        pass

class GEM(Proposal):
    """First-order Euler-Maruyama with guidance (standard score convention)."""
    def propagate(self, x_k, sigma_k, sigma_km1, denoiser, grad_loglik_fn, schedule, key):
        D_theta = denoiser(x_k, sigma_k)
        score = (D_theta - x_k) / (sigma_k ** 2)
        
        z = torch.randn_like(x_k, generator=key)
        noise_coeff = torch.sqrt(sigma_k ** 2 - sigma_km1 ** 2)
        delta_sigma2 = sigma_k ** 2 - sigma_km1 ** 2  # > 0
        
        grad_loglik = grad_loglik_fn(x_k)
        mu_prior = x_k + delta_sigma2 * score
        mu_guide = mu_prior + delta_sigma2 * grad_loglik
        
        x_km1 = x_k + delta_sigma2 * (score + grad_loglik) + noise_coeff * z
        
        aux = {
            'mu_prior': mu_prior.detach(),
            'mu_guide': mu_guide.detach(),
            'grad_loglik': grad_loglik,
            'noise_coeff': noise_coeff,
            'z': z,
        }
        return x_km1, aux


class HeunSDE(Proposal):
    """Second-order stochastic Heun method (additive noise SDE integrator).
    
    Two stages, same noise z throughout. Exact Girsanov Brownian increment recovery.
    """
    def propagate(self, x_k, sigma_k, sigma_km1, denoiser, grad_loglik_fn, schedule, key):
        z = torch.randn_like(x_k, generator=key)
        noise_coeff = torch.sqrt(sigma_k ** 2 - sigma_km1 ** 2)
        delta_sigma2 = sigma_k ** 2 - sigma_km1 ** 2  # > 0
        
        # Stage 1: Euler prediction
        s_prior_k = (denoiser(x_k, sigma_k) - x_k) / (sigma_k ** 2)
        s_guide_k = grad_loglik_fn(x_k)
        x_pred = x_k + delta_sigma2 * (s_prior_k + s_guide_k) + noise_coeff * z
        
        # Stage 2: Heun correction
        s_prior_pred = (denoiser(x_pred, sigma_km1) - x_pred) / (sigma_km1 ** 2)
        s_guide_pred = grad_loglik_fn(x_pred)
        
        x_km1 = (x_k
                 + 0.5 * delta_sigma2 * (s_prior_k + s_guide_k + s_prior_pred + s_guide_pred)
                 + noise_coeff * z)
        
        aux = {
            'z': z,
            'noise_coeff': noise_coeff,
            'grad_loglik': s_guide_k,       # reference point = x_k (constant interpolation)
            'grad_loglik_pred': s_guide_pred,
            'x_pred': x_pred.detach(),
            'mu_guide': (x_k + delta_sigma2 * (s_prior_k + s_guide_k)).detach(),
        }
        return x_km1, aux


class SOSaG(Proposal):
    """SOSaG proposal (Millard et al.) for baseline comparison.
    
    Jittering + Heun ODE + guidance correction. Not a clean SDE discretization.
    """
    def __init__(self, gamma=0.2):
        self.gamma = gamma
    
    def propagate(self, x_k, sigma_k, sigma_km1, denoiser, grad_loglik_fn, schedule, key):
        hat_sigma = sigma_k + self.gamma * sigma_k
        psi = torch.randn_like(x_k, generator=key)
        hat_x = x_k + torch.sqrt(hat_sigma ** 2 - sigma_k ** 2) * psi
        
        d_k = (hat_x - denoiser(hat_x, hat_sigma)) / hat_sigma
        x_pred = hat_x + (sigma_km1 - hat_sigma) * d_k
        
        if sigma_km1 > 0:
            d_km1 = (x_pred - denoiser(x_pred, sigma_km1)) / sigma_km1
            x_denoised = hat_x + (sigma_km1 - hat_sigma) * 0.5 * (d_k + d_km1)
        else:
            x_denoised = x_pred
        
        delta_sigma2 = sigma_k ** 2 - sigma_km1 ** 2  # > 0, uses original sigma
        grad_loglik = grad_loglik_fn(hat_x)
        x_km1 = x_denoised + delta_sigma2 * grad_loglik
        
        aux = {'psi': psi, 'hat_x': hat_x, 'hat_sigma': hat_sigma,
               'x_denoised': x_denoised, 'grad_loglik': grad_loglik}
        return x_km1, aux
```

### 4.3 `smc/weights.py`

```python
import torch
from abc import ABC, abstractmethod

class WeightUpdate(ABC):
    @abstractmethod
    def compute_log_inc(self, x_k, x_km1, sigma_k, sigma_km1,
                        aux_k, aux_km1, log_likelihood_fn, schedule):
        pass


class UnifiedWeight(WeightUpdate):
    """λ-ρ unified weight with Girsanov correction.
    
    For GEM: λ=1 exactly recovers TDS (verified in idea.md §4).
    For Heun-SDE: λ=1 gives asymptotically consistent correction.
    For λ=0: reduces to pBS.
    """
    def __init__(self, lam=1.0, rho=1.0):
        self.lam = lam
        self.rho = rho
    
    def compute_log_inc(self, x_k, x_km1, sigma_k, sigma_km1,
                        aux_k, aux_km1, log_likelihood_fn, schedule):
        # Likelihood ratio
        log_lik_km1 = log_likelihood_fn(x_km1)
        log_lik_k = log_likelihood_fn(x_k)
        log_lik_ratio = log_lik_km1 - log_lik_k
        
        if self.lam == 0.0:
            return self.rho * log_lik_ratio
        
        # Girsanov correction C_k
        delta_sigma2 = sigma_k ** 2 - sigma_km1 ** 2  # > 0
        grad = aux_k['grad_loglik']                    # [N, C, H, W]
        
        # Quadratic variation: 0.5 * Δσ² * ||∇log p̃||²  (per particle)
        quad_var = 0.5 * delta_sigma2 * torch.sum(grad ** 2, dim=tuple(range(1, grad.dim())))
        
        # Itô integral: g * ∇log p̃^T ΔW
        # Recover g*ΔW from noise (works for GEM and Heun-SDE)
        g_dW = aux_k['noise_coeff'] * aux_k['z']        # [N, C, H, W]
        ito = torch.sum(grad * g_dW, dim=tuple(range(1, grad.dim())))
        
        C_k = ito - quad_var  # [N]
        return self.rho * log_lik_ratio + self.lam * C_k
```

### 4.4 `smc/core.py`

```python
import torch
from torch_utils.misc import auto_device

class ParticleFilter:
    def __init__(self, proposal, weight_update, schedule, num_particles,
                 resample_threshold=0.5, device=None):
        self.proposal = proposal
        self.weight_update = weight_update
        self.schedule = schedule
        self.N = num_particles
        self.resample_threshold = resample_threshold
        self.device = device if device is not None else auto_device()
    
    def run(self, x_init, denoiser, log_likelihood_fn, grad_loglik_fn):
        x_k = x_init.to(self.device)
        log_w = torch.zeros(self.N, device=self.device)
        diagnostics = {'ess': [], 'resample_steps': []}
        
        for step in range(self.schedule.num_steps):
            sigma_k = self.schedule.sigma_k(step)
            sigma_km1 = self.schedule.sigma_k(step + 1)
            key = torch.Generator(device=self.device).manual_seed(step)
            
            x_km1, aux_k = self.proposal.propagate(
                x_k, sigma_k, sigma_km1, denoiser, grad_loglik_fn, self.schedule, key)
            
            log_w_inc = self.weight_update.compute_log_inc(
                x_k, x_km1, sigma_k, sigma_km1, aux_k, None,
                log_likelihood_fn, self.schedule)
            
            log_w = log_w + log_w_inc
            log_w = log_w - torch.logsumexp(log_w, dim=0)
            
            ess = 1.0 / torch.sum(torch.exp(2 * log_w), dim=0)
            diagnostics['ess'].append(ess.item())
            
            if ess < self.resample_threshold * self.N:
                idx = torch.multinomial(torch.exp(log_w), self.N, replacement=True)
                x_km1 = x_km1[idx]
                log_w = torch.zeros(self.N, device=self.device)
                diagnostics['resample_steps'].append(step)
            
            x_k = x_km1
        
        return x_k, log_w, diagnostics
```

---

## 5. PDE Runners

### 5.1 Burgers Runner (`scripts/generate_burgers_smc.py`)

```python
# Pseudocode structure:

def generate_burgers_smc(config):
    # 1. Load data and network (same as existing generate_burgers.py)
    # 2. Build SigmaSchedule
    sched = SigmaSchedule(
        sigma_min=config['generate']['sigma_min'],
        sigma_max=config['generate']['sigma_max'],
        num_steps=config['test']['iterations'],
        rho=config['generate']['rho'])
    
    # 3. Select proposal
    if config['generate']['proposal'] == 'gem':
        proposal = GEM()
    elif config['generate']['proposal'] == 'heun_sde':
        proposal = HeunSDE()
    elif config['generate']['proposal'] == 'sosag':
        proposal = SOSaG(gamma=config['generate'].get('gamma', 0.2))
    
    # 4. Select weight
    weight = UnifiedWeight(
        lam=config['generate'].get('lam', 1.0),
        rho=config['generate'].get('rho', 1.0))
    
    # 5. Create particle filter
    pf = ParticleFilter(proposal, weight, sched,
                        num_particles=config['generate']['batch_size'],
                        resample_threshold=config['generate'].get('resample_threshold', 0.5))
    
    # 6. Initialize (N particles, sigma_max noise)
    x_init = torch.randn([N, C, H, W], dtype=torch.float64, device=device) * sigma_max
    
    # 7. Define likelihood and gradient functions (Burgers-specific)
    #    These wrap the PDE residual + observation loss computation
    
    def log_likelihood_fn(x):
        a_phys = inverse_scale(x)   # from (-1,1) to physical space
        pde_residual = burgers_pde_residual(a_phys)
        obs_error = (a_phys - observations) * mask
        return -(beta * torch.norm(obs_error)**2 + omega * torch.norm(pde_residual)**2)
    
    def grad_loglik_fn(x):
        x = x.detach().clone().requires_grad_(True)
        L = log_likelihood_fn(x)
        return torch.autograd.grad(L.sum(), x)[0]
    
    # 8. Run
    x_final, log_w, diag = pf.run(x_init, denoiser, log_likelihood_fn, grad_loglik_fn)
    
    # 9. Save results and diagnostics
```

### 5.2 Darcy Runner

Same structure as Burgers runner, with:
- 2D $128\times128$ grid, 2 channels ($a$, $u$)
- PDE residual: $-\nabla\cdot(a\nabla u) - 1$ (finite-difference stencils)
- Scaling: $a = (a+1.5)/0.2$, $u = (u+0.9)/115$
- 500 random sensors per field

---

## 6. Practical Considerations and Gotchas

### Gradient Flow

The `grad_loglik_fn` computes $\nabla_x\log\tilde{p}(y|x)$ by backprop through the denoiser. This requires:

```python
def grad_loglik_fn(x):
    x_in = x.detach().clone().requires_grad_(True)
    L = log_likelihood_fn(x_in)          # computes denoising + PDE loss
    return torch.autograd.grad(L.sum(), x_in)[0]
```

The `detach().clone()` ensures we don't backprop through the entire SMC history.

### Tensor Shapes

- `x_k`: `[N, C, H, W]` — all operations are vectorized over the particle dimension
- `log_w`: `[N]` — scalar weight per particle
- Girsanov `quad_var` and `ito`: `[N]` — per-particle corrections
- Use `torch.sum(x, dim=tuple(range(1, x.dim())))` to reduce spatial/channel dims per particle

### Numerical Stability

- **Log-weights**: Always maintain in log-space. Use `log_w - logsumexp(log_w)` for normalization.
- **ESS computation**: `ess = 1.0 / sum(exp(2 * log_w))` — stable in log-space.
- **After resampling**: Reset `log_w = torch.zeros(N)` (uniform weights in log-space).
- **Float precision**: All tensors in `float64`.

### Step Indexing

The loop index `step` goes from `0` to `num_steps - 1`:
- `sigma_k = schedule.sigma_steps[step]`
- `sigma_km1 = schedule.sigma_steps[step + 1]`
- `schedule.delta_sigma2(step)` uses the same index

### Random Seed Management

Use a dedicated `torch.Generator` per step:

```python
key = torch.Generator(device=device).manual_seed(step + base_seed)
```

Do NOT rely on global `torch.manual_seed` inside the loop.

### Heun-SDE: Which `grad_loglik` Goes into `aux`?

The Girsanov correction with **constant interpolation** uses $\nabla\log\tilde{p}$ evaluated at $x_k$ (the interval start). Store `s_guide_k` in `aux['grad_loglik']`.

For **linear interpolation**, you would need both `s_guide_k` and `s_guide_pred`. This is implemented by reading from `aux['grad_loglik_pred']` if available.

### Sigma Schedule Edge Cases

- **Last step** (sigma=0): `sigma_km1 = 0`. The `delta_sigma2` is positive. The denoiser at sigma=0 should be close to the identity (no denoising).
- **Girsanov at very small sigma**: As $\sigma \to 0$, $\nabla\log\tilde{p}$ may become large because the likelihood landscape sharpens. The quadratic variation term $0.5\Delta\sigma^2\|\nabla\log\tilde{p}\|^2$ should compensate, but monitor for numerical issues.

### Verifying GEM-Girsanov == GEM-TDS

After implementing Steps 1-4, run:

```python
# Burgers, N=4, K=100
# Compare per-step log_w_inc from:
#   1. UnifiedWeight(lam=1.0) on GEM  → uses Girsanov correction
#   2. Closed-form TDS formula        → Gaussian density ratio
# They should match to numerical precision (idea.md §4).
```

If they differ, check:
- Sign of `delta_sigma2` in propagation
- Sign of `quad_var` (should be subtracted)
- Brownian increment recovery: `g_dW = noise_coeff * z = x_km1 - mu_guide`

### SOSaG for Baseline Reproduction

When reproducing Millard et al. results:
- Use `gamma=0.2` (EDM default)
- Use `UnifiedWeight(lam=0.0, rho=1.0)` for pBS
- The `grad_loglik_fn` should follow the paper's convention: evaluate at `hat_x`, guidance coefficient uses original `sigma_k` (not `hat_sigma`)

### Config-Driven Experimentation

Use YAML configs to specify proposal and weight type:

```yaml
# configs/experiment.yaml
generate:
  proposal: heun_sde        # gem | heun_sde | sosag
  lam: 1.0
  rho: 1.0
  batch_size: 4
  iterations: 2000
  resample_threshold: 0.5
```

The runner reads `config['generate']['proposal']` to instantiate the appropriate class.

---

## Implementation Checklist

- [ ] `smc/schedule.py`: `SigmaSchedule` class
- [ ] `smc/proposals.py`: `GEM`, `HeunSDE`, `SOSaG` classes
- [ ] `smc/weights.py`: `UnifiedWeight(lam, rho)` class
- [ ] `smc/core.py`: `ParticleFilter` class
- [ ] `scripts/generate_burgers_smc.py`: Burgers runner
- [ ] `scripts/generate_darcy_smc.py`: Darcy runner
- [ ] Verification: GEM($\lambda{=}1$) == GEM-TDS numerically
- [ ] Verification: SOSaG-pBS reproduces published Darcy results
- [ ] Diagnostics: ESS trajectory, weight variance, resample times
