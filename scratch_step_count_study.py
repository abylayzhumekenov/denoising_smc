import torch, json
from smc.check_toy_mixture import simulate as simulate_vtau
from smc.check_toy_mixture_millard import simulate_millard
from smc.toy_mixture import GaussianMixture, sigma_schedule

mixture = GaussianMixture(w=[0.5, 0.5], mu=[-3.0, 3.0], var=[1.0, 1.0])
y, r = 0.5, 1.0
n_particles = 50000
n_reps = 8
step_counts = [250, 500, 1000, 2000]
true_posterior = mixture.posterior(y, r)
true_mean = (true_posterior.w * true_posterior.mu).sum().item()
print(f'true mean: {true_mean:.4f}', flush=True)

results = {'true_mean': true_mean, 'vtau': {}, 'millard': {}}

for num_steps in step_counts:
    sigmas = sigma_schedule(sigma_min=0.02, sigma_max=20.0, num_steps=num_steps, rho=7.0)

    means_vtau = []
    for rep in range(n_reps):
        seed = 5000 * num_steps + rep
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        x, log_w = simulate_vtau(mixture, n_particles, sigmas, y, r, guidance='approx',
                                  generator=gen, accumulate_weight=True, ess_threshold=0.5)
        w = torch.softmax(log_w, dim=0)
        means_vtau.append((w * x).sum().item())
    means_vtau_t = torch.tensor(means_vtau)
    rmse_vtau = ((means_vtau_t - true_mean) ** 2).mean().sqrt().item()
    std_vtau = means_vtau_t.std(unbiased=True).item()
    bias_vtau = means_vtau_t.mean().item() - true_mean
    results['vtau'][num_steps] = {'means': means_vtau, 'rmse': rmse_vtau, 'std': std_vtau, 'bias': bias_vtau}
    print(f'[V_tau]    steps={num_steps:5d}  bias={bias_vtau:+.4f}  std={std_vtau:.4f}  rmse={rmse_vtau:.4f}  means={[round(m,3) for m in means_vtau]}', flush=True)

    means_millard = []
    for rep in range(n_reps):
        seed = 6000 * num_steps + rep
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        x, log_w = simulate_millard(mixture, n_particles, sigmas, y, r, gen, lam=1.0, rho=1.0,
                                     accumulate_weight=True, ess_threshold=0.5)
        w = torch.softmax(log_w, dim=0)
        means_millard.append((w * x).sum().item())
    means_millard_t = torch.tensor(means_millard)
    rmse_millard = ((means_millard_t - true_mean) ** 2).mean().sqrt().item()
    std_millard = means_millard_t.std(unbiased=True).item()
    bias_millard = means_millard_t.mean().item() - true_mean
    results['millard'][num_steps] = {'means': means_millard, 'rmse': rmse_millard, 'std': std_millard, 'bias': bias_millard}
    print(f'[Millard]  steps={num_steps:5d}  bias={bias_millard:+.4f}  std={std_millard:.4f}  rmse={rmse_millard:.4f}  means={[round(m,3) for m in means_millard]}', flush=True)

with open('/private/tmp/claude-449869890/-Users-alvarem-Library-CloudStorage-GoogleDrive-miguelangel-alvarezballesteros-kaust-edu-sa-Other-computers-My-MacBook-Pro-MEGA-0KAUST-0Denoising/fe747770-e6ee-493d-aa96-caaf722c2b47/scratchpad/step_count_replication_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('DONE')
