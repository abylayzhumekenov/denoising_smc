"""
Unconditional sampling from the pretrained EDM prior — no observations, no guidance.

This is the pure prior model: it draws samples from p_theta(a, u) as learned during
training, with no conditioning on sparse sensors or PDE residuals. It is exactly the
Euler+Heun (2nd order) ODE sampler used inside scripts/generate_darcy.py, with every
observation/PDE-guidance term (lines using a_GT, u_GT, zeta_*) removed.

No data/*.mat file is needed — only the pretrained network pickle.

Usage:
    python3 sample_prior.py --network pretrained-models/pretrained-darcy.pkl --outdir prior_samples

The output .npy has shape [batch, 2, res, res]: channel 0 is the coefficient field a,
channel 1 is the solution field u, both still in the model's normalized (-1, 1) space.
Unscaling to physical units is PDE-specific (for Darcy: a = (a+1.5)/0.2, u = (u+0.9)/115)
so it's left as a separate step below rather than baked in.
"""

import os
import pickle
import click
import numpy as np
import torch
import tqdm
from torch_utils.misc import auto_device


def edm_heun_sample(net, batch_size, num_steps, sigma_min, sigma_max, rho, device, seed=0):
    """Pure EDM 2nd-order (Heun) ODE sampler. No guidance, no observations."""
    torch.manual_seed(seed)

    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    latents = torch.randn(
        [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
        device=device,
    )
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[
            torch.randint(net.label_dim, size=[batch_size], device=device)
        ]

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    sigma_t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])

    x_next = latents.to(torch.float64) * sigma_t_steps[0]

    for i, (sigma_cur, sigma_next) in tqdm.tqdm(
        list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit="step"
    ):
        x_cur = x_next
        sigma_t = net.round_sigma(sigma_cur)

        # Euler step: D_theta denoises x_cur at noise level sigma_t
        denoised = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur = (x_cur - denoised) / sigma_t
        x_next = x_cur + (sigma_next - sigma_t) * d_cur

        # 2nd order (Heun) correction
        if i < num_steps - 1:
            denoised_next = net(x_next, sigma_next, class_labels=class_labels).to(torch.float64)
            d_prime = (x_next - denoised_next) / sigma_next
            x_next = x_cur + (sigma_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


@click.command()
@click.option("--network", "network_pkl", required=True, help="Path to pretrained .pkl")
@click.option("--outdir", default="prior_samples", help="Where to save samples")
@click.option("--batch-size", default=4, help="Number of samples to draw")
@click.option("--num-steps", default=2000, help="Number of ODE steps")
@click.option("--sigma-min", default=0.002)
@click.option("--sigma-max", default=80.0)
@click.option("--rho", default=7.0)
@click.option("--seed", default=0)
@click.option("--device", default="auto")
def main(network_pkl, outdir, batch_size, num_steps, sigma_min, sigma_max, rho, seed, device):
    dev = auto_device() if device == "auto" else torch.device(device)

    print(f'Loading network from "{network_pkl}"...')
    with open(network_pkl, "rb") as f:
        net = pickle.load(f)["ema"].to(dev)  # EMA weights = the actual prior model

    x = edm_heun_sample(
        net, batch_size, num_steps, sigma_min, sigma_max, rho, dev, seed=seed
    )

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "prior_samples.npy")
    np.save(out_path, x.detach().cpu().numpy())
    print(f"Saved {batch_size} unconditional prior samples to {out_path}")


if __name__ == "__main__":
    main()
