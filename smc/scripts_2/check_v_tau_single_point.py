"""Sanity check: compute V_tau(x) at a single representative point (sigma_t=5.0, real Burgers
model + data), matching the setup in smc/scripts_2/hutchinson_findings.md, using the fused/corrected
implementation in smc/scripts_2/weightings/doob_vtau.py.

Usage: venv/bin/python -m smc.scripts_2.check_v_tau_single_point [num_probes]
"""

import sys

import torch

from torch_utils.misc import auto_device
from smc.scripts_2.models.burgers import random_sensor, load_ground_truth, load_network, burgers_ell_fn
from smc.scripts_2.weightings.doob_vtau import compute_v_tau_terms


def main():
    num_probes = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    device = auto_device()
    torch.manual_seed(0)

    ground_truth = load_ground_truth('data/testing/burgers.mat', 0, device)
    mask = random_sensor(5, 128, seed=0, device=device)
    net = load_network('pretrained-models/pretrained-burgers.pkl', device)

    sigma_t = torch.tensor(5.0, dtype=torch.float64, device=device)
    x_cur = torch.randn(1, 1, 128, 128, dtype=torch.float64, device=device) * sigma_t

    ell_fn = burgers_ell_fn(net, ground_truth, mask, zeta_obs=320, device=device)

    print(f'Running with num_probes={num_probes} ...', flush=True)
    result = compute_v_tau_terms(ell_fn, x_cur, sigma_t, num_probes, generator=torch.Generator(device=device).manual_seed(0))

    print()
    for k, v in result.items():
        if k != 'samples':
            print(f'{k:20s} = {v}')
    print()
    print('samples:', result['samples'])


if __name__ == '__main__':
    main()
