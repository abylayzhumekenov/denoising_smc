"""Multi-x V_tau(x) sweep: same sigma_t, several independent x draws, real Burgers model+data.

Usage: venv/bin/python -m smc.scripts_2.check_v_tau_multi_x [num_probes] [num_points]
"""

import sys
import json
import time

import torch

from torch_utils.misc import auto_device
from smc.scripts_2.models.burgers import random_sensor, load_ground_truth, load_network, burgers_ell_fn
from smc.scripts_2.weightings.doob_vtau import compute_v_tau_terms


def main():
    num_probes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    num_points = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    device = auto_device()

    ground_truth = load_ground_truth('data/testing/burgers.mat', 0, device)
    mask = random_sensor(5, 128, seed=0, device=device)
    net = load_network('pretrained-models/pretrained-burgers.pkl', device)

    ell_fn = burgers_ell_fn(net, ground_truth, mask, zeta_obs=320, device=device)
    sigma_t_value = 5.0
    out_path = f'smc/scripts_2/v_tau_multi_x_results_M{num_probes}_N{num_points}.json'

    results = []
    for seed in range(num_points):
        torch.manual_seed(seed)
        sigma_t = torch.tensor(sigma_t_value, dtype=torch.float64, device=device)
        x_cur = torch.randn(1, 1, 128, 128, dtype=torch.float64, device=device) * sigma_t

        t0 = time.time()
        result = compute_v_tau_terms(
            ell_fn, x_cur, sigma_t, num_probes,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        result['seed'] = seed
        result['time_s'] = time.time() - t0
        results.append(result)

        print(f'--- seed={seed} done in {result["time_s"]:.1f}s ---', flush=True)
        for k, v in result.items():
            if k != 'samples':
                print(f'  {k:20s} = {v}')
        print(flush=True)

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
