"""Dispatcher for the new SMC sampler, mirroring ``generate_pde.py``.

Usage:

    python3 generate_pde_smc.py --config configs/smc/burgers.yaml

Optional overrides: ``--n-particles``, ``--num-steps``, ``--run-id``, ``--out-dir`` apply to the
``smc`` block; ``--obs-weight`` / ``--obs-weight-a`` / ``--obs-weight-u`` / ``--pde-weight`` apply
to ``smc.likelihood``; ``--rho-temp``, ``--rho-temp-init``, ``--lambda-girs`` and ``--proposal``
apply to ``smc``; ``--offset`` / ``--sensor-seed`` / ``--sensor-seed-a`` / ``--sensor-seed-u``
apply to ``data``; ``--seed`` applies to ``generate``.
"""

import argparse

import yaml

from smc import burgers, darcy


def main(config, overrides):
    smc = config.setdefault('smc', {})
    likelihood = smc.setdefault('likelihood', {})
    data = config.setdefault('data', {})
    gen = config.setdefault('generate', {})
    for key, value in overrides.items():
        if value is None:
            continue
        if key in ('obs_weight', 'obs_weight_a', 'obs_weight_u', 'pde_weight'):
            likelihood[key] = value
        elif key in ('offset', 'sensors', 'sensor_seed', 'sensor_seed_a', 'sensor_seed_u'):
            data[key] = value
        elif key == 'seed':
            gen[key] = value
        else:
            smc[key] = value

    name = config['data']['name']
    if name == 'Burgers':
        return burgers.run(config)
    if name == 'Darcy':
        return darcy.run(config)
    raise SystemExit(f"no SMC implementation for PDE {name!r}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the SMC sampler on a PDE config')
    parser.add_argument('--config', type=str, default='configs/smc/burgers.yaml')
    parser.add_argument('--n-particles', type=int, default=None)
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--obs-weight', type=float, default=None)
    parser.add_argument('--obs-weight-a', type=float, default=None)
    parser.add_argument('--obs-weight-u', type=float, default=None)
    parser.add_argument('--pde-weight', type=float, default=None)
    parser.add_argument('--rho-temp', type=float, default=None)
    parser.add_argument('--rho-temp-init', type=float, default=None)
    parser.add_argument('--lambda-girs', type=float, default=None)
    parser.add_argument('--offset', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sensor-seed', type=int, default=None)
    parser.add_argument('--sensor-seed-a', type=int, default=None)
    parser.add_argument('--sensor-seed-u', type=int, default=None)
    parser.add_argument('--proposal', type=str, default=None, choices=['em', 'heun'],
                        help="SMC proposal integrator: 'em' (Euler--Maruyama) or 'heun' (2nd order).")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg, dict(n_particles=args.n_particles, num_steps=args.num_steps,
                   run_id=args.run_id, out_dir=args.out_dir,
                   obs_weight=args.obs_weight, obs_weight_a=args.obs_weight_a,
                   obs_weight_u=args.obs_weight_u, pde_weight=args.pde_weight,
                   rho_temp=args.rho_temp, rho_temp_init=args.rho_temp_init,
                   lambda_girs=args.lambda_girs, proposal=args.proposal,
                   offset=args.offset, seed=args.seed,
                   sensor_seed=args.sensor_seed, sensor_seed_a=args.sensor_seed_a,
                   sensor_seed_u=args.sensor_seed_u))
