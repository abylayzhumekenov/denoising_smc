"""Dispatcher for the new SMC sampler, mirroring ``generate_pde.py``.

Usage:

    python3 generate_pde_smc.py --config configs/smc/burgers.yaml

Optional overrides (applied to the ``smc`` block): ``--n-particles``, ``--num-steps``,
``--run-id``, ``--out-dir``.
"""

import argparse

import yaml

from smc import burgers


def main(config, overrides):
    smc = config.setdefault('smc', {})
    for key, value in overrides.items():
        if value is not None:
            smc[key] = value

    name = config['data']['name']
    if name == 'Burgers':
        return burgers.run(config)
    raise SystemExit(f"no SMC implementation for PDE {name!r}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the SMC sampler on a PDE config')
    parser.add_argument('--config', type=str, default='configs/smc/burgers.yaml')
    parser.add_argument('--n-particles', type=int, default=None)
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg, dict(n_particles=args.n_particles, num_steps=args.num_steps,
                   run_id=args.run_id, out_dir=args.out_dir))
