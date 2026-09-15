"""Shared run-output helpers for baseline (ODE) and SMC runs.

Every run writes one directory:

    results/<method>/<pde>/<run_id>/
        result.*      # the solver's native array format (.npy/.mat)
        config.yaml   # fully resolved config, including run_id
        metrics.json  # scalar metrics

``run_id`` defaults to ``<timestamp>_<6-char hash of the resolved config>`` and can be overridden
with ``generate.run_id``. The output root defaults to ``results/<method>`` and can be overridden
with ``generate.out_dir``. Neither override requires editing the upstream configs.

This is shared infrastructure (docs/vision.md, D11/D13): the ODE baseline and the SMC
implementation both import it; neither imports the other.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import yaml


def _config_digest(config, section='generate'):
    payload = copy.deepcopy(config)
    block = payload.get(section, {})
    if isinstance(block, dict):
        block.pop('run_id', None)
        block.pop('out_dir', None)
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:6]


def new_run_id(config, section='generate'):
    """Timestamped run id with a short content hash of the resolved config."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{_config_digest(config, section)}"


def make_run_dir(config, pde, default_root='results/ode', section='generate'):
    """Create and return ``(run_dir, run_id)`` for a run of ``pde`` (a lowercase slug).

    ``out_dir`` and ``run_id`` are read from ``config[section]`` and default to
    ``default_root`` and a timestamped config hash respectively.
    """
    block = config.get(section, {}) or {}
    root = Path(block.get('out_dir', default_root))
    run_id = block.get('run_id') or new_run_id(config, section)
    run_dir = root / pde / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def write_config(run_dir, config, run_id, section='generate'):
    """Write the resolved config (with ``run_id`` filled into ``section``) next to the results."""
    resolved = copy.deepcopy(config)
    resolved.setdefault(section, {})
    resolved[section]['run_id'] = run_id
    with open(Path(run_dir) / 'config.yaml', 'w') as f:
        yaml.safe_dump(resolved, f, sort_keys=False)


def write_metrics(run_dir, metrics):
    """Write scalar metrics to ``metrics.json``."""
    with open(Path(run_dir) / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
