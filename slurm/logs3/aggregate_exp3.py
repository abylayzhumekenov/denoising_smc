"""Regenerate the exp3 (Stage 2: method suite) summary table from ``results/``.

Run at the end of each exp3 sbatch (and any time by hand). Scans ``results/``, filters to the
campaign settings (ODE: ``test.iterations == K``; SMC: ``num_steps == K``, ``n_particles == N``,
``run_id`` starts with ``PREFIX``), writes ``slurm/logs3/exp3_table.md`` and prints it. Safe with
partial or no results.

Env overrides (used by local smoke tests): ``EXP3_K`` (2000), ``EXP3_N`` (8),
``EXP3_PREFIX`` (exp3), ``EXP3_OUT`` (slurm/logs3/exp3_table.md).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

COLS = ['run', 'method', 'beta', 'omega', 'lambda_girs', 'rho_temp', 'rho_init',
        'N', 'K', 'rel_err', 'frac_obs', 'frac_pde', 'ess_mean', 'ess_last', 'seed']
ROOT = Path('results')
OUT = Path(os.environ.get('EXP3_OUT', 'slurm/logs3/exp3_table.md'))
K = int(os.environ.get('EXP3_K', '2000'))
N = int(os.environ.get('EXP3_N', '8'))
PREFIX = os.environ.get('EXP3_PREFIX', 'exp3')


def _load(run_dir):
    metrics = json.load(open(run_dir / 'metrics.json'))
    config = yaml.safe_load(open(run_dir / 'config.yaml'))
    return metrics, config


def _ess(run_dir):
    npz = run_dir / 'result.npz'
    if not npz.exists():
        return '', ''
    d = np.load(npz)
    if 'ess_history' not in d or not len(d['ess_history']):
        return '', ''
    return float(np.mean(d['ess_history'])), float(d['ess_history'][-1])


def _smc_row(run_dir):
    m, c = _load(run_dir)
    s = c.get('smc', {})
    ll = s.get('likelihood', {})
    ess_mean, ess_last = _ess(run_dir)
    return dict(run=m.get('run_id', run_dir.name), method='smc',
                beta=ll.get('obs_weight', ''), omega=ll.get('pde_weight', ''),
                lambda_girs=s.get('lambda_girs', ''), rho_temp=s.get('rho_temp', ''),
                rho_init=s.get('rho_temp_init', ''), N=s.get('n_particles', ''),
                K=s.get('num_steps', ''), rel_err=m.get('relative_error'),
                frac_obs=m.get('guidance_frac_obs_mean', ''),
                frac_pde=m.get('guidance_frac_pde_mean', ''),
                ess_mean=ess_mean, ess_last=ess_last, seed=m.get('seed', ''))


def _ode_row(run_dir):
    m, c = _load(run_dir)
    return dict(run=m.get('run_id', run_dir.name), method='ode', beta='', omega='',
                lambda_girs='', rho_temp='', rho_init='',
                N=c.get('generate', {}).get('batch_size', ''),
                K=c.get('test', {}).get('iterations', ''), rel_err=m.get('relative_error'),
                frac_obs='', frac_pde='', ess_mean='', ess_last='', seed=m.get('seed', ''))


rows = []
for p in sorted((ROOT / 'ode' / 'burgers').glob('*/metrics.json')):
    try:
        r = _ode_row(p.parent)
    except Exception:
        continue
    if r['K'] == K:
        rows.append(r)
for p in sorted((ROOT / 'smc' / 'burgers').glob('*/metrics.json')):
    try:
        r = _smc_row(p.parent)
    except Exception:
        continue
    if r['K'] == K and r['N'] == N and str(r['run']).startswith(PREFIX):
        rows.append(r)

lines = [f'<!-- exp3 table, generated {time.strftime("%Y-%m-%d %H:%M:%S")}, {len(rows)} runs -->',
         '| ' + ' | '.join(COLS) + ' |',
         '|' + '|'.join(['---'] * len(COLS)) + '|']
for r in rows:
    lines.append('| ' + ' | '.join(str(r.get(c, '')) for c in COLS) + ' |')
table = '\n'.join(lines)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(table + '\n')
print(table)
