"""Regenerate the exp1 summary table from whatever is currently in ``results/``.

Run at the end of each exp1 sbatch (and any time by hand). It scans the run dirs, filters to the
exp1 settings (ODE ``iterations == 2000``; SMC ``num_steps == 2000`` and ``n_particles == 8``),
writes ``slurm/logs1/exp1_table.md`` (overwrite) and prints the table. Safe to call repeatedly,
with partial or no results.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Allow `import smc.burgers` when this file is run as a script (sys.path[0] is the script dir).
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

COLS = ['run', 'method', 'beta', 'omega', 'rho_temp', 'rho_init', 'lambda_girs',
        'N', 'K', 'rel_err', 'rel_err_norm', 'resid_out', 'resid_diag', 'ess_final']
OUT = Path('slurm/logs1/exp1_table.md')


def _load(run_dir):
    metrics = json.load(open(run_dir / 'metrics.json'))
    config = yaml.safe_load(open(run_dir / 'config.yaml'))
    return metrics, config


def _rms_resid(u):
    """RMS of the raw Burger residual ``f`` on a saved raw field (same stencil as the runner).

    Blank on any failure. The reference is the *truth's* residual under the same stencil, not zero.
    """
    try:
        import torch
        from smc.burgers import burger_residual
        u = np.asarray(u, dtype=np.float64)
        if u.ndim == 2:
            u = u[None, None]
        elif u.ndim == 3:
            u = u[:, None]
        f = burger_residual(torch.tensor(u, dtype=torch.float64))
        return round(float((f ** 2).mean().sqrt()), 5)
    except Exception:
        return ''


def _smc_row(run_dir):
    m, c = _load(run_dir)
    s = c.get('smc', {})
    ll = s.get('likelihood', {})
    ess, resid_out, resid_diag = '', '', ''
    npz = run_dir / 'result.npz'
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        if 'ess_history' in d and len(d['ess_history']):
            ess = float(d['ess_history'][-1])
        if 'diag_loss_pde' in d and len(d['diag_loss_pde']):
            resid_diag = round(float(np.sqrt(d['diag_loss_pde'][-1])), 5)
        if 'weighted_mean' in d:
            resid_out = _rms_resid(d['weighted_mean'])
    return dict(run=m.get('run_id', run_dir.name), method='smc',
                beta=ll.get('obs_weight', ''), omega=ll.get('pde_weight', ''),
                rho_temp=s.get('rho_temp', ''), rho_init=s.get('rho_temp_init', ''),
                lambda_girs=s.get('lambda_girs', ''),
                N=s.get('n_particles', ''), K=s.get('num_steps', ''),
                rel_err=m.get('relative_error'), rel_err_norm=m.get('relative_error_norm', ''),
                resid_out=resid_out, resid_diag=resid_diag, ess_final=ess)


def _ode_row(run_dir):
    m, c = _load(run_dir)
    resid_out = ''
    npy = run_dir / 'result.npy'
    if npy.exists():
        try:
            resid_out = _rms_resid(np.load(npy))
        except Exception:
            resid_out = ''
    return dict(run=m.get('run_id', run_dir.name), method='ode', beta='', omega='',
                rho_temp='', rho_init='', lambda_girs='',
                N=c.get('generate', {}).get('batch_size', ''),
                K=c.get('test', {}).get('iterations', ''),
                rel_err=m.get('relative_error'), rel_err_norm=m.get('relative_error_norm', ''),
                resid_out=resid_out, resid_diag='', ess_final='')


rows = []
for p in sorted(Path('results/ode/burgers').glob('*/metrics.json')):
    try:
        r = _ode_row(p.parent)
    except Exception:
        continue
    if r['K'] == 2000:
        rows.append(r)
for p in sorted(Path('results/smc/burgers').glob('*/metrics.json')):
    try:
        r = _smc_row(p.parent)
    except Exception:
        continue
    if r['K'] == 2000 and r['N'] == 8:
        rows.append(r)

lines = [f'<!-- exp1 table, generated {time.strftime("%Y-%m-%d %H:%M:%S")}, {len(rows)} runs -->',
         '| ' + ' | '.join(COLS) + ' |',
         '|' + '|'.join(['---'] * len(COLS)) + '|']
for r in rows:
    lines.append('| ' + ' | '.join(str(r.get(c, '')) for c in COLS) + ' |')
table = '\n'.join(lines)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(table + '\n')
print(table)
