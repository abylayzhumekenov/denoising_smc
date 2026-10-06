"""Regenerate the exp4 (proposal x K) summary table from ``results/``.

exp4 sweeps proposal {em, heun} x weighting {pbs, girs, unw} x K, at beta=1e4, omega=0, N=8, plus
an ODE K-sweep. Scans ``results/``, filters to the campaign run ids (SMC: ``run_id`` starts with
``exp4_`` and ``n_particles == N``; ODE: ``run_id`` starts with ``exp4_ode``), writes
``slurm/logs4/exp4_table.md`` and prints it. Safe with partial or no results.

Env overrides (used by local smoke tests): ``EXP4_N`` (8), ``EXP4_PREFIX`` (exp4),
``EXP4_OUT`` (slurm/logs4/exp4_table.md).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Allow `import smc.burgers` when this file is run as a script (sys.path[0] is the script dir).
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

COLS = ['run', 'method', 'proposal', 'beta', 'omega', 'ratio', 'lambda_girs', 'rho_temp',
        'rho_init', 'N', 'K', 'rel_err', 'frac_obs', 'frac_pde', 'corr_rel', 'resid_out',
        'resid_diag', 'ess_mean', 'ess_last', 'seed']
ROOT = Path('results')
OUT = Path(os.environ.get('EXP4_OUT', 'slurm/logs4/exp4_table.md'))
N = int(os.environ.get('EXP4_N', '8'))
PREFIX = os.environ.get('EXP4_PREFIX', 'exp4')


def _load(run_dir):
    metrics = json.load(open(run_dir / 'metrics.json'))
    config = yaml.safe_load(open(run_dir / 'config.yaml'))
    return metrics, config


def _rms_resid(u):
    """RMS of the raw Burger residual ``f`` on a saved raw field (same stencil as the runner).

    Blank on any failure (missing arrays, torch unavailable). The reference is the *truth's*
    residual under the same stencil, not zero.
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


def _npz_info(run_dir):
    """``(ess_mean, ess_last, resid_out, resid_diag)`` from ``result.npz``; blanks if absent.

    ``resid_out``  = RMS Burger residual of the saved output field (``weighted_mean``).
    ``resid_diag`` = RMS residual of the last saved denoised estimate
                     (``sqrt(diag_loss_pde[-1])``), a smoother proxy for the output residual.
    """
    out = dict(ess_mean='', ess_last='', resid_out='', resid_diag='')
    npz = run_dir / 'result.npz'
    if not npz.exists():
        return out
    try:
        d = np.load(npz, allow_pickle=True)
    except Exception:
        return out
    if 'ess_history' in d and len(d['ess_history']):
        out['ess_mean'] = float(np.mean(d['ess_history']))
        out['ess_last'] = float(d['ess_history'][-1])
    if 'diag_loss_pde' in d and len(d['diag_loss_pde']):
        out['resid_diag'] = round(float(np.sqrt(d['diag_loss_pde'][-1])), 5)
    if 'weighted_mean' in d:
        out['resid_out'] = _rms_resid(d['weighted_mean'])
    return out


def _smc_row(run_dir):
    m, c = _load(run_dir)
    s = c.get('smc', {})
    ll = s.get('likelihood', {})
    info = _npz_info(run_dir)
    beta = ll.get('obs_weight', '')
    omega = ll.get('pde_weight', '')
    ratio = (omega / beta) if isinstance(beta, (int, float)) and isinstance(omega, (int, float)) and beta else ''
    return dict(run=m.get('run_id', run_dir.name), method='smc',
                proposal=m.get('proposal', s.get('proposal', '')),
                beta=beta, omega=omega, ratio=ratio,
                lambda_girs=s.get('lambda_girs', ''), rho_temp=s.get('rho_temp', ''),
                rho_init=s.get('rho_temp_init', ''), N=s.get('n_particles', ''),
                K=s.get('num_steps', ''), rel_err=m.get('relative_error'),
                frac_obs=m.get('guidance_frac_obs_mean', ''),
                frac_pde=m.get('guidance_frac_pde_mean', ''),
                corr_rel=m.get('proposal_correction_rel_mean', ''),
                resid_out=info['resid_out'], resid_diag=info['resid_diag'],
                ess_mean=info['ess_mean'], ess_last=info['ess_last'], seed=m.get('seed', ''))


def _ode_row(run_dir):
    m, c = _load(run_dir)
    resid_out = ''
    npy = run_dir / 'result.npy'
    if npy.exists():
        try:
            resid_out = _rms_resid(np.load(npy))
        except Exception:
            resid_out = ''
    return dict(run=m.get('run_id', run_dir.name), method='ode', proposal='ode',
                beta='', omega='', ratio='', lambda_girs='', rho_temp='', rho_init='',
                N=c.get('generate', {}).get('batch_size', ''),
                K=c.get('test', {}).get('iterations', ''), rel_err=m.get('relative_error'),
                frac_obs='', frac_pde='', corr_rel='', resid_out=resid_out, resid_diag='',
                ess_mean='', ess_last='', seed=m.get('seed', ''))


rows = []
for p in sorted((ROOT / 'ode' / 'burgers').glob('*/metrics.json')):
    try:
        r = _ode_row(p.parent)
    except Exception:
        continue
    if str(r['run']).startswith(PREFIX + '_ode'):
        rows.append(r)
for p in sorted((ROOT / 'smc' / 'burgers').glob('*/metrics.json')):
    try:
        r = _smc_row(p.parent)
    except Exception:
        continue
    if r['N'] == N and str(r['run']).startswith(PREFIX + '_'):
        rows.append(r)


def _sort_key(r):
    order = {'ode': 0, 'em': 1, 'heun': 2}
    return (order.get(r['proposal'], 9), str(r['lambda_girs']), str(r['rho_temp']),
            r['K'] if isinstance(r['K'], int) else 0)


rows.sort(key=_sort_key)

lines = [f'<!-- exp4 table, generated {time.strftime("%Y-%m-%d %H:%M:%S")}, {len(rows)} runs -->',
         '| ' + ' | '.join(COLS) + ' |',
         '|' + '|'.join(['---'] * len(COLS)) + '|']
for r in rows:
    lines.append('| ' + ' | '.join(str(r.get(c, '')) for c in COLS) + ' |')
table = '\n'.join(lines)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(table + '\n')
print(table)
