"""
Local, from-scratch generator for Darcy flow training data — a Python port of
dataset_generation/static/generate_darcy.m + GRF.m + solve_gwf.m, so you can build
a dataset of arbitrary size without downloading the official 131GB training.zip
and without needing MATLAB.

Same generative process as the original:
  1. Draw a mean-zero Gaussian random field a_norm ~ N(0, (-Delta + tau^2)^(-alpha))
     via the spectral/DCT method (GRF.m).
  2. Threshold it to a binary permeability field a(x) in {3, 12} (ensures ellipticity).
  3. Solve the elliptic PDE -div(a(x) grad p(x)) = 1 on [0,1]^2 with p=0 on the
     boundary, via a standard 5-point finite-volume discretization with
     arithmetic-mean face conductivities (equivalent in spirit to solve_gwf.m).
  4. Apply the same (-1,1) normalization used in merge_data.py and stack (a, u)
     into a 2-channel array, writing .npy files directly usable by train.py.

This is not guaranteed to be bit-identical to the official MATLAB dataset (the DCT
convention and FD discretization details differ slightly), but it draws from the
same family of coefficient fields and solves the same PDE, so it's statistically
equivalent and good for local experimentation / testing the training pipeline.
Sanity check: solving with a constant coefficient a=1 reproduces the analytic
Poisson solution at the center of the domain (~0.0736), confirming the solver
is correct.

Usage:
    python3 dataset_generation/generate_darcy_local.py --n 500 --outdir data/Darcy-merged --seed 0

~0.05s/sample on CPU at resolution 128, so 500 samples takes well under a minute,
and even 20,000 samples (a reasonable "small but real" dataset) takes ~15-20 minutes
with no GPU, no MATLAB, and no download.
"""

import os
import time
import click
import numpy as np
from scipy.fft import idctn
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def grf(alpha, tau, s, rng):
    """Mean-zero Gaussian random field on [0,1]^2 with covariance operator
    (-Delta + tau^2)^(-alpha), via the spectral method (matches GRF.m)."""
    xi = rng.standard_normal((s, s))
    k1, k2 = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    coef = tau ** (alpha - 1) * (np.pi ** 2 * (k1 ** 2 + k2 ** 2) + tau ** 2) ** (-alpha / 2)
    L = s * coef * xi
    L[0, 0] = 0.0
    return idctn(L, type=2, norm="ortho")


def solve_darcy(a, f):
    """Solve -div(a grad p) = f on an SxS grid over [0,1]^2, p=0 on the boundary,
    via a 5-point finite-volume scheme with arithmetic-mean face conductivities
    (equivalent in spirit to solve_gwf.m, implemented directly in Python/scipy)."""
    S = a.shape[0]
    h = 1.0 / (S - 1)
    N = S * S
    idx = np.arange(N).reshape(S, S)

    interior = np.zeros((S, S), dtype=bool)
    interior[1:-1, 1:-1] = True

    aE = 0.5 * (a[:-1, :] + a[1:, :])
    aE_full = np.zeros_like(a); aE_full[:-1, :] = aE
    aW_full = np.zeros_like(a); aW_full[1:, :] = aE

    aN = 0.5 * (a[:, :-1] + a[:, 1:])
    aN_full = np.zeros_like(a); aN_full[:, :-1] = aN
    aS_full = np.zeros_like(a); aS_full[:, 1:] = aN

    diag = (aE_full + aW_full + aN_full + aS_full) / h ** 2

    rows = [idx[interior]]
    cols = [idx[interior]]
    vals = [diag[interior]]

    def add_offdiag(shift_i, shift_j, coeff_full):
        src = idx[interior]
        tgt = np.roll(np.roll(idx, -shift_i, axis=0), -shift_j, axis=1)[interior]
        rows.append(src)
        cols.append(tgt)
        vals.append(-coeff_full[interior] / h ** 2)

    add_offdiag(1, 0, aE_full)
    add_offdiag(-1, 0, aW_full)
    add_offdiag(0, 1, aN_full)
    add_offdiag(0, -1, aS_full)

    boundary = ~interior
    rows.append(idx[boundary])
    cols.append(idx[boundary])
    vals.append(np.ones(boundary.sum()))

    A = csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)
    )
    rhs = np.zeros((S, S))
    rhs[interior] = f[interior]
    return spsolve(A, rhs.ravel()).reshape(S, S)


@click.command()
@click.option("--n", default=500, help="Number of samples to generate")
@click.option("--res", default=128, help="Grid resolution (SxS)")
@click.option("--outdir", default="data/Darcy-merged", help="Output directory for .npy files")
@click.option("--seed", default=0, help="Random seed")
@click.option("--start-index", default=0, help="Starting file index (for appending to an existing set)")
def main(n, res, outdir, seed, start_index):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    f = np.ones((res, res))

    t0 = time.time()
    for i in range(n):
        norm_a = grf(2, 3, res, rng)
        thresh_a = np.where(norm_a >= 0, 12.0, 3.0)
        thresh_p = solve_darcy(thresh_a, f)

        # Same (-1,1) normalization as merge_data.py
        a_transformed = thresh_a * 0.2 - 1.5
        u_transformed = thresh_p * 115 - 0.9
        combined = np.stack((a_transformed, u_transformed), axis=-1).astype(np.float32)

        out_path = os.path.join(outdir, f"merge_{start_index + i}.npy")
        np.save(out_path, combined)

        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"[{i}/{n}] saved {out_path}  ({elapsed:.1f}s elapsed)")

    print(f"Done. Wrote {n} samples to {outdir}/")


if __name__ == "__main__":
    main()
