import numpy as np

d = np.load("slurm/logs/burgers_gem_N8_K2000_51193894.npz")  # fill in the real filename
print(d.files)

w, err = d["weights"], d["per_particle_rel_err"]
print(f"n_resample: {int(d['n_resample'])}  (should be 0)")
print(f"weighted_rel_err: {float(d['weighted_rel_err']):.5f}")
print(f"per-particle error: min={err.min():.5f} max={err.max():.5f} mean={err.mean():.5f}")
print(f"corr(weight, error): {np.corrcoef(w, err)[0,1]:.4f}")
print(list(zip(w, err)))
