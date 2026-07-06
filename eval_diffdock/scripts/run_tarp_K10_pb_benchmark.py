"""Compute TARP symRMSD fractions with K=10 for the PoseBusters benchmark set.

Extends the existing K=1 result by rerunning with K=10 reference draws,
giving a smoother coverage estimate for comparison with SigmaDock (K=10).

Saves to metrics/:
  tarp_fractions_symrmsd_K10.npy  — TARP fractions (n, 10), sym-corrected RMSD

Run via SLURM:
    sbatch ~/slurm/thesis/run_tarp_K10_pb_benchmark.sh
"""

import os
import numpy as np

from eval_diffdock.loader import build_results_index
from eval_diffdock.tarp_runner import run_tarp_eval

PB_DIR      = "/home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged"
RESULTS_DIR = os.path.join(PB_DIR, "poses")
DATA_DIR    = "/home/qf226/rds/hpc-work/data/posebusters_benchmark_set"
METRICS     = os.environ.get("METRICS_OUT", os.path.join(PB_DIR, "metrics"))

N_WORKERS = int(os.environ.get("N_WORKERS", 1))

results_index = build_results_index(RESULTS_DIR)
complex_names = np.array(sorted(results_index.keys()))
print(f"Complexes: {len(complex_names)}")

out_K10 = f"{METRICS}/tarp_fractions_symrmsd_K10.npy"
if os.path.exists(out_K10):
    print(f"K=10 already exists at {out_K10}, skipping.")
else:
    print(f"\nRunning TARP symRMSD K=10 (n_workers={N_WORKERS}) ...")
    f = run_tarp_eval(complex_names, results_index, DATA_DIR,
                      K=10, mode="rmsd", seed=42, verbose=True, n_workers=N_WORKERS)
    np.save(out_K10, f)
    print(f"Saved {out_K10}  shape={f.shape}")

print("\nDone.")
