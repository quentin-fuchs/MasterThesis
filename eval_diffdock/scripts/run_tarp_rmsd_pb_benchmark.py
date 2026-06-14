"""
Pre-compute TARP RMSD fractions for the PoseBusters benchmark set.

Saves tarp_fractions_rmsd_K1.npy to the metrics dir so
posebusters_calibration.ipynb can load it from cache.

Run via SLURM:
    sbatch ~/slurm/thesis/run_tarp_rmsd_pb_benchmark.sh
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/qf226/MProject/thesis")
sys.path.insert(0, "/home/qf226/MProject/thesis/diffdock")

from eval_diffdock.loader import build_results_index
from eval_diffdock.tarp_runner import run_tarp_eval

RESULTS_DIR = "/home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged"
DATA_DIR    = "/home/qf226/rds/hpc-work/data/posebusters_benchmark_set"
METRICS     = os.path.join(RESULTS_DIR, "metrics")

N_WORKERS = int(os.environ.get("N_WORKERS", 1))

results_index = {
    d.name: d
    for d in sorted(Path(RESULTS_DIR).iterdir())
    if d.is_dir() and any(d.glob("rank*.sdf"))
}
complex_names = np.array(sorted(results_index.keys()))
print(f"Complexes: {len(complex_names)}")

K = 1
out_path = f"{METRICS}/tarp_fractions_rmsd_K{K}.npy"
if os.path.exists(out_path):
    print(f"K={K} already exists, skipping.")
else:
    print(f"\nRunning TARP RMSD K={K} (n_workers={N_WORKERS}) ...")
    f_rmsd = run_tarp_eval(
        complex_names, results_index, DATA_DIR,
        K=K, mode="rmsd", seed=42, verbose=True, n_workers=N_WORKERS,
    )
    np.save(out_path, f_rmsd)
    print(f"Saved {out_path}  shape={f_rmsd.shape}")

print("\nDone.")
