"""
Pre-compute MIRA scores for the PoseBusters benchmark set.

Saves mira_names_rmsd.npy and mira_scores_rmsd.npy to the metrics dir so
posebusters_calibration.ipynb can load them from cache.

Run via SLURM:
    sbatch ~/slurm/thesis/run_mira_pb_benchmark.sh
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/qf226/MProject/thesis")
sys.path.insert(0, "/home/qf226/MProject/thesis/diffdock")

from eval_diffdock.mira_runner import compute_mira_scores

RESULTS_DIR = "/home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged"
DATA_DIR    = "/home/qf226/rds/hpc-work/data/posebusters_benchmark_set"
METRICS     = os.path.join(RESULTS_DIR, "metrics")

results_index = {
    d.name: d
    for d in sorted(Path(RESULTS_DIR).iterdir())
    if d.is_dir() and any(d.glob("rank*.sdf"))
}
complex_names = np.array(sorted(results_index.keys()))
print(f"Complexes: {len(complex_names)}")

MIRA_NAMES_PATH  = f"{METRICS}/mira_names_rmsd.npy"
MIRA_SCORES_PATH = f"{METRICS}/mira_scores_rmsd.npy"

if os.path.exists(MIRA_SCORES_PATH):
    print("MIRA scores already exist, skipping.")
else:
    print("\nRunning MIRA (metric=rmsd, num_runs=100) ...")
    mira_names, mira_scores = compute_mira_scores(
        complex_names, results_index, DATA_DIR,
        num_runs=100, verbose=True, metric="rmsd",
    )
    np.save(MIRA_NAMES_PATH,  mira_names)
    np.save(MIRA_SCORES_PATH, mira_scores)
    print(f"Saved MIRA scores ({len(mira_scores)} complexes)")
    print(f"Overall MIRA: {mira_scores.mean():.4f} ± {mira_scores.std():.4f}")

print("\nDone.")
