"""
Pre-compute MIRA scores and RMSD accuracy for the PDBBind test set.

Saves mira_names_rmsd.npy, mira_scores_rmsd.npy, and rmsd_accuracy.npz to
the metrics dir so tarp_analysis.ipynb can load them from cache.

Run via SLURM:
    sbatch ~/slurm/thesis/run_mira_pdbbind.sh
"""

import os
import sys
import numpy as np

sys.path.insert(0, "/home/qf226/MProject/thesis")
sys.path.insert(0, "/home/qf226/MProject/thesis/diffdock")

from eval_diffdock.loader import build_results_index
from eval_diffdock.mira_runner import compute_mira_scores, compute_rmsd_accuracy

RESULTS_FULL = "/home/qf226/rds/hpc-work/results/DiffDock/pdbbind_testset/raw_chunks"
MERGED       = "/home/qf226/rds/hpc-work/results/DiffDock/pdbbind_testset"
METRICS      = os.path.join(MERGED, "metrics")
DATA_DIR     = "/home/qf226/rds/hpc-work/data/PDBBind_processed"

complex_names = np.load(f"{METRICS}/complex_names.npy", allow_pickle=True)
results_index = build_results_index(RESULTS_FULL)
print(f"Complexes: {len(complex_names)}, indexed: {len(results_index)}")

# --- MIRA scores (metric=rmsd) ---
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

# --- RMSD accuracy (best-of-N) ---
RMSD_ACC_PATH = f"{METRICS}/rmsd_accuracy.npz"

if os.path.exists(RMSD_ACC_PATH):
    print("RMSD accuracy already exists, skipping.")
else:
    print("\nRunning RMSD accuracy ...")
    rmsd_acc_names, rmsd_acc_min, rmsd_acc_fracs = compute_rmsd_accuracy(
        complex_names, results_index, DATA_DIR,
        thresholds=(2.0, 5.0), verbose=True,
    )
    np.savez(RMSD_ACC_PATH,
             names=rmsd_acc_names, min_rmsds=rmsd_acc_min, fracs=rmsd_acc_fracs)
    print(f"Saved RMSD accuracy ({len(rmsd_acc_names)} complexes)")
    print(f"  < 2Å: {(rmsd_acc_min < 2.0).mean()*100:.1f}%")
    print(f"  < 5Å: {(rmsd_acc_min < 5.0).mean()*100:.1f}%")

print("\nDone.")
