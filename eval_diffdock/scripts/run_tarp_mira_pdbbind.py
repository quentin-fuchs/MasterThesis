"""
Pre-compute TARP RMSD fractions, MIRA scores, and RMSD accuracy for the PDBBind test set.

Merges run_tarp_rmsd_pdbbind.py and run_mira_pdbbind.py into one job so the
results index and SDF files are loaded only once.

Saves to metrics/:
  tarp_fractions_rmsd_K10.npy  — TARP fractions (n, 10), sym-corrected RMSD
  tarp_fractions_rmsd_K1.npy   — TARP fractions (n,  1)
  mira_names_symrmsd.npy       — complex names for MIRA (sym-corrected RMSD)
  mira_scores_symrmsd.npy      — per-complex MIRA scores
  rmsd_accuracy.npz            — top-1 and any-sample RMSD accuracy

Run via SLURM:
    sbatch ~/slurm/thesis/run_tarp_mira_pdbbind.sh
"""

import os
import numpy as np

from eval_diffdock.loader import build_results_index
from eval_diffdock.tarp_runner import run_tarp_eval
from eval_diffdock.mira_runner import compute_mira_scores, compute_rmsd_accuracy

RESULTS_FULL = "/home/qf226/rds/hpc-work/results/DiffDock/pdbbind_testset/raw_chunks"
MERGED       = "/home/qf226/rds/hpc-work/results/DiffDock/pdbbind_testset"
METRICS      = os.path.join(MERGED, "metrics")
DATA_DIR     = "/home/qf226/rds/hpc-work/data/PDBBind_processed"

N_WORKERS = int(os.environ.get("N_WORKERS", 1))

complex_names = np.load(f"{METRICS}/complex_names.npy", allow_pickle=True)
results_index = build_results_index(RESULTS_FULL)
print(f"Complexes: {len(complex_names)}, indexed: {len(results_index)}")

# --- TARP RMSD K=10 ---
K = 10
out_K = f"{METRICS}/tarp_fractions_rmsd_K{K}.npy"
if os.path.exists(out_K):
    print(f"K={K} already exists, skipping.")
else:
    print(f"\nRunning TARP RMSD K={K} (n_workers={N_WORKERS}) ...")
    f = run_tarp_eval(complex_names, results_index, DATA_DIR,
                      K=K, mode="rmsd", seed=42, verbose=True, n_workers=N_WORKERS)
    np.save(out_K, f)
    print(f"Saved {out_K}  shape={f.shape}")

# --- TARP RMSD K=1 ---
out_K1 = f"{METRICS}/tarp_fractions_rmsd_K1.npy"
if os.path.exists(out_K1):
    print("K=1 already exists, skipping.")
else:
    print(f"\nRunning TARP RMSD K=1 (n_workers={N_WORKERS}) ...")
    f = run_tarp_eval(complex_names, results_index, DATA_DIR,
                      K=1, mode="rmsd", seed=42, verbose=True, n_workers=N_WORKERS)
    np.save(out_K1, f)
    print(f"Saved {out_K1}  shape={f.shape}")

# --- MIRA scores (sym-corrected RMSD via spyrmsd) ---
out_mira_scores = f"{METRICS}/mira_scores_symrmsd.npy"
out_mira_names  = f"{METRICS}/mira_names_symrmsd.npy"
if os.path.exists(out_mira_scores):
    print("MIRA scores already exist, skipping.")
else:
    print(f"\nRunning MIRA (metric=symrmsd, num_runs=100, n_workers={N_WORKERS}) ...")
    mira_names, mira_scores = compute_mira_scores(
        complex_names, results_index, DATA_DIR,
        num_runs=100, verbose=True, metric="symrmsd", n_workers=N_WORKERS,
    )
    np.save(out_mira_names,  mira_names)
    np.save(out_mira_scores, mira_scores)
    print(f"Saved MIRA scores ({len(mira_scores)} complexes)  "
          f"mean={mira_scores.mean():.4f}")

# --- RMSD accuracy ---
out_acc = f"{METRICS}/rmsd_accuracy.npz"
if os.path.exists(out_acc):
    print("RMSD accuracy already exists, skipping.")
else:
    print("\nRunning RMSD accuracy ...")
    acc_names, acc_min, acc_fracs = compute_rmsd_accuracy(
        complex_names, results_index, DATA_DIR,
        thresholds=(2.0, 5.0), verbose=True,
    )
    np.savez(out_acc, names=acc_names, min_rmsds=acc_min, fracs=acc_fracs)
    print(f"Saved RMSD accuracy ({len(acc_names)} complexes)  "
          f"<2Å: {(acc_min < 2.0).mean()*100:.1f}%  "
          f"<5Å: {(acc_min < 5.0).mean()*100:.1f}%")

print("\nDone.")
