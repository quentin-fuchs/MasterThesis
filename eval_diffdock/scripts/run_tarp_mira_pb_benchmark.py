"""
Pre-compute TARP RMSD fractions and MIRA scores for the PoseBusters benchmark set.

Merges run_tarp_rmsd_pb_benchmark.py and run_mira_pb_benchmark.py into one job
so the results index and SDF files are loaded only once.

Saves to metrics/:
  tarp_fractions_symrmsd_K1.npy  — TARP fractions (n, 1), sym-corrected RMSD
  mira_names_symrmsd.npy      — complex names for MIRA (sym-corrected RMSD)
  mira_scores_symrmsd.npy     — per-complex MIRA scores

Run via SLURM:
    sbatch ~/slurm/thesis/run_tarp_mira_pb_benchmark.sh
"""

import os
import numpy as np

from eval_diffdock.loader import build_results_index
from eval_diffdock.tarp_runner import run_tarp_eval
from eval_diffdock.mira_runner import compute_mira_scores

PB_DIR      = "/home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged"
RESULTS_DIR = os.path.join(PB_DIR, "poses")
DATA_DIR    = "/home/qf226/rds/hpc-work/data/posebusters_benchmark_set"
METRICS     = os.path.join(PB_DIR, "metrics")

N_WORKERS = int(os.environ.get("N_WORKERS", 1))

results_index = build_results_index(RESULTS_DIR)
complex_names = np.array(sorted(results_index.keys()))
print(f"Complexes: {len(complex_names)}")

# --- TARP RMSD K=1 ---
out_K1 = f"{METRICS}/tarp_fractions_symrmsd_K1.npy"
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

print("\nDone.")
