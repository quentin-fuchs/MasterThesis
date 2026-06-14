"""
Pre-compute TARP RMSD fractions for the PDBBind test set.

Saves tarp_fractions_rmsd_K{K}.npy and tarp_fractions_rmsd_K1.npy to the
metrics dir so tarp_analysis.ipynb can load them from cache.

Run via SLURM:
    sbatch ~/slurm/thesis/run_tarp_rmsd_pdbbind.sh
"""

import os
import sys
import numpy as np

sys.path.insert(0, "/home/qf226/MProject/thesis")
sys.path.insert(0, "/home/qf226/MProject/thesis/diffdock")

from eval_diffdock.loader import build_results_index
from eval_diffdock.tarp_runner import run_tarp_eval

RESULTS_FULL = "/home/qf226/rds/hpc-work/results/DiffDock/pdbbind_testset/raw_chunks"
MERGED       = "/home/qf226/rds/hpc-work/results/DiffDock/pdbbind_testset"
METRICS      = os.path.join(MERGED, "metrics")
DATA_DIR     = "/home/qf226/rds/hpc-work/data/PDBBind_processed"

N_WORKERS = int(os.environ.get("N_WORKERS", 1))

complex_names = np.load(f"{METRICS}/complex_names.npy", allow_pickle=True)
results_index = build_results_index(RESULTS_FULL)
print(f"Complexes: {len(complex_names)}, indexed: {len(results_index)}")

# --- TARP RMSD K=10 (phase 2 of tarp_analysis notebook) ---
K = 10
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

# --- TARP RMSD K=1 (protein-family panel in tarp_analysis notebook) ---
K1 = 1
out_path_k1 = f"{METRICS}/tarp_fractions_rmsd_K{K1}.npy"
if os.path.exists(out_path_k1):
    print(f"K={K1} already exists, skipping.")
else:
    print(f"\nRunning TARP RMSD K={K1} (n_workers={N_WORKERS}) ...")
    f_rmsd_k1 = run_tarp_eval(
        complex_names, results_index, DATA_DIR,
        K=K1, mode="rmsd", seed=42, verbose=True, n_workers=N_WORKERS,
    )
    np.save(out_path_k1, f_rmsd_k1)
    print(f"Saved {out_path_k1}  shape={f_rmsd_k1.shape}")

print("\nDone.")
