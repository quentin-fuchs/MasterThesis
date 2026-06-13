"""
PoseBusters pose-validity filtering for the PoseBusters benchmark set.

Evaluates all DiffDock predicted poses for the 303-complex PoseBusters
benchmark using PoseBusters 'dock' mode (geometry + protein-clash checks,
no reference crystal pose required). Results are cached as JSON.

Run via SLURM:
    sbatch ~/slurm/diffdock_pb_eval_posebusters.sh
"""

import sys
import numpy as np

sys.path.insert(0, "/home/qf226/MProject/DiffDock")

from utils.posebusters_eval import run_posebusters

RESULTS_DIR = "/home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged"
DATA_DIR    = "/home/qf226/rds/hpc-work/data/posebusters_benchmark_set"
CACHE       = f"{RESULTS_DIR}/metrics/posebusters_results_pb.json"

from pathlib import Path

complex_names = sorted(
    d.name for d in Path(RESULTS_DIR).iterdir()
    if d.is_dir() and any(d.glob("rank*.sdf"))
)
print(f"Complexes with predictions: {len(complex_names)}")

pb_results = run_posebusters(
    complex_names,
    results_dir=RESULTS_DIR,
    data_dir=DATA_DIR,
    config="dock",
    cache_path=CACHE,
    verbose=True,
    protein_suffix="_protein.pdb",
)

n_valid = np.array([len(v["valid_ranks"]) for v in pb_results.values()])
total_valid = int(n_valid.sum())
total_poses = sum(v["n_total"] for v in pb_results.values())
n_any = int((n_valid > 0).sum())

print(f"\nComplexes with ≥1 PB-valid pose: {n_any}/{len(pb_results)} ({100*n_any/len(pb_results):.1f}%)")
print(f"Overall pass rate:               {total_valid}/{total_poses} = {100*total_valid/total_poses:.1f}%")
print(f"Per-complex — mean: {n_valid.mean():.1f}  median: {np.median(n_valid):.0f}  min: {n_valid.min()}  max: {n_valid.max()}")
print(f"Saved to {CACHE}")
