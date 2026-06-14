"""Update complex_names.npy for pb_evaluate_v2_merged to include all 305 complexes."""
import sys
sys.path.insert(0, "/home/qf226/MProject/DiffDock")
from pathlib import Path
import numpy as np

PB_DIR      = "/home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged"
RESULTS_DIR = f"{PB_DIR}/poses"
METRICS_DIR = f"{PB_DIR}/metrics"

complex_names = sorted(
    d.name for d in Path(RESULTS_DIR).iterdir()
    if d.is_dir() and any(d.glob("rank*.sdf"))
)
print(f"Found {len(complex_names)} complexes with predictions in {RESULTS_DIR}")

old = np.load(f"{METRICS_DIR}/complex_names.npy")
print(f"Old complex_names.npy: {len(old)} entries")

np.save(f"{METRICS_DIR}/complex_names.npy", np.array(complex_names))
print(f"Saved updated complex_names.npy: {len(complex_names)} entries → {METRICS_DIR}/complex_names.npy")
