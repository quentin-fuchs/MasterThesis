"""
Standalone runner for PoseBusters pose validity filtering.

Run via SLURM: sbatch ~/slurm/diffdock_posebusters.sh
"""

import sys
import numpy as np

from eval_diffdock.pb_eval import run_posebusters

RDS       = "/home/qf226/rds/hpc-work"
MERGED    = f"{RDS}/results/DiffDock/pdbbind_testset"
DATA_DIR  = f"{RDS}/data/PDBBind_processed"
CACHE     = f"{MERGED}/posebusters_results.json"

complex_names = np.load(f"{MERGED}/complex_names.npy", allow_pickle=True)

run_posebusters(
    complex_names,
    results_dir=MERGED,
    data_dir=DATA_DIR,
    config="dock",
    cache_path=CACHE,
    verbose=True,
)
