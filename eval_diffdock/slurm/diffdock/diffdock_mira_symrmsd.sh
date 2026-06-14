#!/bin/bash
#SBATCH --job-name=diffdock_mira_symrmsd
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --output=/home/qf226/MProject/thesis/diffdock/logs/diffdock_mira_symrmsd_%j.out
#SBATCH --error=/home/qf226/MProject/thesis/diffdock/logs/diffdock_mira_symrmsd_%j.err

# Compute MIRA scores with symmetry-corrected RMSD (spyrmsd) for the full
# DiffDock PoseBusters benchmark (308 complexes, 40 samples each).
#
# Output (saved to METRICS_DIR):
#   mira_names_symrmsd.npy   — PDB IDs of successfully evaluated complexes
#   mira_scores_symrmsd.npy  — corresponding per-complex MIRA scores

THESIS_DIR=/home/qf226/MProject/thesis
RDS=/home/qf226/rds/hpc-work
export RESULTS_DIR=$RDS/results/DiffDock/pb_evaluate_v2_merged/poses
export METRICS_DIR=$RDS/results/DiffDock/pb_evaluate_v2_merged/metrics
export DATA_DIR=$RDS/data/posebusters_benchmark_set

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$THESIS_DIR:$PYTHONPATH"

cd "$THESIS_DIR/diffdock"
mkdir -p logs

echo "=== DiffDock PoseBusters — MIRA symRMSD evaluation ==="
echo "Results dir : $RESULTS_DIR"
echo "Data dir    : $DATA_DIR"
echo "Metrics out : $METRICS_DIR"
echo "Workers     : 8"
echo "Started     : $(date)"
echo

python - <<'EOF'
import sys, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')

import os
RESULTS_DIR = os.environ['RESULTS_DIR']
METRICS_DIR = os.environ['METRICS_DIR']
DATA_DIR    = os.environ['DATA_DIR']
THESIS_DIR  = os.environ.get('PYTHONPATH', '').split(':')[0]

from eval_diffdock.loader import build_results_index
from eval_diffdock.mira_runner import compute_mira_scores
from molcalib.mira import mira_null

# Load complex names from existing metrics
complex_names = np.load(f"{METRICS_DIR}/complex_names.npy", allow_pickle=True)
print(f"Loaded {len(complex_names)} complex names.", flush=True)

results_index = build_results_index(RESULTS_DIR)
print(f"Results index: {len(results_index)} complexes found.", flush=True)

# Keep only names present in the results index
complex_names = [n for n in complex_names if n in results_index]
print(f"Evaluating {len(complex_names)} complexes (symRMSD metric, 8 workers)...\n",
      flush=True)

names, scores = compute_mira_scores(
    complex_names,
    results_index,
    DATA_DIR,
    num_runs=20,
    metric="symrmsd",
    seed=42,
    verbose=True,
    n_workers=8,
)

# Save
out_names  = f"{METRICS_DIR}/mira_names_symrmsd.npy"
out_scores = f"{METRICS_DIR}/mira_scores_symrmsd.npy"
np.save(out_names,  names)
np.save(out_scores, scores)

S = 40
print(f"\n=== Results ===")
print(f"Complexes evaluated : {len(scores)}")
print(f"Complexes skipped   : {len(complex_names) - len(scores)}")
print(f"Mean MIRA (symrmsd) : {scores.mean():.4f}")
print(f"Null reference S={S} : {mira_null(S):.4f}")
print(f"Saved names  → {out_names}")
print(f"Saved scores → {out_scores}")
EOF

echo
echo "Finished: $(date)"
