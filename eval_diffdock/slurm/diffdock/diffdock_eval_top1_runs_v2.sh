#!/bin/bash
#SBATCH --job-name=diffdock_eval_v2
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_eval_v2_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_eval_v2_%j.err

# Compute TARP RMSD (K=10) and MIRA scores for the two merged run directories:
#   top1_runs_v2_top1_merged/  — 10 poses per complex (rank-1 from each run)
#   top1_runs_v2_top3_merged/  — 30 poses per complex (rank 1-3 from each run)
# Results are saved to metrics/ inside each directory, ready for tarp_analysis.ipynb.

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work
N_WORKERS=14

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"

python - <<'EOF'
import sys, warnings, numpy as np
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import os
N_WORKERS = int(os.environ.get("N_WORKERS", 14))

from utils.tarp_eval import build_results_index, run_tarp_eval
from utils.mira_eval import compute_mira_scores, mira_null

RDS          = "/home/qf226/rds/hpc-work"
RESULTS_FULL = f"{RDS}/results/DiffDock/pdbbind_testset/raw_chunks"
TOP1_MERGED  = f"{RDS}/results/DiffDock/top1_runs_v2_top1_merged"
TOP3_MERGED  = f"{RDS}/results/DiffDock/top1_runs_v2_top3_merged"
DATA_DIR     = f"{RDS}/data/PDBBind_processed"
K            = 10

results_index = build_results_index(RESULTS_FULL)
index_top1    = build_results_index(TOP1_MERGED)
index_top3    = build_results_index(TOP3_MERGED)

common     = sorted(set(results_index) & set(index_top1) & set(index_top3))
common_arr = np.array(common)
print(f"Common complexes: {len(common_arr)}")

# ── TARP RMSD K=10 ────────────────────────────────────────────────────────────
for label, idx, out_dir in [
    ("top1_merged",      index_top1,    TOP1_MERGED),
    ("top3_merged",      index_top3,    TOP3_MERGED),
    ("baseline_common",  results_index, TOP1_MERGED),
]:
    out_path = f"{out_dir}/metrics/tarp_fractions_rmsd_K{K}"
    if label == "baseline_common":
        out_path = f"{TOP1_MERGED}/metrics/tarp_fractions_rmsd_K{K}_baseline"
    out_path += ".npy"

    if os.path.exists(out_path):
        print(f"  {label}: TARP RMSD already cached, skipping.")
        continue

    print(f"\nTARP RMSD K={K} — {label}...")
    f = run_tarp_eval(
        common_arr, idx, DATA_DIR,
        K=K, mode="rmsd", seed=42, verbose=True, n_workers=N_WORKERS,
    )
    np.save(out_path, f)
    print(f"  Saved: {out_path}  shape={f.shape}")

# ── MIRA scores ───────────────────────────────────────────────────────────────
for label, idx, out_dir, S in [
    ("top1_merged",     index_top1,    TOP1_MERGED, 10),
    ("top3_merged",     index_top3,    TOP3_MERGED, 30),
    ("baseline_common", results_index, TOP1_MERGED, 40),
]:
    out_path = f"{out_dir}/metrics/mira_scores.npy"
    if label == "baseline_common":
        out_path = f"{TOP1_MERGED}/metrics/mira_baseline_common.npy"

    if os.path.exists(out_path):
        print(f"  {label}: MIRA already cached, skipping.")
        continue

    print(f"\nMIRA (S={S}) — {label}...")
    _, scores = compute_mira_scores(
        common_arr, idx, DATA_DIR,
        num_runs=100, verbose=True, metric="rmsd",
    )
    np.save(out_path, scores)
    ref = mira_null(S=S)
    print(f"  Saved: {out_path}  mean={scores.mean():.4f}  ref={ref:.4f}")

print("\nAll evaluations complete.")
EOF
