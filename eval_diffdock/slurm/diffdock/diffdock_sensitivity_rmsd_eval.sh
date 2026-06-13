#!/bin/bash
#SBATCH --job-name=dd_sens_rmsd
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/dd_sens_rmsd_%A_%a.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/dd_sens_rmsd_%A_%a.err

# Flat RMSD evaluation for the n_steps sensitivity study.
# For each condition: computes rmsds.npy + top1_rmsd.npy (via run_rmsd_eval.py),
# then flat RMSD TARP (K=1) and MIRA, cached to metrics/.
#
# Submit after all inference is complete:
#   sbatch ~/slurm/DiffDock/diffdock_sensitivity_rmsd_eval.sh

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work
SENS_ROOT=$RDS/results/DiffDock/sensitivity_ode_nsteps_v2
DATA_DIR=$RDS/data/posebusters_benchmark_set
N_WORKERS=14

CONDS=(sde_10 sde_20 sde_50 ode_20)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

RESULTS_DIR=$SENS_ROOT/$COND
OUT_DIR=$RESULTS_DIR/metrics

echo "Condition: $COND"
echo "Results:   $RESULTS_DIR"
echo "Metrics:   $OUT_DIR"

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p "$OUT_DIR"

echo ""
echo "=== RMSD eval (rmsds.npy, top1_rmsd.npy) ==="
python analysis/run_rmsd_eval.py \
    --results_dir "$RESULTS_DIR" \
    --data_dir    "$DATA_DIR" \
    --out_dir     "$OUT_DIR" \
    --max_samples 40

echo ""
echo "=== Flat RMSD TARP (K=1) and MIRA ==="
python - <<PYEOF
import sys, os, warnings, numpy as np
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

from pathlib import Path
from utils.tarp_eval import run_tarp_eval, ecp_from_fractions
from utils.mira_eval import compute_mira_scores, mira_null

RESULTS_DIR = "$RESULTS_DIR"
DATA_DIR    = "$DATA_DIR"
OUT_DIR     = "$OUT_DIR"
N_WORKERS   = $N_WORKERS

def build_flat_index(results_dir):
    return {
        d.name: d
        for d in sorted(Path(results_dir).iterdir())
        if d.is_dir() and any(d.glob("rank*.sdf"))
    }

results_index = build_flat_index(RESULTS_DIR)
complex_names = np.array(sorted(results_index.keys()))
print(f"Complexes found: {len(complex_names)}")

# ── TARP (K=1, RMSD) ──────────────────────────────────────────────────────────
tarp_path = f"{OUT_DIR}/tarp_fractions_rmsd_K1.npy"
if os.path.exists(tarp_path):
    print("TARP already cached, skipping.")
    f_rmsd = np.load(tarp_path)
else:
    print("Computing TARP RMSD K=1 ...")
    f_rmsd = run_tarp_eval(
        complex_names, results_index, DATA_DIR,
        K=1, mode="rmsd", seed=42, verbose=True, n_workers=N_WORKERS,
    )
    np.save(tarp_path, f_rmsd)
    print(f"Saved: {tarp_path}  shape={f_rmsd.shape}")

# ── MIRA (RMSD) ───────────────────────────────────────────────────────────────
mira_scores_path = f"{OUT_DIR}/mira_scores_rmsd.npy"
mira_names_path  = f"{OUT_DIR}/mira_names_rmsd.npy"
if os.path.exists(mira_scores_path):
    print("MIRA already cached, skipping.")
else:
    print("Computing MIRA RMSD ...")
    mira_names, mira_scores = compute_mira_scores(
        complex_names, results_index, DATA_DIR,
        num_runs=100, verbose=True, metric="rmsd",
    )
    np.save(mira_scores_path, mira_scores)
    np.save(mira_names_path,  mira_names)
    ref = mira_null(S=40)
    print(f"Saved: {mira_scores_path}  mean={mira_scores.mean():.4f}  null={ref:.4f}")

print("Done.")
PYEOF

echo ""
echo "Done: $COND"
