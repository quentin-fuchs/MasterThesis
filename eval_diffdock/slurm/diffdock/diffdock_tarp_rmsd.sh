#!/bin/bash
#SBATCH --job-name=diffdock_tarp
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_tarp_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_tarp_%j.err

# Run full RMSD-based TARP evaluation over the 322-complex PDBBind test set.
#
# Usage:
#   sbatch diffdock_tarp_rmsd.sh [K]
#
# Default K=100 (reference points per complex).

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data
K=${1:-100}

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"

echo "Running TARP RMSD evaluation (K=$K)..."

python - <<EOF
import sys, warnings, numpy as np
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

from utils.tarp_eval import (
    build_results_index, run_tarp_eval,
    ecp_from_fractions, atc_score
)

complex_names = np.load("results/testset_eval_merged/complex_names.npy", allow_pickle=True)
results_index = build_results_index("results/testset_eval_full")

f_rmsd = run_tarp_eval(
    complex_names, results_index, "${RDS}/PDBBind_processed",
    K=$K, mode="rmsd", seed=42, verbose=True, n_workers=8
)
out = f"results/testset_eval_merged/tarp_fractions_rmsd_K$K.npy"
np.save(out, f_rmsd)

ecp, alpha = ecp_from_fractions(f_rmsd, n_bins=50)
print(f"\n=== TARP RMSD results (K=$K) ===")
print(f"Complexes contributing: {len(f_rmsd)}")
print(f"ATC score: {atc_score(ecp, alpha):+.4f}")
print(f"Mean coverage fraction: {f_rmsd[~np.isnan(f_rmsd)].mean():.4f}")
print(f"Saved to {out}")
EOF
