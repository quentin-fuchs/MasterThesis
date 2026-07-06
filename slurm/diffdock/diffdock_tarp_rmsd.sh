#!/bin/bash
#SBATCH --job-name=diffdock_tarp
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/qf226/MProject/thesis/logs/diffdock_tarp_%j.out
#SBATCH --error=/home/qf226/MProject/thesis/logs/diffdock_tarp_%j.err

# Run RMSD-based TARP evaluation over the 322-complex PDBBind test set.
#
# Usage:
#   sbatch diffdock_tarp_rmsd.sh [K]
#
# Default K=10 reference draws per complex.

THESIS_DIR=/home/qf226/MProject/thesis
RDS=/home/qf226/rds/hpc-work
K=${1:-10}

export RESULTS_DIR=$RDS/results/DiffDock/pdbbind_testset/raw_chunks
export METRICS_DIR=$RDS/results/DiffDock/pdbbind_testset/metrics
export DATA_DIR=$RDS/data/PDBBind_processed
export K_VAL=$K

source ~/.bashrc
conda activate analysis
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$THESIS_DIR:$PYTHONPATH"

mkdir -p "$THESIS_DIR/logs"
cd "$THESIS_DIR"

echo "=== DiffDock PDBBind — TARP RMSD evaluation (K=$K) ==="
echo "Results dir : $RESULTS_DIR"
echo "Data dir    : $DATA_DIR"
echo "Metrics out : $METRICS_DIR"
echo "Workers     : 8"
echo "Started     : $(date)"
echo

python - <<'EOF'
import os, warnings
import numpy as np
warnings.filterwarnings('ignore')

RESULTS_DIR = os.environ['RESULTS_DIR']
METRICS_DIR = os.environ['METRICS_DIR']
DATA_DIR    = os.environ['DATA_DIR']
K           = int(os.environ['K_VAL'])

from eval_diffdock.loader import build_results_index
from eval_diffdock.tarp_runner import run_tarp_eval
from molcalib.tarp import ecp_from_fractions

complex_names = np.load(f"{METRICS_DIR}/complex_names.npy", allow_pickle=True)
print(f"Loaded {len(complex_names)} complex names.", flush=True)

results_index = build_results_index(RESULTS_DIR)
print(f"Results index: {len(results_index)} complexes found.", flush=True)

complex_names = [n for n in complex_names if n in results_index]
print(f"Evaluating {len(complex_names)} complexes (K={K}, mode=rmsd, 8 workers)...\n",
      flush=True)

f_matrix = run_tarp_eval(
    complex_names, results_index, DATA_DIR,
    K=K, mode="rmsd", seed=42, verbose=True, n_workers=8,
)

out = f"{METRICS_DIR}/tarp_fractions_symrmsd_K{K}.npy"
np.save(out, f_matrix)

ecp, alpha = ecp_from_fractions(f_matrix)
atc = float(np.trapz(ecp - alpha, alpha))

print(f"\n=== TARP RMSD results (K={K}) ===")
print(f"Complexes contributing : {f_matrix.shape[0]}")
print(f"ATC score              : {atc:+.4f}  (0 = perfect calibration)")
print(f"Mean coverage fraction : {np.nanmean(f_matrix):.4f}")
print(f"Saved to               : {out}")
EOF

echo
echo "Finished: $(date)"
