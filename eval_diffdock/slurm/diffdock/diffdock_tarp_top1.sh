#!/bin/bash
#SBATCH --job-name=diffdock_tarp_top1
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_tarp_top1_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_tarp_top1_%j.err

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
export RDS=/home/qf226/rds/hpc-work/data

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"

echo "Running TARP top-1 RMSD evaluation (K=20, rank1 only)..."

python - <<'EOF'
import sys, warnings, os
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt

from utils.tarp_eval import (
    build_results_index, run_tarp_eval,
    ecp_from_fractions, bootstrap_ecp, plot_ecp, atc_score
)

MERGED      = "results/testset_eval_merged"
RESULTS_DIR = "results/testset_eval_full"
DATA_DIR    = os.path.join(os.environ["RDS"], "PDBBind_processed")

complex_names = np.load(f"{MERGED}/complex_names.npy", allow_pickle=True)
results_index = build_results_index(RESULTS_DIR)

f_top1 = run_tarp_eval(
    complex_names, results_index, DATA_DIR,
    K=20, mode="rmsd", seed=42, verbose=True, n_workers=8,
    _max_samples=1,
)

out = f"{MERGED}/tarp_fractions_top1_rmsd_K20.npy"
np.save(out, f_top1)

ecp, alpha = ecp_from_fractions(f_top1, n_bins=50)
boot = bootstrap_ecp(f_top1, n_bootstrap=500)

print(f"\n=== TARP top-1 results (K=20) ===")
print(f"Complexes contributing: {len(f_top1)}")
print(f"ATC score: {atc_score(ecp, alpha):+.4f}")
print(f"Saved to {out}")

fig, ax = plt.subplots(figsize=(5, 5))
plot_ecp(ecp, alpha, ax=ax, label="DiffDock-L top-1", color="C2", bootstrap_ecps=boot)
ax.set_title("TARP — Top-1 sample (pose calibration)")
plt.tight_layout()
plt.savefig(f"{MERGED}/tarp_ecp_top1.png", dpi=150, bbox_inches='tight')
print("Plot saved to tarp_ecp_top1.png")
EOF
