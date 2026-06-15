#!/bin/bash
#SBATCH --job-name=dd_sens_grp
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/dd_sens_grp_%A_%a.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/dd_sens_grp_%A_%a.err

# Per-group (translation / rotation / torsion) TARP and MIRA for the
# n_steps sensitivity study (sde_10 / sde_20 / sde_50).
#
# sde_20 baseline complexes must be copied first:
#   python scripts/copy_sde20_complexes.py
#
# Then submit this array:
#   sbatch ~/slurm/DiffDock/diffdock_sensitivity_group_eval.sh

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work
SENS_ROOT=$RDS/results/DiffDock/sensitivity_ode_nsteps_v2
DATA_DIR=$RDS/data/posebusters_benchmark_set
WHITELIST=$DIFFDOCK_DIR/data/splits/pb_sensitivity_rand100.txt

CONDS=(sde_10 sde_20 sde_50 ode_20)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

POSES_DIR=$SENS_ROOT/$COND/poses
OUT_DIR=$SENS_ROOT/$COND/metrics/group_eval

echo "Condition: $COND"
echo "Poses:     $POSES_DIR"
echo "Output:    $OUT_DIR"

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p logs "$OUT_DIR"

# Save whitelist as .npy so run_group_eval/run_group_mira can read it
NAMES_NPY=$OUT_DIR/complex_names_input.npy
python - <<PYEOF
import numpy as np
names = open("$WHITELIST").read().split()
np.save("$NAMES_NPY", np.array(names))
print(f"Saved {len(names)} complex names to $NAMES_NPY")
PYEOF

echo ""
echo "=== Per-group TARP + distances ==="
python analysis/run_group_eval.py \
    --complex_names_npy "$NAMES_NPY" \
    --results_dir       "$POSES_DIR" \
    --data_dir          "$DATA_DIR" \
    --out_dir           "$OUT_DIR" \
    --K 100 \
    --seed 42 \
    --n_workers 30

echo ""
echo "=== Per-group MIRA ==="
python analysis/run_group_mira.py \
    --complex_names_npy "$NAMES_NPY" \
    --results_dir       "$POSES_DIR" \
    --data_dir          "$DATA_DIR" \
    --out_dir           "$OUT_DIR" \
    --num_runs 100 \
    --seed 42 \
    --n_workers 30

echo ""
echo "Done: $COND"
