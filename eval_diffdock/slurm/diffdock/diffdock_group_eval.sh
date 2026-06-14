#!/bin/bash
#SBATCH --job-name=diffdock_group_eval
#SBATCH --output=/home/qf226/MProject/thesis/logs/group_eval_%j.out
#SBATCH --error=/home/qf226/MProject/thesis/logs/group_eval_%j.err
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Usage:
#   sbatch ~/slurm/DiffDock/diffdock_group_eval.sh pdbbind
#   sbatch ~/slurm/DiffDock/diffdock_group_eval.sh posebusters

DATASET="${1:-pdbbind}"
THESIS_DIR="/home/qf226/MProject/thesis"
RDS=/home/qf226/rds/hpc-work
N_WORKERS=14    # leave 2 CPUs free for OS / IO

source ~/.bashrc
conda activate diffdock
export PYTHONPATH="$THESIS_DIR:$PYTHONPATH"

cd "$THESIS_DIR"

if [ "$DATASET" = "pdbbind" ]; then
    echo "[group_eval] Running on PDBBind test set"
    python eval_diffdock/scripts/run_group_eval.py \
        --complex_names_npy $RDS/results/DiffDock/pdbbind_testset/metrics/complex_names.npy \
        --results_dir       $RDS/results/DiffDock/pdbbind_testset/poses \
        --data_dir          $RDS/data/PDBBind_processed \
        --out_dir           $RDS/results/DiffDock/pdbbind_testset/metrics/group_eval \
        --K 100 \
        --seed 42 \
        --n_workers "$N_WORKERS"

elif [ "$DATASET" = "posebusters" ]; then
    echo "[group_eval] Running on PoseBusters benchmark (pb_evaluate_v2_merged)"
    python eval_diffdock/scripts/run_group_eval.py \
        --complex_names_npy $RDS/results/DiffDock/pb_evaluate_v2_merged/metrics/complex_names.npy \
        --results_dir       $RDS/results/DiffDock/pb_evaluate_v2_merged \
        --data_dir          $RDS/data/posebusters_benchmark_set \
        --out_dir           $RDS/results/DiffDock/pb_evaluate_v2_merged/metrics/group_eval \
        --K 100 \
        --seed 42 \
        --n_workers "$N_WORKERS"

else
    echo "Unknown dataset '$DATASET'. Use 'pdbbind' or 'posebusters'." >&2
    exit 1
fi

echo "[group_eval] Done."
