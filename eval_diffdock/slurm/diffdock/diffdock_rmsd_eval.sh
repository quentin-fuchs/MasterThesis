#!/bin/bash
#SBATCH --job-name=diffdock_rmsd_eval
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/qf226/MProject/thesis/logs/diffdock_rmsd_eval_%j.out
#SBATCH --error=/home/qf226/MProject/thesis/logs/diffdock_rmsd_eval_%j.err

# Usage:
#   sbatch ~/slurm/DiffDock/diffdock_rmsd_eval.sh [pdbbind|posebusters]
# Default: posebusters

DATASET="${1:-posebusters}"
THESIS_DIR=/home/qf226/MProject/thesis
RDS=/home/qf226/rds/hpc-work

source ~/.bashrc
conda activate diffdock
export PYTHONPATH="$THESIS_DIR:$PYTHONPATH"

cd "$THESIS_DIR"

if [ "$DATASET" = "pdbbind" ]; then
    echo "=== PDBBind test set ==="
    python eval_diffdock/scripts/run_rmsd_eval.py \
        --results_dir $RDS/results/DiffDock/pdbbind_testset/poses \
        --data_dir    $RDS/data/PDBBind_processed \
        --out_dir     $RDS/results/DiffDock/pdbbind_testset/metrics \
        --max_samples 40
elif [ "$DATASET" = "posebusters" ]; then
    echo "=== PoseBusters benchmark ==="
    python eval_diffdock/scripts/run_rmsd_eval.py \
        --results_dir $RDS/results/DiffDock/pb_evaluate_v2_merged \
        --data_dir    $RDS/data/posebusters_benchmark_set \
        --out_dir     $RDS/results/DiffDock/pb_evaluate_v2_merged/metrics \
        --max_samples 40
else
    echo "Unknown dataset '$DATASET'. Use 'pdbbind' or 'posebusters'." >&2
    exit 1
fi
