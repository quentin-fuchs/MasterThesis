#!/bin/bash
#SBATCH --job-name=diffdock_group_eval
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/group_eval_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/group_eval_%j.err
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Usage:
#   sbatch ~/slurm/diffdock_group_eval.sh pdbbind
#   sbatch ~/slurm/diffdock_group_eval.sh posebusters

DATASET="${1:-pdbbind}"
WORKDIR="/home/qf226/MProject/DiffDock"
RDS=/home/qf226/rds/hpc-work/data
N_WORKERS=14    # leave 2 CPUs free for OS / IO

source ~/.bashrc
conda activate diffdock

cd "$WORKDIR"
mkdir -p logs

if [ "$DATASET" = "pdbbind" ]; then
    echo "[group_eval] Running on PDBBind test set"
    python analysis/run_group_eval.py \
        --complex_names_npy /home/qf226/rds/hpc-work/results/DiffDock/pdbbind_testset/metrics/complex_names.npy \
        --results_dir       /home/qf226/rds/hpc-work/results/DiffDock/pdbbind_testset/poses \
        --data_dir          $RDS/PDBBind_processed \
        --out_dir           /home/qf226/rds/hpc-work/results/DiffDock/pdbbind_testset/metrics/group_eval \
        --K 100 \
        --seed 42 \
        --n_workers "$N_WORKERS"

elif [ "$DATASET" = "posebusters" ]; then
    echo "[group_eval] Running on PoseBusters benchmark (pb_evaluate_v2_merged)"
    python analysis/run_group_eval.py \
        --complex_names_npy /home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged/metrics/complex_names.npy \
        --results_dir       /home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged \
        --data_dir          $RDS/posebusters_benchmark_set \
        --out_dir           /home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged/metrics/group_eval \
        --K 100 \
        --seed 42 \
        --n_workers "$N_WORKERS"

else
    echo "Unknown dataset '$DATASET'. Use 'pdbbind' or 'posebusters'." >&2
    exit 1
fi

echo "[group_eval] Done."
