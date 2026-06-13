#!/bin/bash
#SBATCH --job-name=diffdock_rmsd_eval
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_rmsd_eval_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_rmsd_eval_%j.err

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data

source ~/.bashrc
conda activate diffdock
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p logs

echo "=== PoseBusters test set ==="
python analysis/run_rmsd_eval.py \
    --results_dir results/posebusters_inference \
    --data_dir    $RDS/posebusters_benchmark_set \
    --out_dir     results/posebusters_inference/metrics \
    --max_samples 40
