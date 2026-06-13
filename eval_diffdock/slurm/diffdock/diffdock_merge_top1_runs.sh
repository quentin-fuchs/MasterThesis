#!/bin/bash
#SBATCH --job-name=diffdock_merge_top1
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_merge_top1_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_merge_top1_%j.err

# Merge top1_runs_v2 into two combined pose directories:
#   top1_runs_v2_top1_merged/  — rank1 from each of the 10 runs = 10 poses per complex
#   top1_runs_v2_top3_merged/  — rank1-3 from each run          = 30 poses per complex

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS_RESULTS=/home/qf226/rds/hpc-work/results/DiffDock
RUNS_DIR=${RDS_RESULTS}/top1_runs_v2

source ~/.bashrc
conda activate diffdock

cd "$DIFFDOCK_DIR"

python utils/merge_top1_runs.py \
    --runs_dir  "${RUNS_DIR}" \
    --out_prefix "${RDS_RESULTS}/top1_runs_v2"
