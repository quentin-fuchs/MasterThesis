#!/bin/bash
#SBATCH -J sd_group_eval
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o /home/qf226/MProject/thesis/logs/group_eval_%j.out

RESULTS_DIR="/home/qf226/rds/hpc-work/results/SigmaDock/sigmadock_pb_308"
DATA_DIR="/home/qf226/rds/hpc-work/data/posebusters_benchmark_set"

source /home/qf226/.bashrc
conda activate sigmadock

cd /home/qf226/MProject/thesis

python eval_sigmadock/scripts/group_eval.py \
    "$RESULTS_DIR" \
    --data-dir "$DATA_DIR" \
    --K 100 \
    --num-runs 100 \
    --seed 42
