#!/bin/bash
#SBATCH --job-name=pb_filter_pb
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/qf226/MProject/thesis/logs/pb_eval_posebusters_%j.out
#SBATCH --error=/home/qf226/MProject/thesis/logs/pb_eval_posebusters_%j.err

# PoseBusters validity filtering for the PoseBusters benchmark set.

THESIS_DIR=/home/qf226/MProject/thesis

source ~/.bashrc
conda activate posebusters
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$THESIS_DIR:$PYTHONPATH"

cd "$THESIS_DIR"

python eval_diffdock/scripts/run_pb_eval_posebusters.py
