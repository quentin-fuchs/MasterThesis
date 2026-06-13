#!/bin/bash
#SBATCH --job-name=pb_filter_pb
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/pb_eval_posebusters_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/pb_eval_posebusters_%j.err

source ~/.bashrc
conda activate posebusters
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /home/qf226/MProject/DiffDock
mkdir -p logs

python analysis/run_pb_eval_posebusters.py
