#!/bin/bash
#SBATCH --job-name=pb_filter
#SBATCH --account=mphil-dis-sl2-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/posebusters_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/posebusters_%j.err

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /home/qf226/MProject/DiffDock
mkdir -p logs

python analysis/run_posebusters.py
