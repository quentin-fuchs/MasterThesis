#!/bin/bash
#SBATCH --job-name=dd_notebook
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/dd_notebook_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/dd_notebook_%j.err

NOTEBOOK=${1:-/home/qf226/MProject/DiffDock/notebooks/posebusters_calibration.ipynb}

source ~/.bashrc
conda activate diffdock

cd /home/qf226/MProject/DiffDock

jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    "$NOTEBOOK"

echo "Done: $NOTEBOOK"
