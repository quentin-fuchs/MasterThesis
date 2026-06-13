#!/bin/bash
#SBATCH --job-name=sd_notebook
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/qf226/MProject/sigmadock/logs/sd_notebook_%j.out
#SBATCH --error=/home/qf226/MProject/sigmadock/logs/sd_notebook_%j.err

NOTEBOOK=${1:-/home/qf226/MProject/sigmadock/notebooks/06_mira_tarp_analysis.ipynb}

source ~/.bashrc
conda activate sigmadock

cd /home/qf226/MProject/sigmadock

jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    "$NOTEBOOK"

echo "Done: $NOTEBOOK"
