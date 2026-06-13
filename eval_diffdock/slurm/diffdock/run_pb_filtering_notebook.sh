#!/bin/bash
#SBATCH --job-name=pb_filt_nb
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/pb_filt_nb_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/pb_filt_nb_%j.err

set -euo pipefail

NOTEBOOK=/home/qf226/MProject/DiffDock/notebooks/pb_filtering_analysis.ipynb

source /home/qf226/miniconda3/etc/profile.d/conda.sh
conda activate diffdock

cd /home/qf226/MProject/DiffDock

jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 \
    "$NOTEBOOK"

echo "Notebook execution complete: $NOTEBOOK"
