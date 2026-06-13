#!/bin/bash -l
#SBATCH --job-name=sigmadock-tarp-rmsd
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/qf226/MProject/sigmadock/logs/mira_tarp_rmsd_%j.out
#SBATCH --error=/home/qf226/MProject/sigmadock/logs/mira_tarp_rmsd_%j.err

# TARP RMSD evaluation for SigmaDock PoseBusters predictions.
# Skips MIRA (uses cached scores from mira_tarp.npz).
# Saves fractions to mira_tarp_rmsd.npz alongside the centroid results.

PROJECT_DIR="/home/qf226/MProject/sigmadock"
DATA_DIR="/home/qf226/rds/hpc-work/data/sigmadock_pb/posebusters_paper/posebusters_benchmark_set"
RESULTS_DIR="/home/qf226/rds/hpc-work/results/SigmaDock/sigmadock_pb_308"

cd "${PROJECT_DIR}" || exit 1
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sigmadock || { echo "ERROR: sigmadock env not found."; exit 1; }

python scripts/mira_tarp_eval.py \
  "${RESULTS_DIR}" \
  --data-dir "${DATA_DIR}" \
  --K 10 \
  --mode rmsd \
  --n-bootstrap 200 \
  --skip-mira \
  --output "${RESULTS_DIR}/mira_tarp_rmsd.npz"
