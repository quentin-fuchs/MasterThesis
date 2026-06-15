#!/bin/bash -l
#SBATCH --job-name=sigmadock-mira-tarp
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/qf226/MProject/thesis/logs/mira_tarp_%j.out
#SBATCH --error=/home/qf226/MProject/thesis/logs/mira_tarp_%j.err

# MIRA + TARP evaluation for SigmaDock PoseBusters predictions.
#
# Usage:
#   sbatch ~/slurm/sigmadock/mira_tarp.sh [results_dir] [--mode centroid|rmsd] [--K 100]
#
# Defaults to the 307-complex merged run.  Pass extra args after results_dir
# to override TARP mode or K, e.g.:
#   sbatch ~/slurm/sigmadock/mira_tarp.sh /path/to/results --mode rmsd --K 200

PROJECT_DIR="/home/qf226/MProject/thesis"
DATA_DIR="/home/qf226/rds/hpc-work/data/sigmadock_pb/posebusters_paper/posebusters_benchmark_set"

RESULTS_DIR="${1:-/home/qf226/rds/hpc-work/results/SigmaDock/sigmadock_pb_308}"
shift || true

cd "${PROJECT_DIR}" || exit 1
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sigmadock || { echo "ERROR: sigmadock env not found."; exit 1; }

python eval_sigmadock/scripts/run_mira_tarp.py \
  "${RESULTS_DIR}" \
  --data-dir "${DATA_DIR}" \
  --K 100 \
  --mode centroid \
  --n-bootstrap 200 \
  "$@"
