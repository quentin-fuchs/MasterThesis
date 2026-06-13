#!/bin/bash -l
#SBATCH --job-name=sigmadock-rmsd
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/home/qf226/MProject/sigmadock/logs/rmsd_%j.out
#SBATCH --error=/home/qf226/MProject/sigmadock/logs/rmsd_%j.err

PROJECT_DIR="/home/qf226/MProject/sigmadock"
RESULTS_DIR="${1:-${PROJECT_DIR}/results/sanity_check/results/sanity_check/sample_checkpoint_0}"
shift || true

cd "${PROJECT_DIR}" || exit 1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sigmadock || { echo "ERROR: could not activate sigmadock env"; exit 1; }

python scripts/eval_rmsd.py "${RESULTS_DIR}" "$@"
