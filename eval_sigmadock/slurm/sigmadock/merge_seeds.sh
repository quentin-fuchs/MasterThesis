#!/bin/bash -l
#SBATCH --job-name=sigmadock-merge
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/qf226/MProject/thesis/logs/merge_%j.out
#SBATCH --error=/home/qf226/MProject/thesis/logs/merge_%j.err

# Merge 100-complex and 208-complex seed directories into a single 308-complex run.

PROJECT_DIR="/home/qf226/MProject/thesis"
BASE="/home/qf226/rds/hpc-work/results/SigmaDock"

DIR1="${BASE}/sigmadock_pb/results/posebusters/sample_checkpoint_0"
DIR2="${BASE}/sigmadock_pb_208/results/posebusters/sample_checkpoint_0"
OUTPUT="${BASE}/sigmadock_pb_308"

cd "${PROJECT_DIR}" || exit 1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sigmadock || { echo "ERROR: could not activate sigmadock env"; exit 1; }

python eval_sigmadock/scripts/merge_seeds.py \
  --dir1  "${DIR1}" \
  --dir2  "${DIR2}" \
  --output "${OUTPUT}"
