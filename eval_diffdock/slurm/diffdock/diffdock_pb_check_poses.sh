#!/bin/bash
#SBATCH --job-name=dd-pb-checks
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/pb_check_poses_%A_%a.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/pb_check_poses_%A_%a.err

# PoseBusters stereochemical checks on DiffDock rank1-5 poses.
# 6 array tasks × ~51 complexes each = 305 complexes total.
# Stores pb_checks.json per complex for use with compute_mixed_score().
# Uses the sigmadock env where posebusters is installed.
#
# Run alongside or after diffdock_gnina_rescore.sh.
# Aggregate with: python analysis/collect_gnina_rescoring.py

DIFFDOCK_DIR="/home/qf226/MProject/DiffDock"

cd "${DIFFDOCK_DIR}" || exit 1
mkdir -p logs

source ~/.bashrc
conda activate sigmadock || { echo "ERROR: could not activate sigmadock env"; exit 1; }

echo "Array task ${SLURM_ARRAY_TASK_ID}: PB checks chunk ${SLURM_ARRAY_TASK_ID} of 6"

python analysis/pb_check_poses.py --chunk_idx "${SLURM_ARRAY_TASK_ID}"
