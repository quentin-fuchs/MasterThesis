#!/bin/bash -l
#SBATCH --job-name=sigmadock-rescore
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-39
#SBATCH --output=/home/qf226/MProject/sigmadock/logs/rescore_%A_%a.out
#SBATCH --error=/home/qf226/MProject/sigmadock/logs/rescore_%A_%a.err

# Post-hoc Vinardo rescoring for all 40 seeds of the PoseBusters run.
# One array task per seed; each scores 100 complexes with gnina and saves
# rescoring.pt alongside the existing predictions.pt and posebusters.pt.

PROJECT_DIR="/home/qf226/MProject/sigmadock"
MODEL_DIR="/home/qf226/rds/hpc-work/results/sigmadock_pb/results/posebusters/sample_checkpoint_0"
SEED_DIR="${MODEL_DIR}/seed_${SLURM_ARRAY_TASK_ID}"

cd "${PROJECT_DIR}" || exit 1
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sigmadock || { echo "ERROR: could not activate sigmadock env"; exit 1; }

# Vinardo is CPU-only; cap threads so we don't fight other jobs on the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Rescoring seed ${SLURM_ARRAY_TASK_ID}: ${SEED_DIR}"
python scripts/gnina_rescore.py "${SEED_DIR}"
