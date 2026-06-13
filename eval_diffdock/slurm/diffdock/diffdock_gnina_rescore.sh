#!/bin/bash
#SBATCH --job-name=dd-gnina-rescore
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/gnina_rescore_%A_%a.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/gnina_rescore_%A_%a.err

# Post-hoc Vinardo rescoring of DiffDock rank1-5 poses for the PoseBusters benchmark.
# 6 array tasks × ~51 complexes each = 305 complexes total.
# gnina is CPU-only (--no_gpu); uses the sigmadock env where gnina is installed.
#
# Output: <results_dir>/<complex_id>/rescoring_vinardo.json per complex.
# Aggregate with: python analysis/collect_gnina_rescoring.py

DIFFDOCK_DIR="/home/qf226/MProject/DiffDock"

cd "${DIFFDOCK_DIR}" || exit 1
mkdir -p logs

source ~/.bashrc
conda activate sigmadock || { echo "ERROR: could not activate sigmadock env"; exit 1; }

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Array task ${SLURM_ARRAY_TASK_ID}: rescoring chunk ${SLURM_ARRAY_TASK_ID} of 6"

python analysis/gnina_rescore_poses.py --chunk_idx "${SLURM_ARRAY_TASK_ID}"
