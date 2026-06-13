#!/bin/bash
#SBATCH --job-name=diffdock_failed_esm
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_failed_esm_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_failed_esm_%j.err

# Generate ESM2-650M embeddings for the 27 failed posebusters complexes using
# ProDy-based sequence extraction (same as graph construction) so dimensions
# match the receptor feature tensors — fixing the tensor size mismatch.

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"

python generate_failed_esm_embeddings.py \
    --split_path data/splits/pb_failed_eval.txt \
    --data_dir   $RDS/posebusters_benchmark_set \
    --output_pt  $RDS/embeddings/posebusters_esm2_failed.pt
