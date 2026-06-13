#!/bin/bash
#SBATCH --job-name=diffdock_pb
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_pb_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_pb_%j.err

# DiffDock inference on the PoseBusters benchmark set (308 complexes, 40 samples each).
# Requires the ESM embeddings job to have completed first:
#   data/posebusters_esm2_embeddings.pt

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p logs

python inference.py \
    --config default_inference_args.yaml \
    --protein_ligand_csv $RDS/inference/posebusters_inference.csv \
    --out_dir results/posebusters_inference \
    --esm_embeddings_path $RDS/embeddings/posebusters_esm2.pt \
    --samples_per_complex 40 \
    --batch_size 40
