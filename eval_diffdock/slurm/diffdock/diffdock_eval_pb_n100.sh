#!/bin/bash
#SBATCH --job-name=dd_pb_n100
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/dd_pb_n100_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/dd_pb_n100_%j.err

# Experiment C: 100 samples per complex on chunk_0 (62 complexes).
# Goal: check if oracle (best-of-100) rises toward 50%, diagnosing whether
# the gap to the paper is a sample-quantity problem or a model quality ceiling.

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data
SPLIT_PATH=${1:-$DIFFDOCK_DIR/data/splits/pb_chunks/chunk_0.txt}
OUT_DIR=${2:-/home/qf226/rds/hpc-work/results/pb_evaluate_out/n100_chunk0}

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p "$OUT_DIR"

python evaluate.py \
    --config default_inference_args.yaml \
    --dataset posebusters \
    --data_dir $RDS/posebusters_benchmark_set \
    --split_path "$SPLIT_PATH" \
    --cache_path $RDS/cache_pb_fresh_torsion \
    --out_dir "$OUT_DIR" \
    --esm_embeddings_path $RDS/embeddings/posebusters_esm2.pt \
    --samples_per_complex 100 \
    --batch_size 40 \
    --chain_cutoff 10 \
    --protein_file protein \
    --ligand_file ligands \
    --save_predictions \
    --num_workers 4
