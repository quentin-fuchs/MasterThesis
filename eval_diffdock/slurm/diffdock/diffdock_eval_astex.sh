#!/bin/bash
#SBATCH --job-name=dd_eval_astex
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/dd_eval_astex_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/dd_eval_astex_%j.err

# Evaluate DiffDock-L on the Astex Diverse Set (84 complexes, 40 samples).
# Step 1: generate ESM2-650M embeddings for Astex proteins.
# Step 2: run evaluate.py with same flags as the PoseBusters job.

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
ESM_REPO=/home/qf226/MProject/facebookresearch_esm
RDS=/home/qf226/rds/hpc-work/data
OUT_DIR=/home/qf226/rds/hpc-work/results/astex_eval

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p "$OUT_DIR"

# ── Step 1: ESM2 embeddings ────────────────────────────────────────────────────
echo "=== Step 1: ESM2 embeddings for Astex Diverse Set ==="

python datasets/esm_embedding_preparation.py \
    --dataset pdbbind \
    --data_dir $RDS/astex_diverse_set \
    --out_file $RDS/sequences/astex.fasta

python "$ESM_REPO/scripts/extract.py" \
    esm2_t33_650M_UR50D \
    $RDS/sequences/astex.fasta \
    $RDS/astex_esm_output \
    --repr_layers 33 \
    --include per_tok \
    --truncation_seq_length 4096

python datasets/esm_embeddings_to_pt.py \
    --esm_embeddings_path $RDS/astex_esm_output \
    --output_path $RDS/embeddings/astex_esm2.pt

echo "ESM embeddings saved to $RDS/embeddings/astex_esm2.pt"

# ── Step 2: evaluate.py ────────────────────────────────────────────────────────
echo "=== Step 2: DiffDock-L evaluation on Astex ==="

python evaluate.py \
    --config default_inference_args.yaml \
    --dataset posebusters \
    --data_dir $RDS/astex_diverse_set \
    --split_path $RDS/astex_diverse_set/astex_diverse_set_ids.txt \
    --cache_path $RDS/cache_astex_torsion \
    --out_dir "$OUT_DIR" \
    --esm_embeddings_path $RDS/embeddings/astex_esm2.pt \
    --samples_per_complex 40 \
    --batch_size 40 \
    --chain_cutoff 10 \
    --protein_file protein \
    --ligand_file ligands \
    --save_predictions \
    --num_workers 4
