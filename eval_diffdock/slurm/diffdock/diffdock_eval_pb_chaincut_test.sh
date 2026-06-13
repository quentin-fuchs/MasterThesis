#!/bin/bash
#SBATCH --job-name=dd_pb_chaincut_test
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/dd_pb_chaincut_test_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/dd_pb_chaincut_test_%j.err

# Test run: 50 complexes with manually-preprocessed chain-cutoff protein files.
# --chain_cutoff in evaluate.py only affects the cache dir name, not preprocessing.
# Chain cutoff is applied by using protein_chaincut.pdb (pre-cut at 10 Å from ligand)
# with matching chaincut ESM2 embeddings.

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data
OUT_DIR=${1:-/home/qf226/rds/hpc-work/results/pb_chaincut_test50}

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
    --split_path data/splits/pb_chaincut_test50.txt \
    --cache_path $RDS/cache_pb_chaincut_torsion \
    --out_dir "$OUT_DIR" \
    --esm_embeddings_path $RDS/embeddings/posebusters_esm2_chaincut.pt \
    --samples_per_complex 40 \
    --batch_size 40 \
    --protein_file protein_chaincut \
    --ligand_file ligands \
    --save_predictions \
    --num_workers 4
