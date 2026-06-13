#!/bin/bash
#SBATCH --job-name=diffdock_eval_pb
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_eval_pb_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_eval_pb_%j.err

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data
OUT_DIR=${1:-$DIFFDOCK_DIR/results/pb_evaluate_out}
SPLIT_PATH=${2:-$DIFFDOCK_DIR/data/splits/posebusters_pdb_set_correct.txt}

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p logs

python evaluate.py \
    --config default_inference_args.yaml \
    --dataset posebusters \
    --data_dir $RDS/posebusters_benchmark_set \
    --split_path "$SPLIT_PATH" \
    --cache_path $RDS/cache_pb_rebuild \
    --out_dir "$OUT_DIR" \
    --esm_embeddings_path $RDS/embeddings/posebusters_esm2.pt \
    --samples_per_complex 40 \
    --batch_size 40 \
    --chain_cutoff 10 \
    --protein_file protein \
    --ligand_file ligands \
    --save_predictions \
    --num_workers 4
