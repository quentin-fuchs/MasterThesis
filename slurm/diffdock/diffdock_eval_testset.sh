#!/bin/bash
#SBATCH --job-name=diffdock_eval
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/home/qf226/MProject/thesis/logs/diffdock_eval_%j.out
#SBATCH --error=/home/qf226/MProject/thesis/logs/diffdock_eval_%j.err

THESIS_DIR=/home/qf226/MProject/thesis
DIFFDOCK_DIR=$THESIS_DIR/diffdock
RDS=/home/qf226/rds/hpc-work
OUT_DIR=${1:-$RDS/results/DiffDock/pdbbind_testset}
SPLIT_PATH=${2:-data/splits/timesplit_test}
DATA_DIR=${3:-$RDS/data/PDBBind_processed}

source ~/.bashrc
conda activate diffdock_inference
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

mkdir -p "$THESIS_DIR/logs"
cd "$DIFFDOCK_DIR"

python evaluate.py \
    --config default_inference_args.yaml \
    --dataset pdbbind \
    --data_dir "$DATA_DIR" \
    --split_path "$SPLIT_PATH" \
    --cache_path "$RDS/data/cache" \
    --out_dir "$OUT_DIR" \
    --esm_embeddings_path "$RDS/data/embeddings/pdbbind_esm2.pt" \
    --samples_per_complex 40 \
    --batch_size 40 \
    --save_predictions \
    --num_workers 4
