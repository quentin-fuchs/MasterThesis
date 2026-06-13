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
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_eval_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_eval_%j.err

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data
OUT_DIR=${1:-$DIFFDOCK_DIR/results/testset_eval}
SPLIT_PATH=${2:-data/splits/timesplit_test}
DATA_DIR=${3:-$RDS/PDBBind_processed}

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p logs

python evaluate.py \
    --config default_inference_args.yaml \
    --dataset pdbbind \
    --data_dir "$DATA_DIR" \
    --split_path "$SPLIT_PATH" \
    --cache_path $RDS/cache_torsion \
    --out_dir "$OUT_DIR" \
    --esm_embeddings_path $RDS/embeddings/pdbbind_esm2.pt \
    --samples_per_complex 40 \
    --batch_size 40 \
    --save_predictions \
    --num_workers 4
