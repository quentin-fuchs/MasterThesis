#!/bin/bash
# Test version of diffdock_top1_array.sh — 9 complexes from pdbbind_test_9.txt.
# Uses standard PDBBind_processed and 650M ESM embeddings (3B set deleted).
#
# Usage:
#   sbatch ~/slurm/diffdock_top1_array_test.sh

#SBATCH --job-name=diffdock_top1_test
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-39
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_top1_test_%A_%a.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_top1_test_%A_%a.err

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data
OUT_DIR=${DIFFDOCK_DIR}/results/top1_runs_test/run_${SLURM_ARRAY_TASK_ID}

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p "$OUT_DIR"

echo "[run ${SLURM_ARRAY_TASK_ID}/39] output → ${OUT_DIR}"

python evaluate.py \
    --config default_inference_args.yaml \
    --dataset pdbbind \
    --data_dir $RDS/PDBBind_processed \
    --split_path data/splits/pdbbind_test_9.txt \
    --cache_path $RDS/cache_torsion \
    --out_dir "$OUT_DIR" \
    --esm_embeddings_path $RDS/embeddings/pdbbind_esm2.pt \
    --samples_per_complex 10 \
    --batch_size 10 \
    --no_final_step_noise \
    --save_predictions \
    --num_workers 4
