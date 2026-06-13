#!/bin/bash
#SBATCH --job-name=diffdock_cache
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_cache_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_cache_%j.err

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"

# Builds the cache used by diffdock_top1_array.sh (RDS path):
#   $RDS/cache_pb_rebuild_torsion/pdbbind3_limit0_INDEXtimesplit_test_*
#
# Uses --no_model so the dataset is instantiated (triggering cache creation)
# without loading or running the neural networks.
# All args that affect the cache name must match diffdock_top1_array.sh exactly:
#   --config, --dataset, --cache_path, --split_path, --esm_embeddings_path

echo "Building torsion cache for timesplit_test (~360 complexes)..."

python evaluate.py \
    --config default_inference_args.yaml \
    --dataset pdbbind \
    --data_dir $RDS/PDBBind_processed \
    --split_path data/splits/timesplit_test \
    --cache_path $RDS/cache_pb_rebuild \
    --out_dir results/cache_build_tmp \
    --esm_embeddings_path $RDS/embeddings/pdbbind_esm2.pt \
    --no_model \
    --num_workers 4

echo "Cache build complete."
