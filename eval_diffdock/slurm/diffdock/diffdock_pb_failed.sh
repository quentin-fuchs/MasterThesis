#!/bin/bash
#SBATCH --job-name=diffdock_pb_failed
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_pb_failed_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_pb_failed_%j.err

# Inference for the 24 posebusters complexes that failed in job 29514348 due to
# tensor size mismatches caused by modified residues. Runs ESM on the fly (no
# precomputed embeddings) so sequences are parsed by ProDy, matching the graph
# construction in new_extract_receptor_structure.

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
    --protein_ligand_csv $RDS/inference/posebusters_failed_inference.csv \
    --out_dir results/posebusters_inference_v2 \
    --samples_per_complex 40 \
    --batch_size 40
