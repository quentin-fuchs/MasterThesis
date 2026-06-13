#!/bin/bash
#SBATCH --job-name=diffdock
#SBATCH --account=fergusson-sl3-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/diffdock_%j.out
#SBATCH --error=logs/diffdock_%j.err

# Usage: sbatch diffdock_inference.sh <csv_file> [out_dir]
CSV=${1:?Usage: sbatch diffdock_inference.sh <csv_file> [out_dir]}
OUT_DIR=${2:-results/batch_inference}

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /home/qf226/MProject/DiffDock
mkdir -p logs

python inference.py \
    --config default_inference_args.yaml \
    --protein_ligand_csv "$CSV" \
    --out_dir "$OUT_DIR"
