#!/bin/bash
#SBATCH --job-name=diffdock_1dgb
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diffdock_%j.out
#SBATCH --error=logs/diffdock_%j.err

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$DIFFDOCK_DIR"

python inference.py \
    --config default_inference_args.yaml \
    --protein_path $RDS/DockGen/processed_files/1dgb_1_NDP_1/1dgb_1_NDP_1_protein_processed.pdb \
    --ligand_description $RDS/DockGen/processed_files/1dgb_1_NDP_1/1dgb_1_NDP_1_ligand.pdb \
    --out_dir results/1dgb_1_NDP_1_40samples \
    --samples_per_complex 40 \
    --batch_size 10 \
    --no_final_step_noise
