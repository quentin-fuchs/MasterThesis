#!/bin/bash
#SBATCH --job-name=diffdock_6uvp
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
    --protein_path $RDS/PDBBind_processed/6uvp/6uvp_protein_processed.pdb \
    --ligand_description $RDS/PDBBind_processed/6uvp/6uvp_ligand.sdf \
    --out_dir results/6uvp_100samples \
    --samples_per_complex 100 \
    --batch_size 10 \
    --no_final_step_noise
