#!/bin/bash
#SBATCH --job-name=dd_pb_n10
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/dd_pb_n10_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/dd_pb_n10_%j.err

# Evaluate DiffDock-L on 100 random PoseBusters complexes with 10 samples.

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data
RDS_RESULTS=/home/qf226/rds/hpc-work/results

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p "$RDS_RESULTS/pb_evaluate_out/n10_100"

python -m evaluate \
    --config default_inference_args.yaml \
    --dataset posebusters \
    --data_dir $RDS/posebusters_benchmark_set \
    --split_path data/splits/pb_100_random.txt \
    --esm_embeddings_path $RDS/embeddings/posebusters_esm2.pt \
    --chain_cutoff 10 \
    --batch_size 10 \
    --samples_per_complex 10 \
    --protein_file protein \
    --ligand_file ligands \
    --out_dir "$RDS_RESULTS/pb_evaluate_out/n10_100" \

    --num_workers 4
