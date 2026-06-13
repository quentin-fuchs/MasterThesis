#!/bin/bash
#SBATCH --job-name=dd_sens_cache
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/dd_sens_cache_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/dd_sens_cache_%j.err

# Build the graph cache for the sensitivity study's random 100-complex PB subset.
# Run this first, then submit diffdock_sensitivity_ode_nsteps.sh with:
#   sbatch --dependency=afterok:<this_job_id> ~/slurm/DiffDock/diffdock_sensitivity_ode_nsteps.sh

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p logs

echo "Building cache for pb_sensitivity_rand100 (100 complexes)..."

python evaluate.py \
    --config default_inference_args.yaml \
    --dataset posebusters \
    --data_dir $RDS/posebusters_benchmark_set \
    --split_path data/splits/pb_sensitivity_rand100.txt \
    --cache_path $RDS/cache_pb_rebuild \
    --out_dir results/cache_build_tmp \
    --esm_embeddings_path $RDS/embeddings/posebusters_esm2.pt \
    --chain_cutoff 10 \
    --protein_file protein \
    --ligand_file ligands \
    --no_model \
    --num_workers 4

echo "Cache build complete."
