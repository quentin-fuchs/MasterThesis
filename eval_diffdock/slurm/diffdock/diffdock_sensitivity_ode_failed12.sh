#!/bin/bash
#SBATCH --job-name=dd_sens_ode_f12
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/dd_sens_ode_f12_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/dd_sens_ode_f12_%j.err

# Rerun ode_20 inference for the 12 complexes skipped due to wrong ESM tensor
# sizes in posebusters_esm2.pt. Uses posebusters_esm2_failed.pt instead.
# Submit with: sbatch --dependency=afterok:30169859 this_script.sh

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work
OUT_ROOT=$RDS/results/DiffDock/sensitivity_ode_nsteps_v2

echo "Condition: ode_20  steps=20  ode='--ode'  (failed-12 rerun)"

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"

python evaluate.py \
    --config configs/sensitivity_inference_args_ode.yaml \
    --dataset posebusters \
    --data_dir $RDS/data/posebusters_benchmark_set \
    --split_path data/splits/pb_sensitivity_failed12.txt \
    --cache_path $RDS/data/cache_pb_rebuild \
    --out_dir "$OUT_ROOT/ode_20" \
    --esm_embeddings_path $RDS/data/embeddings/posebusters_esm2_failed.pt \
    --samples_per_complex 40 \
    --batch_size 40 \
    --chain_cutoff 10 \
    --protein_file protein \
    --ligand_file ligands \
    --save_predictions \
    --num_workers 4 \
    --inference_steps 20 \
    --ode
