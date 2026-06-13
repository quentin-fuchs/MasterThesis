#!/bin/bash
#SBATCH --job-name=dd_sens_f12
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/dd_sens_f12_%A_%a.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/dd_sens_f12_%A_%a.err

# Rerun inference for the 12 complexes that failed ESM loading in the original
# sensitivity run (posebusters_esm2.pt has wrong sequence lengths for these).
# Uses posebusters_esm2_failed.pt which was recomputed with correct residue counts.
#
# Array: 0=sde_10, 1=sde_50. Results go directly into the existing output dirs
# alongside the 88 already-completed complexes.

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work
OUT_ROOT=$RDS/results/DiffDock/sensitivity_ode_nsteps_v2

COND_NAMES=(sde_10 sde_50)
STEPS_ARR=( 10     50    )

COND=${COND_NAMES[$SLURM_ARRAY_TASK_ID]}
STEPS=${STEPS_ARR[$SLURM_ARRAY_TASK_ID]}

echo "Condition: $COND  steps=$STEPS"

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"

python evaluate.py \
    --config configs/sensitivity_inference_args.yaml \
    --dataset posebusters \
    --data_dir $RDS/data/posebusters_benchmark_set \
    --split_path data/splits/pb_sensitivity_failed12.txt \
    --cache_path $RDS/data/cache_pb_rebuild \
    --out_dir "$OUT_ROOT/$COND" \
    --esm_embeddings_path $RDS/data/embeddings/posebusters_esm2_failed.pt \
    --samples_per_complex 40 \
    --batch_size 40 \
    --chain_cutoff 10 \
    --protein_file protein \
    --ligand_file ligands \
    --save_predictions \
    --num_workers 4 \
    --inference_steps "$STEPS"
