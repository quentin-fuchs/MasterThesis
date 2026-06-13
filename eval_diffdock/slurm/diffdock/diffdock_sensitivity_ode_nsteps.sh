#!/bin/bash
#SBATCH --job-name=dd_sens
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/dd_sens_%A_%a.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/dd_sens_%A_%a.err

# ODE diagnostic + n_steps sensitivity on a random 100-complex PB subset.
# Split: data/splits/pb_sensitivity_rand100.txt (100 random from 308, seed=42)
#
# Two orthogonal questions, kept isolated:
#   n_steps sensitivity: vary steps in SDE mode (sde_10, sde_50)
#   ODE diagnostic:      fix steps=20, remove stochasticity (ode_20)
#
# Baseline (sde_20) reuses existing results:
#   ~/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged/
#
# Build cache first, then submit with dependency:
#   cache_id=$(sbatch --parsable ~/slurm/DiffDock/diffdock_sensitivity_build_cache.sh)
#   sbatch --dependency=afterok:${cache_id} ~/slurm/DiffDock/diffdock_sensitivity_ode_nsteps.sh

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work
OUT_ROOT=$RDS/results/DiffDock/sensitivity_ode_nsteps_v2

COND_NAMES=(sde_10  sde_50  ode_20 )
STEPS_ARR=( 10      50      20     )
ODE_ARR=(   ""      ""      "--ode" )

COND=${COND_NAMES[$SLURM_ARRAY_TASK_ID]}
STEPS=${STEPS_ARR[$SLURM_ARRAY_TASK_ID]}
ODE_FLAG=${ODE_ARR[$SLURM_ARRAY_TASK_ID]}

echo "Condition: $COND  steps=$STEPS  ode='$ODE_FLAG'"

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p logs "$OUT_ROOT/$COND"

python evaluate.py \
    --config configs/sensitivity_inference_args.yaml \
    --dataset posebusters \
    --data_dir $RDS/data/posebusters_benchmark_set \
    --split_path data/splits/pb_sensitivity_rand100.txt \
    --cache_path $RDS/data/cache_pb_rebuild \
    --out_dir "$OUT_ROOT/$COND" \
    --esm_embeddings_path $RDS/data/embeddings/posebusters_esm2.pt \
    --samples_per_complex 40 \
    --batch_size 40 \
    --chain_cutoff 10 \
    --protein_file protein \
    --ligand_file ligands \
    --save_predictions \
    --num_workers 4 \
    --inference_steps "$STEPS" \
    $ODE_FLAG
