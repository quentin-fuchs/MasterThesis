#!/bin/bash
#SBATCH --job-name=dd_sens_ode
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/dd_sens_ode_%A.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/dd_sens_ode_%A.err

# Rerun of ode_20 condition from the sensitivity analysis (job 30133415_2).
# Previous run crashed with "tr_z referenced before assignment" because the
# temperature scaling block in sampling.py references tr_z unconditionally,
# but tr_z is only assigned in the SDE branch. Fix: pass temp_sampling=1.0
# via CLI so the override block is never entered (condition != 1.0 is False).
# The shared yaml (sensitivity_inference_args.yaml) is left unchanged so
# sde_10/sde_50 are unaffected if rerun.
#
# Output dir is reused: ode_20/ only contained broken aggregate .npy files
# (all 10000 sentinels), no per-complex SDFs. evaluate.py overwrites them.

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work
OUT_ROOT=$RDS/results/DiffDock/sensitivity_ode_nsteps_v2

echo "Condition: ode_20  steps=20  ode='--ode'  temp_sampling=1.0"

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"
mkdir -p logs "$OUT_ROOT/ode_20"

python evaluate.py \
    --config configs/sensitivity_inference_args_ode.yaml \
    --dataset posebusters \
    --data_dir $RDS/data/posebusters_benchmark_set \
    --split_path data/splits/pb_sensitivity_rand100.txt \
    --cache_path $RDS/data/cache_pb_rebuild \
    --out_dir "$OUT_ROOT/ode_20/poses" \
    --esm_embeddings_path $RDS/data/embeddings/posebusters_esm2.pt \
    --samples_per_complex 40 \
    --batch_size 40 \
    --chain_cutoff 10 \
    --protein_file protein \
    --ligand_file ligands \
    --save_predictions \
    --num_workers 4 \
    --inference_steps 20 \
    --ode
