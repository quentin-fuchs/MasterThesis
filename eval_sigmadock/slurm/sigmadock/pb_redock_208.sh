#!/bin/bash -l
#SBATCH --job-name=sigmadock-pb208
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-39
#SBATCH --output=/home/qf226/MProject/sigmadock/logs/pb_redock_208_%A_%a.out
#SBATCH --error=/home/qf226/MProject/sigmadock/logs/pb_redock_208_%A_%a.err

# PoseBusters re-docking on the 208 complexes that DiffDock ran on but were not
# included in the paper's 100-complex test set.
# 40 array jobs × 1 seed each = 40 poses per complex.
#
# Whitelist: ~/rds/hpc-work/data/sigmadock_pb/posebusters_paper/diffdock_remaining_208_ids.txt
# Output:    ~/rds/hpc-work/results/sigmadock_pb_208/

PROJECT_DIR="/home/qf226/MProject/sigmadock"
CKPT="${PROJECT_DIR}/checkpoints/sample_checkpoint_0.ckpt"
DATA_DIR="/home/qf226/rds/hpc-work/data/sigmadock_pb"
WHITELIST="${DATA_DIR}/posebusters_paper/diffdock_remaining_208_ids.txt"
OUTPUT_DIR="/home/qf226/rds/hpc-work/results/sigmadock_pb_208"

cd "${PROJECT_DIR}" || exit 1
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sigmadock || { echo "ERROR: could not activate sigmadock env"; exit 1; }

python scripts/sample.py \
  ckpt="${CKPT}" \
  data_dir="${DATA_DIR}" \
  experiment=posebusters \
  experiments.sdf_regex=".*ligands\.sdf$" \
  +experiments.pdb_regex=".*_protein\.pdb$" \
  output_dir="${OUTPUT_DIR}" \
  seed="${SLURM_ARRAY_TASK_ID}" \
  num_seeds=1 \
  data.batch_size=8 \
  data.num_workers=4 \
  data.blacklist="${WHITELIST}" \
  graph.sample_conformer=true \
  graph.fragmentation_strategy=canonical \
  hardware.devices=auto \
  postprocessing.scoring=vinardo \
  postprocessing.bust_config=redock
