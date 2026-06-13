#!/bin/bash -l
#SBATCH --job-name=sigmadock-sanity
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --array=0-39
#SBATCH --output=/home/qf226/MProject/sigmadock/logs/sanity_%A_%a.out
#SBATCH --error=/home/qf226/MProject/sigmadock/logs/sanity_%A_%a.err

PROJECT_DIR="/home/qf226/MProject/sigmadock"
CKPT="${PROJECT_DIR}/checkpoints/sample_checkpoint_0.ckpt"
DATA_CSV="${PROJECT_DIR}/notebooks/dummy_data/redock_10.csv"
OUTPUT_DIR="${PROJECT_DIR}/results/sanity_check"

cd "${PROJECT_DIR}" || exit 1
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sigmadock || { echo "ERROR: could not activate sigmadock env"; exit 1; }

python scripts/sample.py \
  ckpt="${CKPT}" \
  inference.inference_datafront="${DATA_CSV}" \
  run_tag="sanity_check" \
  output_dir="${OUTPUT_DIR}" \
  seed="${SLURM_ARRAY_TASK_ID}" \
  num_seeds=1 \
  data.batch_size=10 \
  data.num_workers=4 \
  hardware.devices=auto \
  postprocessing.scoring=null \
  postprocessing.bust_config=null
