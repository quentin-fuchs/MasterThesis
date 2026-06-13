#!/bin/bash -l
#SBATCH --job-name=sigmadock-blind
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-39
#SBATCH --output=/home/qf226/MProject/sigmadock/logs/blind_%A_%a.out
#SBATCH --error=/home/qf226/MProject/sigmadock/logs/blind_%A_%a.err

PROJECT_DIR="/home/qf226/MProject/sigmadock"
CKPT="${PROJECT_DIR}/checkpoints/sample_checkpoint_0.ckpt"
DATA_CSV="${PROJECT_DIR}/notebooks/dummy_data/redock_10.csv"
OUTPUT_DIR="${PROJECT_DIR}/results/blind_docking"

cd "${PROJECT_DIR}" || exit 1
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sigmadock || { echo "ERROR: could not activate sigmadock env"; exit 1; }

# pocket_distance_cutoff=100 includes the whole protein in the pocket graph.
# pocket_com_cutoff=100 centres the diffusion on the full protein CoM rather
# than the binding site — no pocket knowledge is used.
# Reduced batch_size because the protein graph is much larger.
python scripts/sample.py \
  ckpt="${CKPT}" \
  inference.inference_datafront="${DATA_CSV}" \
  run_tag="blind_docking" \
  output_dir="${OUTPUT_DIR}" \
  seed="${SLURM_ARRAY_TASK_ID}" \
  num_seeds=1 \
  data.batch_size=4 \
  data.num_workers=4 \
  hardware.devices=auto \
  graph.pocket_distance_cutoff=100.0 \
  graph.pocket_com_cutoff=100.0 \
  postprocessing.scoring=null \
  postprocessing.bust_config=null
