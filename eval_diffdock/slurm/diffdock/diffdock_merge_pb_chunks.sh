#!/bin/bash
#SBATCH --job-name=merge_pb_chunks
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/qf226/MProject/thesis/logs/merge_pb_chunks_%j.out
#SBATCH --error=/home/qf226/MProject/thesis/logs/merge_pb_chunks_%j.err

THESIS_DIR=/home/qf226/MProject/thesis
RDS_RESULTS=/home/qf226/rds/hpc-work/results
CHUNKS_DIR=$RDS_RESULTS/pb_evaluate_v2
OUT_DIR=$RDS_RESULTS/pb_evaluate_v2_merged

source ~/.bashrc
conda activate diffdock

cd "$THESIS_DIR"

# Expose chunk_failed to the merge script as chunk_5
# (chunk_extra120 is excluded — it belongs to a separate 428-complex investigation)
ln -sfn "$CHUNKS_DIR/chunk_failed" "$CHUNKS_DIR/chunk_5"

# Use a temporary staging dir containing only chunks 0-5 so the merge script's
# int-sort doesn't trip over chunk_extra120
STAGE=$(mktemp -d)
for i in 0 1 2 3 4 5; do
    ln -s "$CHUNKS_DIR/chunk_$i" "$STAGE/chunk_$i"
done

python eval_diffdock/scripts/merge_eval_chunks.py "$STAGE" "$OUT_DIR"

rm -rf "$STAGE"
rm -f "$CHUNKS_DIR/chunk_5"
