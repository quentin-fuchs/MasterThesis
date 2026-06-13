#!/bin/bash
# Submit parallelized DiffDock evaluation over the test set.
#
# Usage:
#   bash diffdock_eval_parallel.sh [n_chunks] [out_dir] [split_path] [time_limit] [data_dir]
#
# Defaults:
#   n_chunks   6
#   out_dir    results/testset_eval
#   split_path data/splits/timesplit_test
#   time_limit 05:00:00
#   data_dir   data/PDBBind_processed

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS=/home/qf226/rds/hpc-work/data
N_CHUNKS=${1:-6}
OUT_DIR=${2:-$DIFFDOCK_DIR/results/testset_eval}
SPLIT_PATH=${3:-$DIFFDOCK_DIR/data/splits/timesplit_test}
TIME_LIMIT=${4:-05:00:00}
DATA_DIR=${5:-$RDS/PDBBind_processed}
CHUNKS_DIR=$DIFFDOCK_DIR/data/splits/chunks

mkdir -p "$CHUNKS_DIR"

# Split the test set into N chunk files
python3 -c "
import math
with open('$SPLIT_PATH') as f:
    ids = [l.strip() for l in f if l.strip()]
n = $N_CHUNKS
size = math.ceil(len(ids) / n)
for i in range(n):
    chunk = ids[i*size:(i+1)*size]
    if not chunk:
        continue
    with open(f'$CHUNKS_DIR/chunk_{i}.txt', 'w') as f:
        f.write('\n'.join(chunk) + '\n')
    print(f'Chunk {i}: {len(chunk)} complexes')
"

# Submit one job per chunk
for chunk_file in "$CHUNKS_DIR"/chunk_*.txt; do
    idx=$(basename "$chunk_file" .txt | sed 's/chunk_//')
    chunk_out="$OUT_DIR/chunk_$idx"
    mkdir -p "$chunk_out"
    job_id=$(sbatch --parsable \
        --time="$TIME_LIMIT" \
        ~/slurm/diffdock_eval_testset.sh \
        "$chunk_out" \
        "$chunk_file" \
        "$DATA_DIR")
    echo "Submitted chunk $idx → job $job_id ($(wc -l < "$chunk_file") complexes, time=$TIME_LIMIT) → $chunk_out"
done
