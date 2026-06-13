#!/bin/bash
# Submit parallelized DiffDock evaluate.py over the PoseBusters benchmark set.
#
# Usage:
#   bash ~/slurm/diffdock_eval_pb_parallel.sh [n_chunks] [out_dir] [time_limit]
#
# Defaults:
#   n_chunks   5    (308 IDs → ~62 per chunk)
#   out_dir    results/pb_evaluate_out
#   time_limit 06:00:00
#
# Each chunk job builds its own graph cache (~62 complexes, a few minutes)
# then runs inference. Logs: logs/diffdock_eval_pb_<jobid>.out/err

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
RDS_RESULTS=/home/qf226/rds/hpc-work/results
N_CHUNKS=${1:-5}
OUT_DIR=${2:-$RDS_RESULTS/pb_evaluate_v2}
TIME_LIMIT=${3:-06:00:00}
SPLIT_PATH=$DIFFDOCK_DIR/data/splits/posebusters_pdb_set_correct.txt
CHUNKS_DIR=$DIFFDOCK_DIR/data/splits/pb_chunks

mkdir -p "$CHUNKS_DIR"

# Split the 308-complex set into N chunk files
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

# Submit one GPU job per chunk
for chunk_file in "$CHUNKS_DIR"/chunk_*.txt; do
    idx=$(basename "$chunk_file" .txt | sed 's/chunk_//')
    chunk_out="$OUT_DIR/chunk_$idx"
    mkdir -p "$chunk_out"
    job_id=$(sbatch --parsable \
        --time="$TIME_LIMIT" \
        ~/slurm/DiffDock/diffdock_eval_pb_testset.sh \
        "$chunk_out" \
        "$chunk_file")
    echo "Submitted chunk $idx → job $job_id ($(wc -l < "$chunk_file") complexes, time=$TIME_LIMIT) → $chunk_out"
done
