"""
PoseBusters pose-validity filtering for the PoseBusters benchmark set.

Evaluates all DiffDock predicted poses for the ~305-complex PoseBusters
benchmark using PoseBusters 'dock' mode (geometry + protein-clash checks,
no reference crystal pose required). Results are cached as JSON.

Standalone script — does not import from the thesis package so it can run
in the posebusters conda env (which lacks torch).

Run via SLURM:
    sbatch ~/slurm/DiffDock/diffdock_pb_eval_posebusters.sh
"""

import json
import warnings
import sys
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

RESULTS_DIR = "/home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged"
DATA_DIR    = "/home/qf226/rds/hpc-work/data/posebusters_benchmark_set"
METRICS     = f"{RESULTS_DIR}/metrics"
CACHE       = f"{METRICS}/posebusters_results_pb.json"

complex_names = sorted(
    d.name for d in Path(RESULTS_DIR).iterdir()
    if d.is_dir() and any(d.glob("rank*.sdf"))
)
print(f"Complexes with predictions: {len(complex_names)}")

# Load partial cache so the job is resumable
cached = {}
if Path(CACHE).exists():
    with open(CACHE) as f:
        cached = json.load(f)
    print(f"Loaded partial cache: {len(cached)} complexes already done")

todo = [n for n in complex_names if n not in cached]
print(f"Remaining: {len(todo)}")

from posebusters import PoseBusters
buster = PoseBusters(config="dock")
results = dict(cached)
skipped = 0

for i, pdb_id in enumerate(todo):
    if i % 20 == 0:
        print(f"  [{i}/{len(todo)}] {pdb_id} ...", flush=True)

    complex_dir  = Path(RESULTS_DIR) / pdb_id
    protein_file = Path(DATA_DIR) / pdb_id / f"{pdb_id}_protein.pdb"

    if not complex_dir.exists() or not protein_file.exists():
        print(f"    Skipping {pdb_id}: missing dir or protein file")
        skipped += 1
        continue

    plain = [f for f in complex_dir.iterdir()
             if f.name.startswith("rank") and f.name.endswith(".sdf")
             and "_confidence" not in f.name]
    if len(plain) > 1:
        rank_files = sorted(plain, key=lambda f: int(f.stem.replace("rank", "")))
    else:
        rank_files = sorted(
            [f for f in complex_dir.iterdir()
             if f.name.startswith("rank") and "_confidence" in f.name
             and f.name.endswith(".sdf")],
            key=lambda f: int(f.name.split("_confidence")[0].replace("rank", "")),
        )

    if not rank_files:
        skipped += 1
        continue

    valid_ranks, check_failures, n_processed = [], {}, 0
    for sdf_file in rank_files:
        try:
            df = buster.bust(mol_pred=str(sdf_file), mol_cond=str(protein_file))
            bool_cols = [c for c in df.columns if df[c].dtype == bool]
            if not bool_cols or df.empty:
                continue
            if bool(df[bool_cols].all(axis=1).iloc[0]):
                valid_ranks.append(sdf_file.name)
            for col in bool_cols:
                if not bool(df[col].iloc[0]):
                    check_failures[col] = check_failures.get(col, 0) + 1
            n_processed += 1
        except Exception:
            pass

    results[pdb_id] = {
        "valid_ranks": valid_ranks,
        "n_total": n_processed or len(rank_files),
        "check_failures": check_failures,
    }

    # Save incrementally every 50 complexes
    if (i + 1) % 50 == 0:
        with open(CACHE, "w") as f:
            json.dump(results, f)
        print(f"  Checkpoint saved ({len(results)} complexes)", flush=True)

with open(CACHE, "w") as f:
    json.dump(results, f)

n_valid_arr = np.array([len(v["valid_ranks"]) for v in results.values()])
total_valid = int(n_valid_arr.sum())
total_poses = sum(v["n_total"] for v in results.values())
n_any = int((n_valid_arr > 0).sum())

print(f"\nComplexes with ≥1 PB-valid pose: {n_any}/{len(results)} ({100*n_any/len(results):.1f}%)")
print(f"Overall pass rate: {total_valid}/{total_poses} = {100*total_valid/total_poses:.1f}%")
print(f"Per-complex — mean: {n_valid_arr.mean():.1f}  median: {np.median(n_valid_arr):.0f}  min: {n_valid_arr.min()}  max: {n_valid_arr.max()}")
print(f"Skipped: {skipped}")
print(f"Saved to {CACHE}")
