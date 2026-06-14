"""
PoseBusters pose-validity filtering for the PDBBind test set.

Evaluates all DiffDock predicted poses (40 per complex, 322 complexes) using
PoseBusters 'dock' mode (geometry + protein-clash checks, no reference crystal
pose required). Results are cached as JSON and loaded by tarp_analysis.ipynb.

Run via SLURM:
    sbatch ~/slurm/thesis/run_pb_eval_pdbbind.sh
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

POSES_DIR = "/home/qf226/rds/hpc-work/results/DiffDock/pdbbind_testset/poses"
DATA_DIR  = "/home/qf226/rds/hpc-work/data/PDBBind_processed"
METRICS   = "/home/qf226/rds/hpc-work/results/DiffDock/pdbbind_testset/metrics"
CACHE     = f"{METRICS}/posebusters_results.json"

complex_names = np.load(f"{METRICS}/complex_names.npy", allow_pickle=True).tolist()
print(f"Complexes: {len(complex_names)}")

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

    complex_dir  = Path(POSES_DIR) / pdb_id
    protein_file = Path(DATA_DIR) / pdb_id / f"{pdb_id}_protein_processed.pdb"

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

n_valid_all = sum(len(v["valid_ranks"]) for v in results.values())
n_total_all = sum(v["n_total"] for v in results.values())
n_any = sum(1 for v in results.values() if v["valid_ranks"])
print(f"\nComplexes with ≥1 PB-valid pose: {n_any}/{len(results)} ({100*n_any/len(results):.1f}%)")
print(f"Overall pass rate: {n_valid_all}/{n_total_all} = {100*n_valid_all/n_total_all:.1f}%")
print(f"Skipped: {skipped}")
print(f"Saved to {CACHE}")
