#!/usr/bin/env python
"""
Evaluate DiffDock top-1 and all-poses symmetry-corrected RMSD on a test set.

Saves top1_rmsd.npy (shape n,) and rmsds.npy (shape n, S) to --out_dir,
matching the format already present in the PDBBind test-set metrics directory.

Usage
-----
# PoseBusters (flat results dir — complex dirs at top level):
python analysis/run_rmsd_eval.py \
    --results_dir results/posebusters_inference \
    --data_dir    data/posebusters_benchmark_set \
    --out_dir     results/posebusters_inference/metrics

# PDBBind test set (chunk_* subdirs):
python analysis/run_rmsd_eval.py \
    --results_dir /home/qf226/rds/results/pdbbind_testset/poses \
    --data_dir    data/PDBBind_processed \
    --out_dir     /home/qf226/rds/results/pdbbind_testset/metrics
"""

import argparse
import os
from pathlib import Path

import numpy as np

from eval_diffdock.rmsd_runner import run_rmsd_eval
from eval_diffdock.loader import build_results_index


def main():
    parser = argparse.ArgumentParser(description="Compute DiffDock RMSD accuracy metrics.")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing DiffDock predictions.")
    parser.add_argument("--data_dir", required=True,
                        help="Root data directory containing crystal ligand SDF files.")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to write output .npy files.")
    parser.add_argument("--max_samples", type=int, default=40,
                        help="Number of ranked poses to evaluate per complex (default 40).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Auto-detect flat vs chunk_* structure
    results_index = build_results_index(args.results_dir)
    print(f"Results index: {len(results_index)} complexes")

    complex_names = sorted(results_index.keys())

    names, top1_rmsds, all_rmsds = run_rmsd_eval(
        complex_names, results_index, args.data_dir,
        max_samples=args.max_samples, verbose=True,
    )

    np.save(os.path.join(args.out_dir, "rmsd_names.npy"), names)
    np.save(os.path.join(args.out_dir, "top1_rmsd.npy"), top1_rmsds)
    np.save(os.path.join(args.out_dir, "rmsds.npy"), all_rmsds)
    print(f"\nSaved rmsd_names.npy, top1_rmsd.npy, rmsds.npy → {args.out_dir}")


if __name__ == "__main__":
    main()
