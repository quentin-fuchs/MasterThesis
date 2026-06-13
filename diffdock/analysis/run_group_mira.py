"""
Compute per-group (translation / rotation / torsion) MIRA calibration scores
for DiffDock inference results.

MIRA (Sharief et al. 2026, arXiv:2605.02014) draws T random centers from
each group's prior and checks whether the crystal pose falls inside the same
ball as the predicted samples. Under perfect calibration:

    mira_null(S) = (2/3) × (S + 1) / S  ≈ 0.683 for S = 40

Saves to --out_dir:
  mira_scores_translation.npy   (n_valid,)  per-complex scores
  mira_names_translation.npy    (n_valid,)  PDB IDs
  mira_scores_rotation.npy      (n_valid,)
  mira_names_rotation.npy       (n_valid,)
  mira_scores_torsion.npy       (n_valid,)   rigid ligands (0 bonds) excluded
  mira_names_torsion.npy        (n_valid,)

Usage
-----
PDBBind test set:
  python analysis/run_group_mira.py \\
      --complex_names_npy /home/qf226/rds/results/pdbbind_testset/metrics/complex_names.npy \\
      --results_dir       /home/qf226/rds/results/pdbbind_testset/poses \\
      --data_dir          data/PDBBind_processed \\
      --out_dir           /home/qf226/rds/results/pdbbind_testset/metrics/group_eval \\
      --num_runs 100 --n_workers 8

PoseBusters benchmark:
  python analysis/run_group_mira.py \\
      --complex_names_npy results/posebusters_inference/metrics/complex_names.npy \\
      --results_dir       results/posebusters_inference \\
      --data_dir          data/posebusters_benchmark_set \\
      --out_dir           results/posebusters_inference/metrics/group_eval \\
      --num_runs 100 --n_workers 8
"""

import argparse
import os
import sys
import time

import numpy as np

_DIFFDOCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THESIS   = os.path.dirname(_DIFFDOCK)
for _p in (_DIFFDOCK, _THESIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_diffdock.loader import build_results_index
from eval_diffdock.group_mira_runner import run_group_mira_eval
from molcalib.mira import mira_null


def _build_flat_index(results_dir):
    """Build results index for both flat (PoseBusters) and chunked (PDBBind) layouts.

    Args:
        results_dir: path to the top-level results directory.

    Returns:
        dict mapping pdb_id (str) → Path of the complex subdirectory.
    """
    from pathlib import Path
    p = Path(results_dir)
    chunks = sorted(p.glob("chunk_*"))
    if chunks:
        return build_results_index(str(results_dir))
    index = {}
    for d in p.iterdir():
        if d.is_dir():
            index[d.name] = d
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Per-group MIRA calibration scores for DiffDock."
    )
    parser.add_argument(
        "--complex_names_npy", required=True,
        help="Path to .npy file with PDB ID strings.",
    )
    parser.add_argument(
        "--results_dir", required=True,
        help="Top-level DiffDock inference directory.",
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Root data directory (parent of per-complex subdirectories).",
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Output directory for .npy result files.",
    )
    parser.add_argument(
        "--num_runs", type=int, default=100,
        help="Monte Carlo center draws per complex per group (default: 100).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master random seed (default: 42).",
    )
    parser.add_argument(
        "--n_workers", type=int, default=4,
        help="Parallel worker processes (default: 4).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap the number of DiffDock samples used per complex.",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip groups whose output files already exist in --out_dir.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    complex_names = np.load(args.complex_names_npy, allow_pickle=True)
    print(f"Loaded {len(complex_names)} complex names from {args.complex_names_npy}")

    results_index = _build_flat_index(args.results_dir)
    print(f"Results index: {len(results_index)} entries from {args.results_dir}")

    missing = [n for n in complex_names if n not in results_index]
    if missing:
        print(f"Warning: {len(missing)} complexes not in results index — they will be skipped.")
    complex_names = [n for n in complex_names if n in results_index]
    print(f"Evaluating {len(complex_names)} complexes.\n")

    if args.skip_existing:
        already = all(
            os.path.exists(os.path.join(args.out_dir, f"mira_scores_{g}.npy"))
            for g in ("translation", "rotation", "torsion")
        )
        if already:
            print("All output files already exist — nothing to do (--skip_existing).")
            return

    print("=== Computing per-group MIRA scores ===")
    t0 = time.time()
    group_results = run_group_mira_eval(
        complex_names,
        results_index,
        args.data_dir,
        num_runs=args.num_runs,
        seed=args.seed,
        verbose=True,
        n_workers=args.n_workers,
        max_samples=args.max_samples,
    )
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s.")

    S_typ = args.max_samples or 40
    null = mira_null(S_typ)
    print(f"\nNull reference (S={S_typ}): {null:.4f}\n")

    for g in ("translation", "rotation", "torsion"):
        names, scores = group_results[g]
        np.save(os.path.join(args.out_dir, f"mira_scores_{g}.npy"), scores)
        np.save(os.path.join(args.out_dir, f"mira_names_{g}.npy"), names)
        if len(scores):
            print(f"  {g:12s}: n={len(scores):4d}, "
                  f"mean = {scores.mean():.4f}, "
                  f"std = {scores.std():.4f}, "
                  f"deviation from null = {scores.mean() - null:+.4f}")
            print(f"    → {args.out_dir}/mira_scores_{g}.npy")
        else:
            print(f"  {g:12s}: no valid complexes.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
