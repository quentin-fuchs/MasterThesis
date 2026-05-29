"""
Compute per-group (translation / rotation / torsion) TARP fractions and raw
group distances for the DiffDock PDBBind or PoseBusters test sets.

Saves to --out_dir:
  tarp_fractions_translation.npy  (n_complexes, K)
  tarp_fractions_rotation.npy     (n_complexes, K)
  tarp_fractions_torsion.npy      (n_complexes, K)
  distances_translation.npy       (n_complexes, S)   Å
  distances_rotation.npy          (n_complexes, S)   rad
  distances_torsion_rms.npy       (n_complexes, S)   rad
  complex_names.npy               (n_complexes,)
  n_rot_bonds.npy                 (n_complexes,)     int

Usage
-----
PDBBind test set:
  python analysis/run_group_eval.py \
      --complex_names_npy /home/qf226/rds/results/pdbbind_testset/metrics/complex_names.npy \
      --results_dir       /home/qf226/rds/results/pdbbind_testset/poses \
      --data_dir          data/PDBBind_processed \
      --out_dir           /home/qf226/rds/results/pdbbind_testset/metrics/group_eval \
      --K 100 --n_workers 8

PoseBusters benchmark:
  python analysis/run_group_eval.py \
      --complex_names_npy results/posebusters_inference/metrics/complex_names.npy \
      --results_dir       results/posebusters_inference \
      --data_dir          data/posebusters_benchmark_set \
      --out_dir           results/posebusters_inference/metrics/group_eval \
      --K 100 --n_workers 8
"""

import argparse
import os
import sys
import time

import numpy as np

# Make project root importable when invoked from analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tarp_eval import build_results_index
from utils.group_eval import run_group_tarp_eval, run_group_distances


def _build_flat_index(results_dir):
    """Build results index from either a flat or chunked directory layout.

    PoseBusters inference results are stored in a flat directory (one
    subdir per complex at the top level), whereas PDBBind results use the
    chunk_0 / chunk_1 / ... layout produced by evaluate.py.

    Args:
        results_dir: path to the top-level results directory.

    Returns:
        dict mapping pdb_id (str) → Path of the complex subdirectory.
    """
    from pathlib import Path
    p = Path(results_dir)
    # Try the chunked layout first
    chunks = sorted(p.glob("chunk_*"))
    if chunks:
        return build_results_index(str(results_dir))
    # Fall back to flat layout
    index = {}
    for d in p.iterdir():
        if d.is_dir():
            index[d.name] = d
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Per-group TARP and distance evaluation for DiffDock."
    )
    parser.add_argument(
        "--complex_names_npy", required=True,
        help="Path to .npy file with PDB ID strings (e.g. complex_names.npy).",
    )
    parser.add_argument(
        "--results_dir", required=True,
        help="Top-level directory of DiffDock inference output (contains "
             "one subdir per complex with rank*.sdf files).",
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Root data directory (contains one subdir per PDB ID with the "
             "crystal ligand SDF and protein PDB).",
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Output directory for .npy result files.",
    )
    parser.add_argument(
        "--K", type=int, default=100,
        help="Number of random reference draws per complex per group (default: 100).",
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
        "--skip_distances", action="store_true",
        help="Skip raw group-distance computation (only run TARP).",
    )
    parser.add_argument(
        "--skip_tarp", action="store_true",
        help="Skip TARP computation (only compute raw distances).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load complex names
    complex_names = np.load(args.complex_names_npy, allow_pickle=True)
    print(f"Loaded {len(complex_names)} complex names from {args.complex_names_npy}")

    # Build results index
    results_index = _build_flat_index(args.results_dir)
    print(f"Results index: {len(results_index)} entries from {args.results_dir}")

    # Filter to complexes present in the results index
    missing = [n for n in complex_names if n not in results_index]
    if missing:
        print(f"Warning: {len(missing)} complexes not found in results index "
              f"(e.g. {missing[:3]}); they will be skipped.")
    complex_names = [n for n in complex_names if n in results_index]
    print(f"Evaluating {len(complex_names)} complexes.")

    # --- Raw group distances ---
    if not args.skip_distances:
        print("\n=== Computing raw group distances ===")
        t0 = time.time()
        dist_results = run_group_distances(
            complex_names,
            results_index,
            args.data_dir,
            verbose=True,
            n_workers=args.n_workers,
            max_samples=args.max_samples,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s.")

        # Save
        for key in ("translation", "rotation", "torsion_rms"):
            path = os.path.join(args.out_dir, f"distances_{key}.npy")
            np.save(path, dist_results[key])
            print(f"  Saved {dist_results[key].shape} → {path}")

        # Save names and n_rot_bonds from distance run (more complexes may
        # succeed here than in TARP since TARP also needs protein coords).
        np.save(os.path.join(args.out_dir, "complex_names_distances.npy"),
                dist_results["names"])
        np.save(os.path.join(args.out_dir, "n_rot_bonds.npy"),
                dist_results["n_rot_bonds"])
        print(f"  Saved complex names ({len(dist_results['names'])}) and "
              f"n_rot_bonds to {args.out_dir}")

    # --- Per-group TARP fractions ---
    if not args.skip_tarp:
        print("\n=== Computing per-group TARP fractions ===")
        t0 = time.time()
        tarp_results = run_group_tarp_eval(
            complex_names,
            results_index,
            args.data_dir,
            K=args.K,
            seed=args.seed,
            verbose=True,
            n_workers=args.n_workers,
            max_samples=args.max_samples,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s.")

        for grp in ("translation", "rotation", "torsion"):
            path = os.path.join(args.out_dir, f"tarp_fractions_{grp}.npy")
            np.save(path, tarp_results[grp])
            print(f"  Saved {tarp_results[grp].shape} → {path}")

        np.save(os.path.join(args.out_dir, "complex_names.npy"),
                tarp_results["names"])
        np.save(os.path.join(args.out_dir, "n_rot_bonds_tarp.npy"),
                tarp_results["n_rot_bonds"])
        print(f"  Saved complex names ({len(tarp_results['names'])}) to {args.out_dir}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
