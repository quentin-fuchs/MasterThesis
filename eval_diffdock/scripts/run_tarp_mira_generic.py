"""
Compute TARP RMSD fractions and MIRA scores for any DiffDock results directory.

Handles both flat (poses/) and chunked (chunk_*/) directory layouts via
build_results_index. Derives complex names from the results index unless a
--complex_names_npy is provided.

Saves to --metrics_dir:
  tarp_fractions_symrmsd_K{K}.npy  — TARP fractions (n, K), sym-corrected RMSD
  tarp_fractions_symrmsd_K1.npy    — TARP fractions (n, 1)
  mira_names_symrmsd.npy           — complex names for MIRA
  mira_scores_symrmsd.npy          — per-complex MIRA scores

Usage:
    python eval_diffdock/scripts/run_tarp_mira_generic.py \\
        --results_dir /path/to/results \\
        --data_dir    /path/to/PDBBind_processed \\
        --metrics_dir /path/to/metrics \\
        --K 10 \\
        --num_runs 100 \\
        --n_workers 14
"""

import argparse
import os

import numpy as np

from eval_diffdock.loader import build_results_index
from eval_diffdock.tarp_runner import run_tarp_eval
from eval_diffdock.mira_runner import compute_mira_scores


def main():
    parser = argparse.ArgumentParser(
        description="TARP symRMSD + MIRA evaluation for any DiffDock results directory."
    )
    parser.add_argument("--results_dir", required=True,
                        help="Root of DiffDock output (flat or chunk_*/ layout).")
    parser.add_argument("--data_dir", required=True,
                        help="Root data directory (PDBBind_processed or similar).")
    parser.add_argument("--metrics_dir", required=True,
                        help="Output directory for .npy metric files.")
    parser.add_argument("--complex_names_npy", default=None,
                        help="Optional .npy of PDB IDs; derived from results_index if omitted.")
    parser.add_argument("--K", type=int, default=10,
                        help="Primary TARP reference draws (default: 10). K=1 always run too.")
    parser.add_argument("--num_runs", type=int, default=100,
                        help="MIRA Monte Carlo center draws per complex (default: 100).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.metrics_dir, exist_ok=True)

    results_index = build_results_index(args.results_dir)
    print(f"Results index: {len(results_index)} complexes in {args.results_dir}")

    if args.complex_names_npy:
        complex_names = np.load(args.complex_names_npy, allow_pickle=True)
        complex_names = np.array([n for n in complex_names if n in results_index])
    else:
        complex_names = np.array(sorted(results_index.keys()))

    print(f"Complexes: {len(complex_names)}")

    # Save complex names for downstream use
    names_out = os.path.join(args.metrics_dir, "complex_names.npy")
    if not os.path.exists(names_out):
        np.save(names_out, complex_names)
        print(f"Saved {names_out}")

    N_WORKERS = args.n_workers

    # --- TARP symRMSD K (primary) ---
    out_K = os.path.join(args.metrics_dir, f"tarp_fractions_symrmsd_K{args.K}.npy")
    if os.path.exists(out_K):
        print(f"K={args.K} already exists, skipping.")
    else:
        print(f"\nRunning TARP symRMSD K={args.K} (n_workers={N_WORKERS}) ...")
        f = run_tarp_eval(complex_names, results_index, args.data_dir,
                          K=args.K, mode="rmsd", seed=args.seed,
                          verbose=True, n_workers=N_WORKERS)
        np.save(out_K, f)
        print(f"Saved {out_K}  shape={f.shape}")

    # --- TARP symRMSD K=1 ---
    out_K1 = os.path.join(args.metrics_dir, "tarp_fractions_symrmsd_K1.npy")
    if args.K == 1:
        out_K1 = out_K  # already done above
    elif os.path.exists(out_K1):
        print("K=1 already exists, skipping.")
    else:
        print(f"\nRunning TARP symRMSD K=1 (n_workers={N_WORKERS}) ...")
        f = run_tarp_eval(complex_names, results_index, args.data_dir,
                          K=1, mode="rmsd", seed=args.seed,
                          verbose=True, n_workers=N_WORKERS)
        np.save(out_K1, f)
        print(f"Saved {out_K1}  shape={f.shape}")

    # --- MIRA symRMSD ---
    out_mira_scores = os.path.join(args.metrics_dir, "mira_scores_symrmsd.npy")
    out_mira_names  = os.path.join(args.metrics_dir, "mira_names_symrmsd.npy")
    if os.path.exists(out_mira_scores):
        print("MIRA scores already exist, skipping.")
    else:
        print(f"\nRunning MIRA (metric=symrmsd, num_runs={args.num_runs},"
              f" n_workers={N_WORKERS}) ...")
        mira_names, mira_scores = compute_mira_scores(
            complex_names, results_index, args.data_dir,
            num_runs=args.num_runs, verbose=True,
            metric="symrmsd", seed=args.seed, n_workers=N_WORKERS,
        )
        np.save(out_mira_names,  mira_names)
        np.save(out_mira_scores, mira_scores)
        print(f"Saved MIRA scores ({len(mira_scores)} complexes)  "
              f"mean={mira_scores.mean():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
