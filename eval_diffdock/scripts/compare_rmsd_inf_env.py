#!/usr/bin/env python
"""
Fair top-1 SpyRMSD comparison between two DiffDock runs on the same complexes.

Computes top-1 symmetry-corrected RMSD from scratch using rank1.sdf for both:
  - pdbbind_testset/poses/ (322 complexes — restricted to the intersection)
  - pdbbind_eval_n100_inf_env/ (88 complexes)

Only the overlapping complexes are compared.

Usage
-----
python eval_diffdock/scripts/compare_rmsd_inf_env.py
"""

import numpy as np

from eval_diffdock.loader import build_results_index
from eval_diffdock.rmsd_runner import run_rmsd_eval

RDS = "/home/qf226/rds/hpc-work"
TESTSET_DIR = f"{RDS}/results/DiffDock/pdbbind_testset/poses"
INF_ENV_DIR = f"{RDS}/results/DiffDock/test_runs/pdbbind_eval_n100_inf_env"
DATA_DIR    = f"{RDS}/data/PDBBind_processed"


def main():
    idx_testset = build_results_index(TESTSET_DIR)
    idx_inf_env = build_results_index(INF_ENV_DIR)

    common = sorted(set(idx_testset) & set(idx_inf_env))
    print(f"Testset complexes:  {len(idx_testset)}")
    print(f"inf_env complexes:  {len(idx_inf_env)}")
    print(f"Common complexes:   {len(common)}")
    print()

    print("=== Testset run (restricted to common complexes) ===")
    names_ts, top1_ts, _ = run_rmsd_eval(
        common, idx_testset, DATA_DIR, max_samples=40, verbose=True,
    )

    print()
    print("=== inf_env run (all complexes) ===")
    names_ie, top1_ie, _ = run_rmsd_eval(
        common, idx_inf_env, DATA_DIR, max_samples=40, verbose=True,
    )

    # Align by name (both should return same set but in case of skipped complexes)
    ts_map = dict(zip(names_ts, top1_ts))
    ie_map = dict(zip(names_ie, top1_ie))
    paired = sorted(set(ts_map) & set(ie_map))
    print(f"\n=== Paired comparison ({len(paired)} complexes) ===")
    ts_arr = np.array([ts_map[n] for n in paired])
    ie_arr = np.array([ie_map[n] for n in paired])

    for thresh in [2.0, 5.0]:
        ts_pct = (ts_arr < thresh).mean() * 100
        ie_pct = (ie_arr < thresh).mean() * 100
        diff   = ie_pct - ts_pct
        print(f"  Top-1 < {thresh:.0f}Å:  testset={ts_pct:.1f}%  inf_env={ie_pct:.1f}%  diff={diff:+.1f}pp")

    print(f"  Median top-1 RMSD:  testset={np.nanmedian(ts_arr):.2f}Å  inf_env={np.nanmedian(ie_arr):.2f}Å")

    # Per-complex delta so we can see which complexes drive the gap
    deltas = ie_arr - ts_arr
    worst_idx = np.argsort(deltas)[::-1][:10]
    print("\n  Top-10 complexes with largest inf_env−testset RMSD increase:")
    for i in worst_idx:
        print(f"    {paired[i]:8s}  testset={ts_arr[i]:.2f}Å  inf_env={ie_arr[i]:.2f}Å  Δ={deltas[i]:+.2f}Å")


if __name__ == "__main__":
    main()
