"""
Merge the 10 independent top1 array runs into two combined result directories:

  results/top1_runs_merged/   — 1 pose per run  = 10 poses per complex
  results/top3_runs_merged/   — 3 poses per run = 30 poses per complex

Each merged directory contains a single chunk_0/ subdirectory so it is
compatible with build_results_index() in tarp_eval.py without any changes.

Poses are symlinked (not copied) to avoid duplicating disk space. Only the
rank*.sdf files are linked; rank*_confidence*.sdf files are excluded so
load_sample_coords() does not load duplicate coordinates.

Usage:
    python utils/merge_top1_runs.py
"""

import os
from pathlib import Path

DIFFDOCK_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = DIFFDOCK_DIR / "results" / "top1_runs"
N_RUNS = 10


def merge_runs(n_poses_per_run: int, out_dir: Path) -> None:
    """Symlink the top-n ranked poses from each run into a single merged dir.

    Args:
        n_poses_per_run: Number of top poses to take from each run (1 or 3).
        out_dir: Destination directory (will be created if absent).
    """
    chunk_dir = out_dir / "chunk_0"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Collect complex names from run_0 as the reference set.
    run0 = RUNS_DIR / "run_0"
    complexes = sorted(p.name for p in run0.iterdir() if p.is_dir())
    print(f"Found {len(complexes)} complexes in run_0.")

    skipped = 0
    for complex_name in complexes:
        dest_complex = chunk_dir / complex_name
        dest_complex.mkdir(exist_ok=True)

        rank_out = 1
        for run_idx in range(N_RUNS):
            src_complex = RUNS_DIR / f"run_{run_idx}" / complex_name
            if not src_complex.is_dir():
                skipped += 1
                continue
            for pose_rank in range(1, n_poses_per_run + 1):
                src = src_complex / f"rank{pose_rank}.sdf"
                if not src.exists():
                    continue
                dest = dest_complex / f"rank{rank_out}.sdf"
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                dest.symlink_to(src.resolve())
                rank_out += 1

    total_poses = sum(
        len(list((chunk_dir / c).glob("rank*.sdf"))) for c in complexes
        if (chunk_dir / c).exists()
    )
    print(f"  → {out_dir.name}: {len(complexes)} complexes, "
          f"{total_poses} total poses "
          f"({total_poses // len(complexes) if complexes else 0} per complex avg)")
    if skipped:
        print(f"  Warning: {skipped} complex/run combinations were missing.")


if __name__ == "__main__":
    print("Merging runs (top-1 per run → 10 poses per complex)...")
    merge_runs(
        n_poses_per_run=1,
        out_dir=DIFFDOCK_DIR / "results" / "top1_runs_merged",
    )

    print("\nMerging runs (top-3 per run → 30 poses per complex)...")
    merge_runs(
        n_poses_per_run=3,
        out_dir=DIFFDOCK_DIR / "results" / "top3_runs_merged",
    )

    print("\nDone. Pass either directory to build_results_index() in tarp_eval.py.")
