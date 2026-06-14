"""
Merge the 10 independent top1 array runs into two combined result directories:

  <out_prefix>_top1_merged/  — 1 pose per run  = 10 poses per complex
  <out_prefix>_top3_merged/  — 3 poses per run = 30 poses per complex

Each merged directory contains a single chunk_0/ subdirectory so it is
compatible with build_results_index() in tarp_eval.py without any changes.

Poses are symlinked (not copied) to avoid duplicating disk space. Only the
rank*.sdf files are linked; rank*_confidence*.sdf files are excluded so
load_sample_coords() does not load duplicate coordinates.

Usage:
    python eval_diffdock/preprocess/merge_top1_runs.py --runs_dir <path/to/runs> --out_prefix <path/prefix>

Defaults (backwards-compatible):
    --runs_dir    <diffdock_dir>/results/top1_runs
    --out_prefix  <diffdock_dir>/results/top1_runs   (→ top1_runs_merged, top3_runs_merged)
"""

import argparse
from pathlib import Path

DIFFDOCK_DIR = Path(__file__).resolve().parents[2] / "diffdock"
N_RUNS = 10


def merge_runs(runs_dir: Path, n_poses_per_run: int, out_dir: Path) -> None:
    """Symlink the top-n ranked poses from each run into a single merged dir.

    Args:
        runs_dir: Directory containing run_0/, run_1/, … run_9/ subdirectories.
        n_poses_per_run: Number of top poses to take from each run (1 or 3).
        out_dir: Destination directory (will be created if absent).
    """
    chunk_dir = out_dir / "chunk_0"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    run0 = runs_dir / "run_0"
    complexes = sorted(p.name for p in run0.iterdir() if p.is_dir())
    print(f"Found {len(complexes)} complexes in run_0.")

    skipped = 0
    for complex_name in complexes:
        dest_complex = chunk_dir / complex_name
        dest_complex.mkdir(exist_ok=True)

        rank_out = 1
        for run_idx in range(N_RUNS):
            src_complex = runs_dir / f"run_{run_idx}" / complex_name
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs_dir", type=Path,
        default=DIFFDOCK_DIR / "results" / "top1_runs",
        help="Directory containing run_0/ … run_9/ (default: results/top1_runs)",
    )
    parser.add_argument(
        "--out_prefix", type=Path,
        default=None,
        help="Prefix for output dirs: <prefix>_top1_merged and <prefix>_top3_merged. "
             "Defaults to the same path as --runs_dir.",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    out_prefix: Path = args.out_prefix if args.out_prefix is not None else runs_dir

    print("Merging runs (top-1 per run → 10 poses per complex)...")
    merge_runs(
        runs_dir=runs_dir,
        n_poses_per_run=1,
        out_dir=Path(str(out_prefix) + "_top1_merged"),
    )

    print("\nMerging runs (top-3 per run → 30 poses per complex)...")
    merge_runs(
        runs_dir=runs_dir,
        n_poses_per_run=3,
        out_dir=Path(str(out_prefix) + "_top3_merged"),
    )

    print("\nDone. Pass either directory to build_results_index() in tarp_eval.py.")
