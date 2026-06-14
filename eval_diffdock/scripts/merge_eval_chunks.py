#!/usr/bin/env python3
"""
Merge DiffDock parallel evaluation chunks into a single output directory.

Concatenates all per-complex .npy arrays across chunks and copies
complex prediction subdirectories. Saves individual .npy files
(matching evaluate.py's format) and a single merged.npz for convenience.

Usage:
    python merge_eval_chunks.py <chunks_parent_dir> <out_dir>

Example:
    python merge_eval_chunks.py results/testset_eval_full results/testset_eval_merged
"""

import argparse
import shutil
import sys
import numpy as np
from pathlib import Path

# Arrays that are (N_complexes, samples_per_complex) or (N_complexes,)
ARRAY_FILES = [
    "rmsds.npy",
    "centroid_distances.npy",
    "confidences.npy",
    "min_self_distances.npy",
    "run_times.npy",
    "complex_names.npy",
    "gnina_rmsds.npy",
    "gnina_score.npy",
]


def load_array(path):
    """Return array from .npy, or None if the file contains a None/empty object."""
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0:  # scalar object — evaluate.py saves None as shape-() array
        return None
    return arr


def merge_chunks(chunks_dir: Path, out_dir: Path):
    chunk_dirs = sorted(chunks_dir.glob("chunk_*"), key=lambda p: int(p.name.split("_")[1]))

    if not chunk_dirs:
        sys.exit(f"No chunk_* directories found in {chunks_dir}")

    print(f"Chunks found: {[d.name for d in chunk_dirs]}")

    collected = {f: [] for f in ARRAY_FILES}
    total = 0

    for chunk_dir in chunk_dirs:
        names_path = chunk_dir / "complex_names.npy"
        if not names_path.exists():
            print(f"  {chunk_dir.name}: no complex_names.npy — skipping (job may have failed)")
            continue

        n = len(np.load(names_path, allow_pickle=True))
        print(f"  {chunk_dir.name}: {n} complexes")
        total += n

        for fname in ARRAY_FILES:
            fpath = chunk_dir / fname
            if fpath.exists():
                arr = load_array(fpath)
                if arr is not None:
                    collected[fname].append(arr)

        # Copy complex prediction subdirectories
        for item in sorted(chunk_dir.iterdir()):
            if not item.is_dir():
                continue
            dest = out_dir / item.name
            if dest.exists():
                print(f"    Warning: {item.name} already in output dir, skipping copy")
            else:
                shutil.copytree(item, dest)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate and save each array
    npz_payload = {}
    for fname, arrays in collected.items():
        if not arrays:
            continue
        merged = np.concatenate(arrays, axis=0)
        np.save(out_dir / fname, merged)
        npz_payload[fname.replace(".npy", "")] = merged
        print(f"Saved {fname}: shape={merged.shape}")

    # Also save everything in one .npz for convenience
    np.savez(out_dir / "merged.npz", **npz_payload)
    print(f"\nSaved merged.npz with keys: {list(npz_payload)}")
    print(f"Total: {total} complexes → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chunks_dir", type=Path, help="Parent directory containing chunk_0/, chunk_1/, ...")
    parser.add_argument("out_dir", type=Path, help="Output directory for merged results")
    args = parser.parse_args()
    merge_chunks(args.chunks_dir, args.out_dir)
