"""Merge seed_* result directories from two SigmaDock runs into one combined directory.

All three output file types (predictions.pt, posebusters.pt, rescoring.pt) use
string PDB-ID keys, so merging is a plain dict union. The two source runs must
cover disjoint sets of complexes.

Args (CLI):
    --dir1:   Model directory for run 1 (parent of seed_*/ subdirectories).
    --dir2:   Model directory for run 2 (parent of seed_*/ subdirectories).
    --output: Output model directory (will be created; seed_*/ written directly inside).

Returns:
    Writes merged predictions.pt, posebusters.pt, and rescoring.pt (when present
    in both sources) for every seed found in both directories.
"""

import argparse
import sys
from pathlib import Path

import torch


def load_seed(seed_dir: Path) -> dict:
    """Load all .pt result files from one seed directory."""
    data = {}
    for name in ("predictions", "posebusters", "rescoring"):
        path = seed_dir / f"{name}.pt"
        if path.exists():
            data[name] = torch.load(path, weights_only=False)
    return data


def merge_predictions(p1: dict, p2: dict) -> dict:
    r1, r2 = p1["results"], p2["results"]
    overlap = set(r1) & set(r2)
    if overlap:
        raise ValueError(f"Overlapping complex IDs in predictions: {sorted(overlap)[:5]}")
    return {"results": {**r1, **r2}, "meta": p1.get("meta", {})}


def merge_posebusters(pb1: dict, pb2: dict) -> dict:
    merged = {}
    for key in ("rmsds", "pb_checks", "pb_dicts"):
        d1, d2 = pb1.get(key, {}), pb2.get(key, {})
        overlap = set(d1) & set(d2)
        if overlap:
            raise ValueError(f"Overlapping IDs in posebusters[{key}]: {sorted(overlap)[:5]}")
        merged[key] = {**d1, **d2}
    return merged


def merge_rescoring(r1: dict, r2: dict) -> dict:
    s1, s2 = r1.get("scores", {}), r2.get("scores", {})
    overlap = set(s1) & set(s2)
    if overlap:
        raise ValueError(f"Overlapping IDs in rescoring scores: {sorted(overlap)[:5]}")
    merged = {
        "scores": {**s1, **s2},
        "failed": r1.get("failed", []) + r2.get("failed", []),
    }
    if "score_config" in r1:
        merged["score_config"] = r1["score_config"]
    return merged


def main(dir1: str, dir2: str, output: str) -> None:
    model_dir1, model_dir2, out_dir = Path(dir1), Path(dir2), Path(output)

    def get_seeds(d: Path) -> dict[int, Path]:
        return {
            int(p.name.split("_")[1]): p
            for p in d.glob("seed_*") if p.is_dir()
        }

    seeds1, seeds2 = get_seeds(model_dir1), get_seeds(model_dir2)
    common = sorted(set(seeds1) & set(seeds2))

    print(f"dir1: {model_dir1} ({len(seeds1)} seeds)")
    print(f"dir2: {model_dir2} ({len(seeds2)} seeds)")
    print(f"Merging {len(common)} seeds → {out_dir}")

    if not common:
        print("ERROR: no matching seed numbers found in both directories")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    for seed_num in common:
        d1, d2 = load_seed(seeds1[seed_num]), load_seed(seeds2[seed_num])
        out_seed = out_dir / f"seed_{seed_num}"
        out_seed.mkdir(exist_ok=True)

        if "predictions" in d1 and "predictions" in d2:
            merged = merge_predictions(d1["predictions"], d2["predictions"])
            torch.save(merged, out_seed / "predictions.pt")
            n = len(merged["results"])
        else:
            n = "?"

        if "posebusters" in d1 and "posebusters" in d2:
            torch.save(merge_posebusters(d1["posebusters"], d2["posebusters"]), out_seed / "posebusters.pt")

        if "rescoring" in d1 and "rescoring" in d2:
            torch.save(merge_rescoring(d1["rescoring"], d2["rescoring"]), out_seed / "rescoring.pt")

        print(f"  seed_{seed_num}: {n} complexes")

    print(f"\nDone. {len(common)} seeds written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", required=True, help="Model dir for run 1 (contains seed_*/)")
    parser.add_argument("--dir2", required=True, help="Model dir for run 2 (contains seed_*/)")
    parser.add_argument("--output", required=True, help="Output model directory")
    args = parser.parse_args()
    main(args.dir1, args.dir2, args.output)
