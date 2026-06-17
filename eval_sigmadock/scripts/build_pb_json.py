"""Build a per-complex PoseBusters breakdown JSON for SigmaDock results.

Produces posebusters_results_pb.json matching the DiffDock format:
  {
    "5SAK_ZRY": {
      "valid_ranks": ["rank1", "rank3"],
      "n_total": 40,
      "check_failures": {"minimum_distance_to_protein": 11, ...}
    }
  }

Logic:
  1. Load posebusters.pt from all seed_* directories (full 26-check PB suite).
  2. Optionally load rescoring.pt for heuristic ranking.
  3. Rank the 40 poses per complex using the same heuristic as eval_rmsd.py.
  4. valid_ranks: rank positions (1-indexed) that pass ALL available PB checks.
  5. check_failures: per-check failure counts across all 40 poses (only checks with ≥1 failure).

Args (CLI):
    results_dir:  model directory containing seed_* subdirectories.
    --scoring:    "heuristic" (default), "pb", or "vinardo".
    --output:     output JSON path (default: <results_dir>/posebusters_results_pb.json).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PB_PAPER_CHECKS = [
    "bond_lengths",
    "bond_angles",
    "tetrahedral_chirality",
    "internal_steric_clash",
    "minimum_distance_to_protein",
]


def _load_seeds(results_dir: Path):
    return sorted(
        [p for p in results_dir.glob("seed_*") if (p / "posebusters.pt").exists()],
        key=lambda p: int(p.name.split("_")[1]),
    )


def _pb_mean(passes_dict, checks):
    if passes_dict is None:
        return 0.0
    return float(np.mean([float(passes_dict.get(c, False)) for c in checks]))


def _impute_affinity(affinities):
    """Replace None affinities with worst-case sentinel."""
    valid = [a for a in affinities if a is not None]
    sentinel = (max(valid) * 1.1) if valid and max(valid) > 0 else 10.0
    return [a if a is not None else sentinel for a in affinities]


def _rank_heuristic(affinities, pb_means):
    affs = _impute_affinity(affinities)
    scores = [(-a) * (p ** 4) for a, p in zip(affs, pb_means)]
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def _rank_pb(pb_means):
    return sorted(range(len(pb_means)), key=lambda i: pb_means[i], reverse=True)


def _rank_vinardo(affinities):
    affs = _impute_affinity(affinities)
    return sorted(range(len(affs)), key=lambda i: affs[i])


def main(results_dir: str, scoring: str, output: str | None) -> None:
    model_dir = Path(results_dir)
    out_path  = Path(output) if output else model_dir / "posebusters_results_pb.json"

    seed_dirs = _load_seeds(model_dir)
    if not seed_dirs:
        print(f"No seed directories with posebusters.pt in {results_dir}")
        sys.exit(1)

    S = len(seed_dirs)
    print(f"Found {S} seeds")

    # Load all PB data
    first = torch.load(seed_dirs[0] / "posebusters.pt", weights_only=False)
    dataset_keys = sorted(first["rmsds"].keys())
    N = len(dataset_keys)
    print(f"Dataset: {N} complexes")

    # passes[cid][s] = dict{check→bool} or None
    passes_all = {k: [] for k in dataset_keys}
    for seed_dir in seed_dirs:
        data = torch.load(seed_dir / "posebusters.pt", weights_only=False)
        for k in dataset_keys:
            passes_all[k].append(data["pb_dicts"].get(k))

    # Load Vinardo scores if needed
    affinities_all = None
    if scoring in ("heuristic", "vinardo"):
        affinities_all = {k: [] for k in dataset_keys}
        for seed_dir in seed_dirs:
            rs_path = seed_dir / "rescoring.pt"
            scores = {}
            if rs_path.exists():
                rs = torch.load(rs_path, weights_only=False)
                scores = rs.get("scores", {})
            for k in dataset_keys:
                entry = scores.get(k)
                if entry and isinstance(entry, list) and entry[0]:
                    affinities_all[k].append(entry[0].get("Affinity"))
                else:
                    affinities_all[k].append(None)

    result = {}
    for k in dataset_keys:
        p_list  = passes_all[k]   # list[dict|None], length S
        pb_means = [_pb_mean(p, PB_PAPER_CHECKS) for p in p_list]

        if scoring == "heuristic":
            order = _rank_heuristic(affinities_all[k], pb_means)
        elif scoring == "pb":
            order = _rank_pb(pb_means)
        else:
            order = _rank_vinardo(affinities_all[k])

        # Discover all check keys present across seeds
        all_checks = set()
        for pd in p_list:
            if pd:
                all_checks.update(pd.keys())

        # Per-check failure counts across ALL 40 poses (unranked)
        check_failures = {}
        for check in sorted(all_checks):
            n_fail = sum(1 for pd in p_list if pd is not None and not pd.get(check, True))
            if n_fail > 0:
                check_failures[check] = n_fail

        # Which rank positions pass ALL PB checks (using all discovered checks)
        valid_ranks = []
        for rank_idx, seed_idx in enumerate(order, start=1):
            pd = p_list[seed_idx]
            if pd is not None and all(pd.get(c, False) for c in all_checks):
                valid_ranks.append(f"rank{rank_idx}")

        result[k] = {
            "valid_ranks":    valid_ranks,
            "n_total":        S,
            "check_failures": check_failures,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    n_any_valid = sum(1 for v in result.values() if v["valid_ranks"])
    n_rank1_valid = sum(1 for v in result.values() if v["valid_ranks"] and v["valid_ranks"][0] == "rank1")
    print(f"\nComplexes with ≥1 PB-valid pose: {n_any_valid}/{N} ({n_any_valid/N*100:.1f}%)")
    print(f"Complexes where rank-1 is PB-valid: {n_rank1_valid}/{N} ({n_rank1_valid/N*100:.1f}%)")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build per-complex PoseBusters breakdown JSON for SigmaDock results.")
    parser.add_argument("results_dir",
                        help="Model directory containing seed_* subdirectories.")
    parser.add_argument("--scoring", choices=["heuristic", "pb", "vinardo"],
                        default="heuristic",
                        help="Pose ranking method (default: heuristic = Affinity × PB⁴).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Default: <results_dir>/posebusters_results_pb.json")
    args = parser.parse_args()
    main(args.results_dir, args.scoring, args.output)
