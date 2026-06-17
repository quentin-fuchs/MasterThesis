"""Evaluate top-1 PoseBusters performance for SigmaDock results.

Loads posebusters.pt (symRMSD + PB checks) and rescoring.pt (Vinardo affinity)
from seed_* directories, ranks poses by the selected scoring method, and reports
top-1 success rates at RMSD < 2 Å with and without PoseBusters validity filter.

Ranking modes:
  heuristic (default) — Appendix F.2: score = −Affinity × pb_mean⁴
                         where pb_mean = mean of 5 physicochemical PB checks.
                         Requires posebusters.pt and rescoring.pt.
  pb                  — Mean pass rate of 5 PB checks only. No rescoring.pt needed.
  vinardo             — Vinardo affinity (ascending). Requires rescoring.pt.

Args (CLI):
    results_dir:   model directory containing seed_* subdirectories.
    --scoring:     "heuristic" (default), "pb", or "vinardo".
    --output-dir:  if set, save rmsds.npy (N, S) and top1_rmsd.npy (N,) here.
"""

import argparse
import random
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
    """Return sorted list of seed_* dirs that have posebusters.pt."""
    return sorted(
        [p for p in results_dir.glob("seed_*") if (p / "posebusters.pt").exists()],
        key=lambda p: int(p.name.split("_")[1]),
    )


def _load_pb(seed_dirs):
    """Aggregate posebusters.pt across seeds.

    Returns:
        dataset_keys: sorted list of complex IDs.
        rmsds:  dict[cid] → list[float|None]  length S
        passes: dict[cid] → list[dict[check→bool]|None]  length S
    """
    first = torch.load(seed_dirs[0] / "posebusters.pt", weights_only=False)
    dataset_keys = sorted(first["rmsds"].keys())

    rmsds  = {k: [] for k in dataset_keys}
    passes = {k: [] for k in dataset_keys}

    for seed_dir in seed_dirs:
        data = torch.load(seed_dir / "posebusters.pt", weights_only=False)
        for k in dataset_keys:
            rmsds[k].append(data["rmsds"].get(k))
            passes[k].append(data["pb_dicts"].get(k))  # dict or None

    return dataset_keys, rmsds, passes


def _load_scores(seed_dirs, dataset_keys):
    """Aggregate Vinardo Affinity from rescoring.pt across seeds.

    Returns:
        dict[cid] → list[float|None]  length S
    """
    affinities = {k: [] for k in dataset_keys}
    for seed_dir in seed_dirs:
        rs_path = seed_dir / "rescoring.pt"
        if rs_path.exists():
            rs = torch.load(rs_path, weights_only=False)
            scores = rs.get("scores", {})
        else:
            scores = {}
        for k in dataset_keys:
            entry = scores.get(k)
            if entry and isinstance(entry, list) and entry[0]:
                affinities[k].append(entry[0].get("Affinity"))
            else:
                affinities[k].append(None)
    return affinities


def _pb_mean(passes_dict, checks):
    """Mean pass rate for the given PB checks (treats missing check as False)."""
    if passes_dict is None:
        return 0.0
    return float(np.mean([float(passes_dict.get(c, False)) for c in checks]))


def _impute_none(values, rng, sentinel_fn):
    """Replace None entries by sampling from valid ones; if none, use sentinel."""
    valid = [v for v in values if v is not None]
    if not valid:
        sent = sentinel_fn()
        return [sent if v is None else v for v in values]
    return [rng.choice(valid) if v is None else v for v in values]


def _rank_heuristic(rmsds_s, affinities_s, pb_means_s, rng):
    """Return sorted indices (best first) by −Affinity × pb_mean⁴."""
    max_aff = max((a for a in affinities_s if a is not None), default=0.0)
    sentinel_aff = max_aff * 1.1 if max_aff > 0 else 10.0
    affs = _impute_none(affinities_s, rng, lambda: sentinel_aff)
    scores = [(-a) * (p ** 4) for a, p in zip(affs, pb_means_s)]
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def _rank_pb(pb_means_s):
    """Return sorted indices by PB mean (descending)."""
    return sorted(range(len(pb_means_s)), key=lambda i: pb_means_s[i], reverse=True)


def _rank_vinardo(affinities_s, rng):
    """Return sorted indices by Affinity ascending (lower = better)."""
    max_aff = max((a for a in affinities_s if a is not None), default=0.0)
    sentinel_aff = max_aff * 1.1 if max_aff > 0 else 10.0
    affs = _impute_none(affinities_s, rng, lambda: sentinel_aff)
    return sorted(range(len(affs)), key=lambda i: affs[i])


def main(results_dir: str, scoring: str, output_dir: str | None) -> None:
    model_dir = Path(results_dir)
    seed_dirs = _load_seeds(model_dir)

    if not seed_dirs:
        print(f"No seed directories with posebusters.pt in {results_dir}")
        sys.exit(1)

    S = len(seed_dirs)
    print(f"Found {S} seeds")

    needs_rescoring = scoring in ("heuristic", "vinardo")
    if needs_rescoring:
        has_scores = any((p / "rescoring.pt").exists() for p in seed_dirs)
        if not has_scores:
            print(f"ERROR: --scoring {scoring} requires rescoring.pt")
            sys.exit(1)

    dataset_keys, rmsds, passes = _load_pb(seed_dirs)
    N = len(dataset_keys)
    print(f"Dataset: {N} complexes")

    affinities = _load_scores(seed_dirs, dataset_keys) if needs_rescoring else None

    rng = np.random.default_rng(0)
    py_rng = random.Random(0)

    sorted_rmsds = np.full((N, S), np.nan)
    top1_pass_rmsd = 0
    top1_pass_pb   = 0

    for i, k in enumerate(dataset_keys):
        r_list = rmsds[k]          # list[float|None], length S
        p_list = passes[k]         # list[dict|None], length S
        pb_means = [_pb_mean(p, PB_PAPER_CHECKS) for p in p_list]

        # Impute None rmsds
        valid_r = [r for r in r_list if r is not None]
        sent_r  = (max(valid_r) + 5.0) if valid_r else 20.0
        r_clean = [r if r is not None else sent_r for r in r_list]

        if scoring == "heuristic":
            order = _rank_heuristic(r_list, affinities[k], pb_means, rng)
        elif scoring == "pb":
            order = _rank_pb(pb_means)
        else:  # vinardo
            order = _rank_vinardo(affinities[k], rng)

        sorted_rmsds[i] = [r_clean[j] for j in order]

        best_r = r_clean[order[0]]
        best_passes = p_list[order[0]]

        if best_r < 2.0:
            top1_pass_rmsd += 1
            if best_passes and all(best_passes.get(c, False) for c in PB_PAPER_CHECKS):
                top1_pass_pb += 1

    print(f"\n=== Top-1 results | {S} seeds | {N} complexes ===")
    print(f"  RMSD < 2 Å:           {top1_pass_rmsd / N * 100:.1f}%")
    print(f"  RMSD < 2 Å + PB pass: {top1_pass_pb   / N * 100:.1f}%")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "rmsds.npy",      sorted_rmsds)
        np.save(out / "top1_rmsd.npy",  sorted_rmsds[:, 0])
        print(f"\nSaved rmsds.npy {sorted_rmsds.shape} and top1_rmsd.npy → {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Top-1 RMSD accuracy for SigmaDock PoseBusters results.")
    parser.add_argument("results_dir",
                        help="Model directory containing seed_* subdirectories.")
    parser.add_argument("--scoring", choices=["heuristic", "pb", "vinardo"],
                        default="heuristic",
                        help="Ranking method (default: heuristic = Affinity × PB⁴).")
    parser.add_argument("--output-dir", default=None,
                        help="If set, save rmsds.npy (N, S) and top1_rmsd.npy (N,) here.")
    args = parser.parse_args()
    main(args.results_dir, args.scoring, args.output_dir)
