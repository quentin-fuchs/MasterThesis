"""Evaluate top-1 PoseBusters performance using SigmaDock's built-in statistics pipeline.

Aggregates posebusters.pt (and optionally rescoring.pt) across all seed_* directories,
ranks poses by the selected scoring method, and reports top-1 success rate at RMSD < 2 Å
with and without PoseBusters validity filter.

Two ranking modes are supported:
  pb       (default) — paper's method: mean pass rate of 7 physicochemical PoseBusters
             checks. Requires only posebusters.pt; no external binary needed.
  vinardo  — Vinardo affinity from gnina. Requires rescoring.pt in every seed_* dir
             (produced by gnina_rescore.sh).

Args (CLI):
    results_dir: Path to the model directory containing seed_* subdirectories.
    --scoring:   "pb" (default) or "vinardo".
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sigmadock.chem.statistics import (
    collect_posebusters,
    collect_scores,
    compute_top_k_statistics,
    sort_statistics_for_top_k,
)

# The 7 checks used in the SigmaDock paper's heuristic ranking.
PB_HEURISTIC_CHECKS = [
    "minimum_distance_to_protein",
    "tetrahedral_chirality",
    "internal_energy",
    "internal_steric_clash",
    "double_bond_flatness",
    "bond_lengths",
    "bond_angles",
]


def main(results_dir: str, scoring: str) -> None:
    model_dir = Path(results_dir)

    seed_dirs = sorted(
        [p for p in model_dir.glob("seed_*") if (p / "posebusters.pt").exists()],
        key=lambda p: int(p.name.split("_")[1]),
    )

    if not seed_dirs:
        print(f"No seed directories with posebusters.pt found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(seed_dirs)} seeds")

    first_pb = torch.load(seed_dirs[0] / "posebusters.pt", weights_only=False)
    dataset_keys = sorted(first_pb["rmsds"].keys())
    print(f"Dataset: {len(dataset_keys)} complexes")

    if scoring == "pb":
        print(f"Ranking: PB-check heuristic ({len(PB_HEURISTIC_CHECKS)} checks — paper method)")
        scoring_config = {"pb_checks": PB_HEURISTIC_CHECKS}
        total_scores = None
    elif scoring == "vinardo":
        has_scores = any((p / "rescoring.pt").exists() for p in seed_dirs)
        if not has_scores:
            print("ERROR: --scoring vinardo requires rescoring.pt (run gnina_rescore.sh first)")
            sys.exit(1)
        n_scored = sum(1 for p in seed_dirs if (p / "rescoring.pt").exists())
        print(f"Ranking: Vinardo affinity (ascending) — {n_scored}/{len(seed_dirs)} seeds scored")
        scoring_config = {"score_name": "Affinity"}
        total_scores, _, _ = collect_scores(seed_dirs, dataset_keys, scoring="vinardo")

    total_rmsds, total_pb_checks, total_pb_dicts, missing = collect_posebusters(
        seed_dirs, dataset_keys, verbose=False
    )
    if missing:
        n_missing = sum(len(v) for v in missing.values())
        print(f"Warning: {n_missing} missing seed/complex pairs")

    sorted_rmsds, sorted_pb_checks, *_ = sort_statistics_for_top_k(
        total_rmsds,
        total_pb_checks,
        total_pb_dicts,
        total_scores=total_scores,
        dataset_keys=dataset_keys,
        scoring=scoring,
        **scoring_config,
    )

    avg_passes, avg_passes_with_pb, _ = compute_top_k_statistics(
        sorted_rmsds,
        sorted_pb_checks,
        dataset_keys=dataset_keys,
        ks=[1],
        rmsd_thresholds=[2.0],
    )

    n = len(dataset_keys)
    s = len(seed_dirs)
    print(f"\n=== Top-1 results | {s} seeds | {n} complexes ===")
    print(f"  RMSD < 2 Å:           {avg_passes[1][2.0] * 100:.1f}%")
    print(f"  RMSD < 2 Å + PB pass: {avg_passes_with_pb[1][2.0] * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Path to the model directory containing seed_* subdirectories.")
    parser.add_argument(
        "--scoring",
        choices=["pb", "vinardo"],
        default="pb",
        help="Ranking method: 'pb' (paper method) or 'vinardo' (requires rescoring.pt).",
    )
    args = parser.parse_args()
    main(args.results_dir, args.scoring)
