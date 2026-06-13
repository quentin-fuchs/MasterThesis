"""Evaluate top-1 PoseBusters performance using SigmaDock's built-in statistics pipeline.

Aggregates posebusters.pt (and optionally rescoring.pt) across all seed_* directories,
ranks poses by the selected scoring method, and reports top-1 success rate at RMSD < 2 Å
with and without PoseBusters validity filter.

Three ranking modes are supported:
  heuristic (default) — paper's exact method (Appendix F.2): si = −Affinity × pb_mean^4,
               where pb_mean is the mean of 5 physicochemical PoseBusters checks.
               Requires both posebusters.pt and rescoring.pt.
  pb         — mean pass rate of 5 PB checks only; no Vinardo needed.
  vinardo    — Vinardo affinity from gnina alone. Requires rescoring.pt.

Args (CLI):
    results_dir: Path to the model directory containing seed_* subdirectories.
    --scoring:   "heuristic" (default), "pb", or "vinardo".
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

# The 5 PB checks used in Appendix F.2 of the SigmaDock paper.
PB_PAPER_CHECKS = [
    "bond_lengths",
    "bond_angles",
    "tetrahedral_chirality",
    "internal_steric_clash",
    "minimum_distance_to_protein",
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

    needs_rescoring = scoring in ("heuristic", "vinardo")
    if needs_rescoring:
        has_scores = any((p / "rescoring.pt").exists() for p in seed_dirs)
        if not has_scores:
            print(f"ERROR: --scoring {scoring} requires rescoring.pt (run gnina_rescore.sh first)")
            sys.exit(1)
        n_scored = sum(1 for p in seed_dirs if (p / "rescoring.pt").exists())

    if scoring == "heuristic":
        print(
            f"Ranking: paper heuristic — Affinity × PB^4 ({len(PB_PAPER_CHECKS)} checks) "
            f"— {n_scored}/{len(seed_dirs)} seeds scored"
        )
        scoring_config = {
            "score_name": "Affinity",
            "pb_checks": PB_PAPER_CHECKS,
            "score_bias": 0,
            "pb_exponent": 4,
        }
        total_scores, _, _ = collect_scores(seed_dirs, dataset_keys, scoring="vinardo")
    elif scoring == "pb":
        print(f"Ranking: PB-check mean ({len(PB_PAPER_CHECKS)} checks, no Vinardo)")
        scoring_config = {"pb_checks": PB_PAPER_CHECKS}
        total_scores = None
    elif scoring == "vinardo":
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
        choices=["heuristic", "pb", "vinardo"],
        default="heuristic",
        help=(
            "Ranking method: 'heuristic' (paper method, Affinity × PB^4, default), "
            "'pb' (PB mean only), or 'vinardo' (Vinardo affinity only)."
        ),
    )
    args = parser.parse_args()
    main(args.results_dir, args.scoring)
