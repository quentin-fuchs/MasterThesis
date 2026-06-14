"""
Aggregate per-complex gnina rescoring JSONs into a single CSV.

Walks pb_evaluate_v2_merged for rescoring_<scoring>.json files produced by
gnina_rescore_poses.py, builds a flat table (complex_id, rank, confidence,
affinity), and writes it to metrics/rescoring_<scoring>.csv.

If pb_checks.json files are also present (produced by pb_check_poses.py),
the table is extended with the five SigmaDock stereochemical check columns
and the mixed score s_i = -b_i * p_i^beta (SigmaDock paper, beta=4).

Args (CLI):
    --results_dir: Path to pb_evaluate_v2_merged directory.
    --scoring:     Which rescoring file to collect (default "vinardo").
    --beta:        PB penalty exponent for the mixed score (default 4).

Returns:
    Writes <results_dir>/metrics/rescoring_<scoring>.csv.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# Five stereochemical checks used in the SigmaDock mixed-score heuristic
# (SigmaDock paper, Section 3.1). Must match the keys in pb_checks.json.
SCORING_PB_CHECKS = [
    "bond_lengths",
    "bond_angles",
    "tetrahedral_chirality",
    "internal_steric_clash",
    "minimum_distance_to_protein",
]

RESULTS_DIR = "/home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged/poses"


def compute_mixed_score(affinity: float | None, p: float | None, beta: float = 4.0) -> float | None:
    """
    Compute the SigmaDock mixed confidence score for a single pose.

    Implements s_i = -b_i * p_i^beta (SigmaDock paper, Section 3.1), where b_i
    is the Vinardo binding energy (kcal/mol), p_i is the average of the five
    stereochemical PoseBusters checks, and beta controls the validity penalty.
    Higher scores correspond to higher-confidence poses.

    Args:
        affinity: Vinardo binding energy in kcal/mol (lower = more negative = better).
        p:        Average of the five PB check booleans in [0, 1].
        beta:     Exponent on the PB penalty term (default 4, as in the paper).

    Returns:
        Mixed score float, or None if either input is None.
    """
    if affinity is None or p is None:
        return None
    return -affinity * (p ** beta)


def get_complex_dirs(results_dir: Path) -> list[Path]:
    """Return sorted list of complex subdirectories matching the PDB ID pattern."""
    return sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and re.match(r"^[0-9A-Z]{4}_", d.name)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate gnina rescoring results into CSV.")
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--scoring", default="vinardo")
    parser.add_argument("--beta", type=float, default=4.0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_name = f"rescoring_{args.scoring}.json"
    all_complexes = get_complex_dirs(results_dir)

    rows = []
    missing_rescore = []
    n_with_pb = 0

    for complex_dir in all_complexes:
        rescore_path = complex_dir / output_name
        if not rescore_path.exists():
            missing_rescore.append(complex_dir.name)
            continue

        with open(rescore_path) as f:
            rescore = json.load(f)

        pb_path = complex_dir / "pb_checks.json"
        pb_data = {}
        if pb_path.exists():
            with open(pb_path) as f:
                pb_data = json.load(f)
            n_with_pb += 1

        for rank_key, scores in rescore.items():
            affinity = scores.get("affinity")
            pb_rank = pb_data.get(rank_key, {})
            p = pb_rank.get("p") if pb_rank else None

            row = {
                "complex_id": complex_dir.name,
                "rank": int(rank_key.replace("rank", "")),
                "confidence": scores.get("confidence"),
                "affinity": affinity,
                "p": p,
                "mixed_score": compute_mixed_score(affinity, p, args.beta),
            }
            for check in SCORING_PB_CHECKS:
                row[check] = pb_rank.get(check) if pb_rank else None

            rows.append(row)

    columns = (
        ["complex_id", "rank", "confidence", "affinity", "p", "mixed_score"]
        + SCORING_PB_CHECKS
    )
    df = pd.DataFrame(rows, columns=columns)

    metrics_dir = results_dir.parent / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    out_csv = metrics_dir / f"rescoring_{args.scoring}.csv"
    df.to_csv(out_csv, index=False)

    n_complete = len(all_complexes) - len(missing_rescore)
    print(f"Complexes with rescoring : {n_complete}/{len(all_complexes)}")
    print(f"Complexes with PB checks : {n_with_pb}/{n_complete}")
    if missing_rescore:
        preview = ", ".join(missing_rescore[:10])
        suffix = " ..." if len(missing_rescore) > 10 else ""
        print(f"Missing rescoring ({len(missing_rescore)}): {preview}{suffix}")

    failed_poses = df["affinity"].isna().sum()
    if failed_poses:
        print(f"Poses with failed affinity: {failed_poses}")

    print(f"\nAffinity by rank (kcal/mol):")
    print(df.groupby("rank")["affinity"].describe().round(3).to_string())

    if n_with_pb > 0:
        print(f"\nMixed score by rank (beta={args.beta}):")
        print(df.groupby("rank")["mixed_score"].describe().round(3).to_string())

    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
