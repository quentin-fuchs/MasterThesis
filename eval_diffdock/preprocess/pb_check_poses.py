"""
PoseBusters stereochemical checks for DiffDock rank1–rank5 poses.

Runs PoseBusters in "dock" mode (no reference crystal required) on the top-5
poses of each complex and records the five stereochemical check columns used by
the SigmaDock mixed-score heuristic:

    bond_lengths, bond_angles, tetrahedral_chirality,
    internal_steric_clash, minimum_distance_to_protein

For each rank, stores the per-check boolean and the average p_i across all five
checks, which feeds directly into compute_mixed_score() in
collect_gnina_rescoring.py.

Designed to run as a SLURM array job alongside gnina_rescore_poses.py:
pass --chunk_idx to select a slice of the 305-complex list.

Args (CLI):
    --results_dir: Path to pb_evaluate_v2_merged directory.
    --data_dir:    Path to posebusters_benchmark_set directory.
    --chunk_idx:   Integer 0-(n_chunks-1), selects which slice to process.
    --n_chunks:    Total number of chunks (default 6).
    --ranks:       Comma-separated ranks to check (default "1,2,3,4,5").

Returns:
    Writes <results_dir>/<complex_id>/pb_checks.json per complex.
    Exits non-zero if posebusters is not importable.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

# Five stereochemical checks used in the SigmaDock mixed-score heuristic
# (SigmaDock paper, Section 3.1 / statistics.py)
SCORING_PB_CHECKS = [
    "bond_lengths",
    "bond_angles",
    "tetrahedral_chirality",
    "internal_steric_clash",
    "minimum_distance_to_protein",
]

RESULTS_DIR = "/home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged/poses"
DATA_DIR = "/home/qf226/rds/hpc-work/data/posebusters_benchmark_set"


def check_pose(
    ligand_sdf: Path,
    protein_pdb: Path,
    buster,
) -> dict | None:
    """
    Run PoseBusters on a single pose and return the check values.

    Args:
        ligand_sdf:  Path to the ligand SDF file.
        protein_pdb: Path to the receptor PDB file.
        buster:      Initialised PoseBusters instance (dock config).

    Returns:
        Dict with one bool per check in SCORING_PB_CHECKS plus "p" (their mean),
        or None if PoseBusters raises an exception.
    """
    try:
        df = buster.bust(mol_pred=str(ligand_sdf), mol_cond=str(protein_pdb))
    except Exception as exc:
        print(f"  [WARN] PoseBusters failed: {exc}")
        return None

    if df.empty:
        return None

    row = df.iloc[0]
    result = {}
    for check in SCORING_PB_CHECKS:
        if check in row.index:
            result[check] = bool(row[check])
        else:
            print(f"  [WARN] check '{check}' not found in PoseBusters output")
            result[check] = None

    valid_vals = [v for v in result.values() if v is not None]
    result["p"] = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0

    return result


def pb_check_complex(
    complex_dir: Path,
    data_dir: Path,
    ranks: list[int],
    buster,
) -> dict:
    """
    Run PoseBusters checks on the top-N poses of a single complex.

    Args:
        complex_dir: Path to the complex subdirectory in results.
        data_dir:    Root of the PoseBusters dataset.
        ranks:       List of rank integers to check.
        buster:      Initialised PoseBusters instance.

    Returns:
        Dict mapping "rank{N}" -> {check: bool, ..., "p": float}.
        Returns an empty dict if the protein PDB is missing.
    """
    complex_id = complex_dir.name
    protein_pdb = data_dir / complex_id / f"{complex_id}_protein.pdb"

    if not protein_pdb.exists():
        print(f"  [ERROR] protein not found: {protein_pdb}")
        return {}

    result = {}
    for r in ranks:
        ligand_sdf = complex_dir / f"rank{r}.sdf"
        if not ligand_sdf.exists():
            print(f"  [WARN] rank{r}.sdf missing, skipping")
            continue

        checks = check_pose(ligand_sdf, protein_pdb, buster)
        if checks is None:
            result[f"rank{r}"] = {c: None for c in SCORING_PB_CHECKS}
            result[f"rank{r}"]["p"] = 0.0
        else:
            result[f"rank{r}"] = checks

        p = result[f"rank{r}"]["p"]
        print(f"    rank{r}: p={p:.2f}  {result[f'rank{r}']}")

    return result


def get_complex_dirs(results_dir: Path) -> list[Path]:
    """Return sorted list of complex subdirectories matching the PDB ID pattern."""
    return sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and re.match(r"^[0-9A-Z]{4}_", d.name)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PoseBusters checks on DiffDock rank1-5 poses.")
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--n_chunks", type=int, default=6)
    parser.add_argument("--ranks", default=",".join(str(i) for i in range(1, 41)))
    args = parser.parse_args()

    try:
        from posebusters import PoseBusters
    except ImportError:
        print("[ERROR] posebusters not importable — activate the sigmadock env")
        sys.exit(1)

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    ranks = [int(r) for r in args.ranks.split(",")]

    all_complexes = get_complex_dirs(results_dir)
    chunk_size = math.ceil(len(all_complexes) / args.n_chunks)
    start = args.chunk_idx * chunk_size
    chunk = all_complexes[start : start + chunk_size]

    print(
        f"Chunk {args.chunk_idx}/{args.n_chunks}: {len(chunk)} complexes "
        f"(indices {start}–{start + len(chunk) - 1} of {len(all_complexes)})"
    )
    print(f"PB checks: {SCORING_PB_CHECKS}")

    buster = PoseBusters(config="dock")

    n_done, n_skipped, n_failed = 0, 0, 0

    for complex_dir in chunk:
        out_path = complex_dir / "pb_checks.json"
        if out_path.exists():
            print(f"[SKIP] {complex_dir.name} — pb_checks.json already exists")
            n_skipped += 1
            continue

        print(f"Checking {complex_dir.name} ...")
        checks = pb_check_complex(complex_dir, data_dir, ranks, buster)

        if not checks:
            n_failed += 1
            continue

        with open(out_path, "w") as f:
            json.dump(checks, f, indent=2)
        n_done += 1

    print(f"\nDone.  checked={n_done}  skipped={n_skipped}  failed={n_failed}")


if __name__ == "__main__":
    main()
