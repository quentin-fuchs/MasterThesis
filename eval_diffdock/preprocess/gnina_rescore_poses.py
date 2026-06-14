"""
Post-hoc Vina/Vinardo rescoring of DiffDock poses for the PoseBusters benchmark.

For each complex in the merged results directory, scores rank1–rank5 with gnina
--score_only and saves a per-complex JSON mapping rank -> {confidence, affinity}.
Designed to run as a SLURM array job: pass --chunk_idx to select a slice of the
305-complex list. Results are written alongside the existing pose SDFs so that
collect_gnina_rescoring.py can aggregate them.

Args (CLI):
    --results_dir: Path to pb_evaluate_v2_merged directory.
    --data_dir:    Path to posebusters_benchmark_set directory.
    --chunk_idx:   Integer 0-(n_chunks-1), selects which slice to process.
    --n_chunks:    Total number of chunks (default 6).
    --ranks:       Comma-separated ranks to score (default "1,2,3,4,5").
    --scoring:     Gnina scoring function: "vina" or "vinardo" (default "vinardo").

Returns:
    Writes <results_dir>/<complex_id>/rescoring_<scoring>.json per complex.
    Exits non-zero if gnina is not on PATH.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = "/home/qf226/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged/poses"
DATA_DIR = "/home/qf226/rds/hpc-work/data/posebusters_benchmark_set"


def parse_affinity(stdout: str) -> float | None:
    """Extract Affinity value from gnina --score_only stdout."""
    m = re.search(r"Affinity:\s*([-\d.]+)", stdout)
    return float(m.group(1)) if m else None


def parse_confidence(complex_dir: Path, rank: int) -> float | None:
    """Extract confidence score from the rank{N}_confidence<score>.sdf filename."""
    matches = list(complex_dir.glob(f"rank{rank}_confidence*.sdf"))
    if not matches:
        return None
    m = re.search(r"_confidence([+-]?\d+\.\d+)\.sdf", matches[0].name)
    return float(m.group(1)) if m else None


def score_pose(ligand_sdf: Path, protein_pdb: Path, scoring: str) -> float | None:
    """
    Call gnina --score_only on a single pose.

    Args:
        ligand_sdf:  Path to the ligand SDF file.
        protein_pdb: Path to the receptor PDB file.
        scoring:     Scoring function ("vina" or "vinardo").

    Returns:
        Affinity in kcal/mol, or None if gnina fails.
    """
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    cmd = [
        "gnina",
        "-r", str(protein_pdb),
        "-l", str(ligand_sdf),
        "--autobox_ligand", str(ligand_sdf),
        "--score_only",
        "--scoring", scoring,
        "--cnn_scoring", "none",
        "--no_gpu",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  [WARN] gnina failed (rc={result.returncode}): {result.stderr.strip()[:200]}")
        return None

    return parse_affinity(result.stdout)


def rescore_complex(
    complex_dir: Path,
    data_dir: Path,
    ranks: list[int],
    scoring: str,
) -> dict:
    """
    Score the top-N poses of a single complex.

    Args:
        complex_dir: Path to the complex subdirectory in results.
        data_dir:    Root of the PoseBusters dataset.
        ranks:       List of rank integers to score.
        scoring:     Gnina scoring function.

    Returns:
        Dict mapping "rank{N}" -> {"confidence": float|None, "affinity": float|None}.
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

        confidence = parse_confidence(complex_dir, r)
        affinity = score_pose(ligand_sdf, protein_pdb, scoring)

        result[f"rank{r}"] = {"confidence": confidence, "affinity": affinity}
        affinity_str = f"{affinity:.3f}" if affinity is not None else "FAILED"
        print(f"    rank{r}: confidence={confidence}  affinity={affinity_str}")

    return result


def get_complex_dirs(results_dir: Path) -> list[Path]:
    """Return sorted list of complex subdirectories matching the PDB ID pattern."""
    return sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and re.match(r"^[0-9A-Z]{4}_", d.name)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore DiffDock poses with gnina.")
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--n_chunks", type=int, default=6)
    parser.add_argument("--ranks", default=",".join(str(i) for i in range(1, 41)))
    parser.add_argument("--scoring", default="vinardo", choices=["vina", "vinardo"])
    args = parser.parse_args()

    if shutil.which("gnina") is None:
        print("[ERROR] gnina not found on PATH")
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

    output_name = f"rescoring_{args.scoring}.json"
    n_done, n_skipped, n_failed = 0, 0, 0

    for complex_dir in chunk:
        out_path = complex_dir / output_name
        if out_path.exists():
            print(f"[SKIP] {complex_dir.name} — {output_name} already exists")
            n_skipped += 1
            continue

        print(f"Scoring {complex_dir.name} ...")
        scores = rescore_complex(complex_dir, data_dir, ranks, args.scoring)

        if not scores:
            n_failed += 1
            continue

        with open(out_path, "w") as f:
            json.dump(scores, f, indent=2)
        n_done += 1

    print(f"\nDone.  scored={n_done}  skipped={n_skipped}  failed={n_failed}")


if __name__ == "__main__":
    main()
