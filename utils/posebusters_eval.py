"""
PoseBusters-based pose validity filtering for DiffDock.

Applies the PoseBusters physical/chemical validity checks to DiffDock predicted
poses. Uses the "dock" configuration which does not require a reference crystal
pose, checking only geometric, chemical, and intermolecular constraints.

Checks performed (~20 total):
  Chemical:        RDKit sanitization, connectivity, molecular formula/bonds,
                   tetrahedral chirality, double-bond stereochemistry.
  Intramolecular:  bond lengths/angles within distance-geometry bounds, ring
                   planarity, double-bond flatness, no internal steric clash,
                   energy ratio < 100× ensemble average.
  Intermolecular:  minimum protein-ligand distance ratio > 0.75 × vdW sum,
                   volume overlap with protein < 7.5%.

Reference: Buttenschoen M., Morris G.M., Deane C.M. "PoseBusters: AI-based
docking methods fail to generate physically valid poses or generalise to novel
sequences." Chemical Science 15, 3130-3139 (2024). DOI: 10.1039/D3SC04185A
"""

import json
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem

warnings.filterwarnings("ignore")


def run_posebusters(
    complex_names,
    results_dir: str,
    data_dir: str,
    config: str = "dock",
    cache_path: str = None,
    verbose: bool = True,
    max_complexes: int = None,
    protein_suffix: str = "_protein_processed.pdb",
) -> dict:
    """Run PoseBusters on all DiffDock predicted poses for the test set.

    Iterates over each complex, passes all rank*.sdf files (excluding
    *_confidence*.sdf duplicates) to PoseBusters in a single bust() call per
    complex, and records which poses pass all validity checks.

    Args:
        complex_names: iterable of PDB ID strings.
        results_dir: directory containing per-complex subdirectories with
            rank*.sdf files (e.g. testset_eval_merged/).
        data_dir: root directory containing per-complex subdirectories, each
            with a protein PDB file named {pdb_id}{protein_suffix}.
        config: PoseBusters mode. "dock" checks geometry and protein clashes
            without requiring a reference crystal pose.
        cache_path: if given and the file exists with >0 entries, load cached
            results; if the file does not exist or is empty, run PoseBusters
            and save to this path.
        verbose: print progress every 20 complexes.
        max_complexes: if given, process only the first N complexes. Useful for
            quick tests before running the full set.
        protein_suffix: filename suffix for the protein PDB file inside each
            complex subdirectory. PDBBind uses "_protein_processed.pdb";
            PoseBusters benchmark uses "_protein.pdb".

    Returns:
        dict mapping pdb_id -> {
            "valid_ranks":   list[str] rank filenames (e.g. "rank1.sdf") that
                             pass all PoseBusters checks,
            "n_total":       int total number of rank files found,
            "check_failures": dict[str, int] per-check count of failures across
                             all poses for this complex.
        }
    """
    from posebusters import PoseBusters

    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) > 0:
            if verbose:
                print(f"Loaded PoseBusters cache ({len(cached)} complexes) from {cache_path}")
            return cached
        if verbose:
            print(f"Cache at {cache_path} is empty — recomputing.")

    buster = PoseBusters(config=config)
    results = {}
    complex_names = list(complex_names)
    if max_complexes is not None:
        complex_names = complex_names[:max_complexes]
    n = len(complex_names)
    skipped = 0

    for i, pdb_id in enumerate(complex_names):
        if verbose and i % 20 == 0:
            print(f"  [{i}/{n}] {pdb_id} ...", flush=True)

        complex_dir = Path(results_dir) / pdb_id
        protein_file = Path(data_dir) / pdb_id / f"{pdb_id}{protein_suffix}"

        if not complex_dir.exists() or not protein_file.exists():
            if verbose:
                print(f"    Skipping {pdb_id}: missing directory or protein file")
            skipped += 1
            continue

        plain = [f for f in complex_dir.iterdir()
                 if f.name.startswith("rank") and f.name.endswith(".sdf")
                 and "_confidence" not in f.name]
        if len(plain) > 1:
            rank_files = sorted(plain, key=lambda f: int(f.stem.replace("rank", "")))
        else:
            # inference.py only writes rank1.sdf without confidence suffix;
            # all other ranks are only stored as rank*_confidence*.sdf
            rank_files = sorted(
                [f for f in complex_dir.iterdir()
                 if f.name.startswith("rank") and "_confidence" in f.name
                 and f.name.endswith(".sdf")],
                key=lambda f: int(f.name.split("_confidence")[0].replace("rank", "")),
            )

        if not rank_files:
            skipped += 1
            continue

        valid_ranks = []
        check_failures = {}
        n_processed = 0

        for sdf_file in rank_files:
            try:
                df = buster.bust(
                    mol_pred=str(sdf_file),
                    mol_cond=str(protein_file),
                )
                bool_cols = [c for c in df.columns if df[c].dtype == bool]
                if not bool_cols or df.empty:
                    continue
                is_valid = bool(df[bool_cols].all(axis=1).iloc[0])
                if is_valid:
                    valid_ranks.append(sdf_file.name)
                for col in bool_cols:
                    if not bool(df[col].iloc[0]):
                        check_failures[col] = check_failures.get(col, 0) + 1
                n_processed += 1
            except Exception:
                pass

        results[pdb_id] = {
            "valid_ranks": valid_ranks,
            "n_total": n_processed or len(rank_files),
            "check_failures": check_failures,
        }

    if verbose:
        n_valid = sum(len(v["valid_ranks"]) for v in results.values())
        n_total = sum(v["n_total"] for v in results.values())
        print(f"Done. {len(results)} complexes processed, {skipped} skipped.")
        print(f"Valid poses: {n_valid}/{n_total} ({n_valid/n_total*100:.1f}%)")

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(results, f)
        if verbose:
            print(f"Saved cache to {cache_path}")

    return results


def load_pb_filtered_coords(pdb_id: str, valid_ranks: list, results_dir: str) -> list:
    """Load heavy-atom coordinates for PoseBusters-valid DiffDock poses only.

    Args:
        pdb_id: PDB identifier string.
        valid_ranks: list of rank filename strings that passed PoseBusters
            (e.g. ["rank1.sdf", "rank3.sdf"]).
        results_dir: directory containing per-complex subdirectories.

    Returns:
        list of (N_atoms, 3) numpy arrays, one per valid pose. Poses whose SDF
        file cannot be parsed are silently dropped.
    """
    coords_list = []
    for rank_name in valid_ranks:
        sdf_path = Path(results_dir) / pdb_id / rank_name
        mol = Chem.SDMolSupplier(str(sdf_path), removeHs=True)[0]
        if mol is not None:
            coords_list.append(mol.GetConformer().GetPositions())
    return coords_list


def _rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Plain unaligned RMSD between two (N, 3) coordinate arrays."""
    return float(np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1))))


def compute_rmsd_accuracy_filtered(
    complex_names,
    pb_results: dict,
    results_dir: str,
    data_dir: str,
    thresholds: tuple = (2.0, 5.0),
    verbose: bool = True,
) -> tuple:
    """Compute RMSD accuracy metrics using only PoseBusters-valid poses.

    For each complex, considers only the subset of DiffDock poses that passed
    all PoseBusters checks. Complexes with no valid poses are excluded entirely.

    Uses plain unaligned RMSD, consistent with the Euclidean-distance metric
    MIRA uses internally.

    Args:
        complex_names: iterable of PDB ID strings.
        pb_results: output of run_posebusters().
        results_dir: directory containing per-complex subdirectories.
        data_dir: root directory containing PDBBind_processed/.
        thresholds: RMSD thresholds in Angstroms.
        verbose: print progress every 20 complexes.

    Returns:
        (names_out, min_rmsds, fracs): three arrays.
          names_out: (n_valid,) PDB IDs of successfully evaluated complexes.
          min_rmsds: (n_valid,) minimum RMSD over all PB-valid samples.
          fracs:     (n_valid, len(thresholds)) fraction of valid samples within
                     each threshold.
    """
    from utils.tarp_eval import load_crystal_coords

    complex_names = list(complex_names)
    n = len(complex_names)
    names_out, min_rmsds, fracs = [], [], []
    skipped = 0

    for i, pdb_id in enumerate(complex_names):
        if verbose and i % 20 == 0:
            print(f"  [{i}/{n}] {pdb_id} ...", flush=True)

        pb = pb_results.get(pdb_id, {})
        valid_ranks = pb.get("valid_ranks", [])

        if not valid_ranks:
            skipped += 1
            continue

        try:
            _, crystal = load_crystal_coords(pdb_id, data_dir)
        except Exception as exc:
            if verbose:
                print(f"    Skipping {pdb_id}: {exc}")
            skipped += 1
            continue

        samples = load_pb_filtered_coords(pdb_id, valid_ranks, results_dir)
        if not samples:
            skipped += 1
            continue

        rmsds = np.array([_rmsd(s, crystal) for s in samples])
        names_out.append(pdb_id)
        min_rmsds.append(rmsds.min())
        fracs.append([(rmsds < t).mean() for t in thresholds])

    if verbose:
        print(f"Done. {len(names_out)} complexes evaluated, {skipped} skipped "
              f"(no PB-valid poses or missing crystal).")
        if names_out:
            mr = np.array(min_rmsds)
            for j, t in enumerate(thresholds):
                f = np.array(fracs)[:, j]
                print(f"  Best-of-valid acc (<{t:.0f}Å): {(mr < t).mean():.3f} "
                      f"| Any-valid acc: {f.mean():.3f}")

    return (
        np.array(names_out),
        np.array(min_rmsds, dtype=float),
        np.array(fracs, dtype=float),
    )


def compute_mira_filtered(
    complex_names,
    pb_results: dict,
    results_dir: str,
    data_dir: str,
    num_runs: int = 100,
    verbose: bool = True,
    device=None,
    metric: str = "rmsd",
) -> tuple:
    """Compute MIRA calibration scores using only PoseBusters-valid poses.

    Runs MIRA independently for each complex (T=1 per call) using the subset
    of DiffDock samples that passed all PoseBusters validity checks. Complexes
    where fewer than 2 poses pass PB are excluded (MIRA requires ≥2 samples).

    Args:
        complex_names: iterable of PDB ID strings.
        pb_results: output of run_posebusters().
        results_dir: directory containing per-complex subdirectories.
        data_dir: root directory containing PDBBind_processed/.
        num_runs: Monte Carlo center draws per complex (100 gives stable scores).
        verbose: print progress every 20 complexes.
        device: torch device; auto-detected if None.
        metric: "euclidean" (scaled RMSD, default used in TARP) or "rmsd"
            (per-atom RMSD, consistent with unfiltered mira_eval).

    Returns:
        (names_out, scores): two numpy arrays.
          names_out: (n_valid,) PDB IDs of successfully evaluated complexes.
          scores:    (n_valid,) per-complex MIRA scores.
    """
    from mira_score import get_device
    from utils.mira_eval import _mira_one_complex, mira_null
    from utils.tarp_eval import load_crystal_coords

    if device is None:
        device = get_device()

    complex_names = list(complex_names)
    n = len(complex_names)
    names_out, scores = [], []
    skipped = 0

    for i, pdb_id in enumerate(complex_names):
        if verbose and i % 20 == 0:
            print(f"  [{i}/{n}] {pdb_id} ...", flush=True)

        pb = pb_results.get(pdb_id, {})
        valid_ranks = pb.get("valid_ranks", [])

        if len(valid_ranks) < 2:
            skipped += 1
            continue

        try:
            _, crystal = load_crystal_coords(pdb_id, data_dir)
        except Exception as exc:
            if verbose:
                print(f"    Skipping {pdb_id}: {exc}")
            skipped += 1
            continue

        samples = load_pb_filtered_coords(pdb_id, valid_ranks, results_dir)
        if len(samples) < 2:
            skipped += 1
            continue

        score = _mira_one_complex(crystal, samples, num_runs=num_runs,
                                  device=device, metric=metric)
        if not np.isnan(score):
            names_out.append(pdb_id)
            scores.append(score)
        else:
            skipped += 1

    if verbose:
        S_typical = len(list(pb_results.values())[0].get("valid_ranks", [])) or 40
        ref = mira_null(S_typical)
        print(f"Done. {len(scores)} complexes evaluated, {skipped} skipped "
              f"(< 2 PB-valid poses or missing crystal).")
        print(f"Reference (perfect calibration, S≈{S_typical}): {ref:.4f}")

    return np.array(names_out), np.array(scores, dtype=float)
