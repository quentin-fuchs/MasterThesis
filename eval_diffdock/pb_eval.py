"""PoseBusters-based pose validity filtering for DiffDock.

Applies PoseBusters physical/chemical validity checks to DiffDock predicted
poses using the "dock" configuration (no reference crystal pose required).

Reference: Buttenschoen M., Morris G.M., Deane C.M. "PoseBusters: AI-based
docking methods fail to generate physically valid poses or generalise to novel
sequences." Chemical Science 15, 3130-3139 (2024). DOI: 10.1039/D3SC04185A
"""

import json
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem

from eval_diffdock.loader import get_rank_files

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

    Args:
        complex_names: iterable of PDB ID strings.
        results_dir: directory containing per-complex subdirectories with
            rank*.sdf files.
        data_dir: root directory containing per-complex protein PDB files.
        config: PoseBusters mode. "dock" checks geometry and protein clashes
            without requiring a reference crystal pose.
        cache_path: if given and the file exists, load cached results; if not,
            run PoseBusters and save to this path.
        verbose: print progress every 20 complexes.
        max_complexes: if given, process only the first N complexes.
        protein_suffix: filename suffix for the protein PDB file.

    Returns:
        dict mapping pdb_id → {
            "valid_ranks":    list[str] rank filenames that pass all checks,
            "n_total":        int total number of rank files found,
            "check_failures": dict[str, int] per-check failure counts.
        }
    """
    cached = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if verbose and cached:
            print(f"Loaded PoseBusters cache ({len(cached)} complexes) from {cache_path}")
        if not cached:
            if verbose:
                print(f"Cache at {cache_path} is empty — recomputing.")

    complex_names = list(complex_names)
    if max_complexes is not None:
        complex_names = complex_names[:max_complexes]
    complex_names = [n for n in complex_names if n not in cached]
    if not complex_names:
        return cached

    from posebusters import PoseBusters
    buster = PoseBusters(config=config)
    results = dict(cached)
    n = len(complex_names)
    skipped = 0
    if verbose:
        print(f"Running PoseBusters on {n} new complexes ...")

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

        rank_files = get_rank_files(complex_dir)
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
        new_results = {k: v for k, v in results.items() if k not in cached}
        n_valid = sum(len(v["valid_ranks"]) for v in new_results.values())
        n_total = sum(v["n_total"] for v in new_results.values())
        print(f"Done. {len(new_results)} new complexes processed, {skipped} skipped.")
        if n_total:
            print(f"Valid poses (new): {n_valid}/{n_total} ({n_valid/n_total*100:.1f}%)")
        print(f"Total in cache: {len(results)} complexes")

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
        valid_ranks: list of rank filename strings that passed PoseBusters.
        results_dir: directory containing per-complex subdirectories.

    Returns:
        list of (N_atoms, 3) numpy arrays, one per valid pose.
    """
    coords_list = []
    for rank_name in valid_ranks:
        sdf_path = Path(results_dir) / pdb_id / rank_name
        mol = Chem.SDMolSupplier(str(sdf_path), removeHs=True)[0]
        if mol is not None:
            coords_list.append(mol.GetConformer().GetPositions())
    return coords_list


def compute_rmsd_accuracy_filtered(
    complex_names,
    pb_results: dict,
    results_dir: str,
    data_dir: str,
    thresholds: tuple = (2.0, 5.0),
    verbose: bool = True,
) -> tuple:
    """Compute RMSD accuracy metrics using only PoseBusters-valid poses.

    Args:
        complex_names: iterable of PDB ID strings.
        pb_results: output of run_posebusters().
        results_dir: directory containing per-complex subdirectories.
        data_dir: root directory containing crystal structures.
        thresholds: RMSD thresholds in Angstroms.
        verbose: print progress every 20 complexes.

    Returns:
        (names_out, min_rmsds, fracs): three arrays.
    """
    from molcalib.distances import rmsd as _rmsd
    from eval_diffdock.loader import load_crystal_coords

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
            _, all_crystal = load_crystal_coords(pdb_id, data_dir)
            crystal = all_crystal[0]
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
        print(f"Done. {len(names_out)} complexes evaluated, {skipped} skipped.")
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
    metric: str = "rmsd",
) -> tuple:
    """Compute MIRA calibration scores using only PoseBusters-valid poses.

    Args:
        complex_names: iterable of PDB ID strings.
        pb_results: output of run_posebusters().
        results_dir: directory containing per-complex subdirectories.
        data_dir: root directory containing crystal structures.
        num_runs: Monte Carlo center draws per complex.
        verbose: print progress every 20 complexes.
        metric: "euclidean" or "rmsd".

    Returns:
        (names_out, scores): two numpy arrays.
    """
    import torch
    from molcalib.mira import mira_null
    from eval_diffdock.loader import load_crystal_coords
    from eval_diffdock.mira_runner import compute_mira_one_complex

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            _, all_crystal = load_crystal_coords(pdb_id, data_dir)
            crystal = all_crystal[0]
        except Exception as exc:
            if verbose:
                print(f"    Skipping {pdb_id}: {exc}")
            skipped += 1
            continue

        samples = load_pb_filtered_coords(pdb_id, valid_ranks, results_dir)
        if len(samples) < 2:
            skipped += 1
            continue

        score = compute_mira_one_complex(crystal, samples, num_runs=num_runs,
                                         device=device, metric=metric)
        if not np.isnan(score):
            names_out.append(pdb_id)
            scores.append(score)
        else:
            skipped += 1

    if verbose:
        S_typical = len(list(pb_results.values())[0].get("valid_ranks", [])) or 40
        ref = mira_null(S_typical)
        print(f"Done. {len(scores)} complexes evaluated, {skipped} skipped.")
        print(f"Reference (perfect calibration, S≈{S_typical}): {ref:.4f}")

    return np.array(names_out), np.array(scores, dtype=float)
