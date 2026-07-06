"""
Per-complex symmetry-corrected RMSD evaluation for DiffDock predictions.

Computes the spyrmsd symmetry-corrected RMSD between each DiffDock predicted
pose and the crystal structure. Returns top-1 RMSD (rank1 vs crystal) and
all-poses RMSD (all S predictions vs crystal), matching the format of
existing PDBBind metrics (top1_rmsd.npy, rmsds.npy).

Typical usage
-------------
>>> from eval_diffdock.rmsd_runner import run_rmsd_eval
>>> names, top1, all_rmsds = run_rmsd_eval(complex_names, results_index, data_dir)
>>> print(f"Top-1 < 2Å: {(top1 < 2).mean() * 100:.1f}%")
"""

import warnings

import numpy as np

from eval_diffdock.loader import load_crystal_coords, load_sample_coords
from molcalib import compute_rmsd_symmetry_multi

warnings.filterwarnings("ignore")


def compute_complex_rmsds(pdb_id, results_index, data_dir, max_samples=40):
    """Compute symmetry-corrected RMSD for all predicted poses of one complex.

    Args:
        pdb_id: PDB identifier string.
        results_index: dict mapping pdb_id to Path of the complex directory.
        data_dir: root data directory containing the crystal ligand SDF.
        max_samples: maximum number of ranked poses to evaluate (default 40).

    Returns:
        numpy array of shape (max_samples,) with RMSD values in Ångströms.
        Entries are NaN for poses that could not be loaded or evaluated.
        Returns None if the crystal structure or predictions cannot be loaded.
    """
    try:
        crystal_mol, all_crystal_coords = load_crystal_coords(pdb_id, data_dir)
        sample_coords = load_sample_coords(pdb_id, results_index)
    except (FileNotFoundError, ValueError, OSError):
        return None

    if not sample_coords:
        return None

    sample_coords = sample_coords[:max_samples]

    # Filter out poses with wrong atom count (use first conformer as reference shape)
    valid_coords = [c for c in sample_coords if c.shape == all_crystal_coords[0].shape]
    if not valid_coords:
        return None

    rmsds_valid = compute_rmsd_symmetry_multi(crystal_mol, all_crystal_coords, valid_coords)

    # Pad back to max_samples with NaN so the output is a fixed-length vector
    out = np.full(max_samples, np.nan)
    out[:len(rmsds_valid)] = rmsds_valid
    return out


def run_rmsd_eval(complex_names, results_index, data_dir,
                  max_samples=40, verbose=True):
    """Compute symmetry-corrected RMSD for all complexes in a test set.

    Args:
        complex_names: iterable of PDB identifier strings.
        results_index: dict mapping pdb_id to Path of the complex directory.
        data_dir: root data directory containing crystal ligand SDF files.
        max_samples: number of ranked poses to evaluate per complex (default 40).
        verbose: if True, print progress every 20 complexes and a final summary.

    Returns:
        (names, top1_rmsds, all_rmsds): three numpy arrays.
            names:       shape (n,)            — complex identifiers.
            top1_rmsds:  shape (n,)            — RMSD of rank-1 pose vs crystal.
            all_rmsds:   shape (n, max_samples) — RMSD of every pose vs crystal;
                         NaN for missing/failed poses.
    """
    complex_names = list(complex_names)
    n = len(complex_names)

    names_out = []
    top1_out = []
    all_out = []
    n_skipped = 0

    for i, pdb_id in enumerate(complex_names):
        if verbose and i % 20 == 0:
            print(f"  [{i}/{n}] {pdb_id} ...", flush=True)

        rmsds = compute_complex_rmsds(pdb_id, results_index, data_dir, max_samples)
        if rmsds is None:
            n_skipped += 1
            continue

        names_out.append(pdb_id)
        top1_out.append(rmsds[0])
        all_out.append(rmsds)

    names_arr = np.array(names_out)
    top1_arr = np.array(top1_out, dtype=float)
    all_arr = np.stack(all_out) if all_out else np.empty((0, max_samples))

    if verbose:
        valid = np.isfinite(top1_arr)
        n_valid = valid.sum()
        print(f"\nEvaluated {n_valid} / {n} complexes ({n_skipped} skipped entirely)")
        for thresh in [2.0, 5.0]:
            frac = (top1_arr[valid] < thresh).mean() * 100
            count = (top1_arr[valid] < thresh).sum()
            print(f"  Top-1 RMSD < {thresh:.0f}Å: {frac:.1f}%  ({count}/{n_valid})")
        # Best-of-S: for each complex take the minimum RMSD across all poses
        best_rmsds = np.nanmin(all_arr, axis=1)
        best_valid = np.isfinite(best_rmsds)
        for thresh in [2.0, 5.0]:
            frac = (best_rmsds[best_valid] < thresh).mean() * 100
            print(f"  Best-of-{max_samples} RMSD < {thresh:.0f}Å: {frac:.1f}%")
        print(f"  Median top-1 RMSD: {np.nanmedian(top1_arr):.2f} Å")

    return names_arr, top1_arr, all_arr
