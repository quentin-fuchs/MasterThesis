"""Batch TARP evaluation pipeline for DiffDock PoseBusters results.

Wraps molcalib.tarp with DiffDock-specific data loading and multiprocessing.
"""

import warnings
from multiprocessing import Pool

import numpy as np

from molcalib.tarp import tarp_fractions
from molcalib.prior import prepare_reference_template
from eval_diffdock.loader import (
    build_results_index, load_crystal_coords,
    load_sample_coords, load_protein_ca_coords,
)

warnings.filterwarnings("ignore")


def _tarp_worker(args):
    """Multiprocessing worker for TARP evaluation of one complex.

    Args:
        args: (pdb_id, results_index, data_dir, K, mode, seed, max_samples)

    Returns:
        (pdb_id, fracs_or_None, error_str_or_None)
    """
    pdb_id, results_index, data_dir, K, mode, seed, max_samples = args
    warnings.filterwarnings("ignore")
    try:
        crystal_mol, all_crystal_coords = load_crystal_coords(pdb_id, data_dir)
        crystal_coords = all_crystal_coords[0]
        sample_coords = load_sample_coords(pdb_id, results_index)
        ca_coords = load_protein_ca_coords(pdb_id, data_dir)
    except Exception as exc:
        return pdb_id, None, f"load error: {exc}"

    if max_samples is not None:
        sample_coords = sample_coords[:max_samples]
    if len(sample_coords) == 0:
        return pdb_id, None, "no valid samples"

    try:
        template_mol, rot_bonds = prepare_reference_template(crystal_mol)
        rng = np.random.default_rng(seed)
        fracs = tarp_fractions(
            crystal_mol, crystal_coords, template_mol, rot_bonds,
            sample_coords, ca_coords, K=K, rng=rng, mode=mode,
        )
        return pdb_id, fracs, None
    except Exception as exc:
        return pdb_id, None, f"compute error: {exc}"


def run_tarp_eval(
    complex_names,
    results_index,
    data_dir,
    K=20,
    mode="rmsd",
    seed=42,
    verbose=True,
    n_workers=1,
    _max_samples=None,
):
    """Run TARP evaluation over all complexes.

    Args:
        complex_names: iterable of PDB ID strings.
        results_index: dict from build_results_index().
        data_dir: root data directory.
        K: number of random reference points per complex.
        mode: "rmsd" (symmetry-corrected) or "centroid".
        seed: master random seed; per-complex seeds via SeedSequence.
        verbose: print progress every 20 complexes.
        n_workers: parallel workers.
        _max_samples: if set, only use this many samples per complex.

    Returns:
        numpy array of shape (n_valid_complexes, K). Use ecp_from_fractions
        and bootstrap_ecp from molcalib.tarp to compute ECP curves.
    """
    complex_names = list(complex_names)
    n = len(complex_names)
    child_seeds = np.random.SeedSequence(seed).spawn(n)
    work = [
        (pdb_id, results_index, data_dir, K, mode, child_seeds[i], _max_samples)
        for i, pdb_id in enumerate(complex_names)
    ]

    rows, skipped, n_done = [], 0, 0

    def _handle(result):
        nonlocal skipped, n_done
        pdb_id, fracs, err = result
        if verbose and n_done % 20 == 0:
            print(f"  [{n_done}/{n}] {pdb_id} ...", flush=True)
        n_done += 1
        if err is not None:
            if verbose:
                print(f"    Skipping {pdb_id}: {err}", flush=True)
            skipped += 1
        elif len(fracs) > 0:
            rows.append(fracs[:K])

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for result in pool.imap(_tarp_worker, work):
                _handle(result)
    else:
        for result in map(_tarp_worker, work):
            _handle(result)

    if verbose:
        print(f"Done. {len(rows)} processed, {skipped} skipped.", flush=True)

    max_k = max((len(r) for r in rows), default=0)
    if max_k == 0:
        return np.empty((0, K))
    out = np.full((len(rows), max_k), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out
