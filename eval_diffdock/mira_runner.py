"""Batch MIRA evaluation pipeline for DiffDock PoseBusters results.

Wraps molcalib.mira with DiffDock-specific data loading and multiprocessing.
"""

import warnings
from multiprocessing import Pool

import numpy as np
import torch

from molcalib.mira import mira_null, mira_score, _mira_euclidean
from eval_diffdock.loader import (
    load_crystal_coords, load_sample_coords, load_protein_ca_coords,
)

warnings.filterwarnings("ignore")


def _mira_symrmsd_worker(args):
    """Multiprocessing worker for symRMSD MIRA on one complex.

    Args:
        args: (pdb_id, results_index, data_dir, num_runs, seed)

    Returns:
        (pdb_id, score_or_nan, error_str_or_None)
    """
    pdb_id, results_index, data_dir, num_runs, seed = args
    warnings.filterwarnings("ignore")
    try:
        from molcalib.prior import prepare_reference_template
        crystal_mol, all_crystal = load_crystal_coords(pdb_id, data_dir)
        crystal_coords = all_crystal[0]
        samples = load_sample_coords(pdb_id, results_index)
        ca_coords = load_protein_ca_coords(pdb_id, data_dir)
        template_mol, rot_bonds = prepare_reference_template(crystal_mol)
    except Exception as exc:
        return pdb_id, float("nan"), f"load error: {exc}"
    if len(samples) < 2:
        return pdb_id, float("nan"), "too few samples"
    try:
        rng = np.random.default_rng(seed)
        score = mira_score(
            crystal_mol, crystal_coords, samples,
            template_mol, rot_bonds, ca_coords,
            num_runs=num_runs, rng=rng,
        )
        return pdb_id, score, None
    except Exception as exc:
        return pdb_id, float("nan"), f"compute error: {exc}"


def compute_mira_scores(
    complex_names,
    results_index,
    data_dir,
    num_runs=20,
    verbose=True,
    metric="symrmsd",
    seed=42,
    n_workers=1,
):
    """Compute per-complex MIRA scores over the full DiffDock test set.

    Args:
        complex_names: iterable of PDB ID strings.
        results_index: dict from build_results_index().
        data_dir: root data directory.
        num_runs: Monte Carlo draws per complex.
        verbose: print progress every 20 complexes.
        metric: "euclidean", "rmsd", or "symrmsd".
        seed: master random seed for symrmsd (ignored for euclidean/rmsd).
        n_workers: parallel workers (symrmsd only).

    Returns:
        (names_out, scores): numpy arrays of length n_valid.
    """
    use_symrmsd = metric == "symrmsd"
    if not use_symrmsd:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = None

    complex_names = list(complex_names)
    n = len(complex_names)
    child_seeds = np.random.SeedSequence(seed).spawn(n) if use_symrmsd else None
    names_out, scores_out = [], []
    skipped = 0

    if use_symrmsd and n_workers > 1:
        work = [
            (pdb_id, results_index, data_dir, num_runs, child_seeds[i])
            for i, pdb_id in enumerate(complex_names)
        ]
        n_done = 0
        with Pool(processes=n_workers) as pool:
            for pdb_id, score, err in pool.imap_unordered(_mira_symrmsd_worker, work):
                if verbose and n_done % 20 == 0:
                    print(f"  [{n_done}/{n}] {pdb_id} ...", flush=True)
                n_done += 1
                if err is not None:
                    if verbose:
                        print(f"    Skipping {pdb_id}: {err}", flush=True)
                    skipped += 1
                elif np.isfinite(score):
                    names_out.append(pdb_id)
                    scores_out.append(score)
                else:
                    skipped += 1
    else:
        from molcalib.prior import prepare_reference_template
        for i, pdb_id in enumerate(complex_names):
            if verbose and i % 20 == 0:
                print(f"  [{i}/{n}] {pdb_id} ...", flush=True)
            try:
                crystal_mol, all_crystal = load_crystal_coords(pdb_id, data_dir)
                crystal_coords = all_crystal[0]
                samples = load_sample_coords(pdb_id, results_index)
            except Exception as exc:
                if verbose:
                    print(f"    Skipping {pdb_id}: {exc}", flush=True)
                skipped += 1
                continue

            if use_symrmsd:
                try:
                    ca_coords = load_protein_ca_coords(pdb_id, data_dir)
                    template_mol, rot_bonds = prepare_reference_template(crystal_mol)
                except Exception as exc:
                    if verbose:
                        print(f"    Skipping {pdb_id} (setup): {exc}", flush=True)
                    skipped += 1
                    continue
                rng = np.random.default_rng(child_seeds[i])
                score = mira_score(
                    crystal_mol, crystal_coords, samples,
                    template_mol, rot_bonds, ca_coords,
                    num_runs=num_runs, rng=rng,
                )
            else:
                score = _mira_euclidean(crystal_coords, samples, num_runs, device, metric)

            if np.isfinite(score):
                names_out.append(pdb_id)
                scores_out.append(score)
            else:
                skipped += 1

    if verbose:
        S = 40
        print(f"Done. {len(scores_out)} evaluated, {skipped} skipped.")
        print(f"Reference (S={S}): {mira_null(S):.4f}")

    return np.array(names_out), np.array(scores_out, dtype=float)


def compute_rmsd_accuracy(
    complex_names,
    results_index,
    data_dir,
    thresholds=(2.0, 5.0),
    verbose=True,
):
    """Compute per-complex top-1 and any-sample RMSD accuracy.

    Args:
        complex_names: iterable of PDB ID strings.
        results_index: dict from build_results_index().
        data_dir: root data directory.
        thresholds: RMSD thresholds in Angstroms.
        verbose: print progress every 20 complexes.

    Returns:
        (names_out, min_rmsds, fracs): arrays of length n_valid.
          fracs has shape (n_valid, len(thresholds)).
    """
    from molcalib.distances import rmsd as _rmsd

    complex_names = list(complex_names)
    n = len(complex_names)
    names_out, min_rmsds, fracs = [], [], []
    skipped = 0

    for i, pdb_id in enumerate(complex_names):
        if verbose and i % 20 == 0:
            print(f"  [{i}/{n}] {pdb_id} ...", flush=True)
        try:
            _, all_crystal = load_crystal_coords(pdb_id, data_dir)
            crystal_coords = all_crystal[0]
            samples = load_sample_coords(pdb_id, results_index)
        except Exception as exc:
            if verbose:
                print(f"    Skipping {pdb_id}: {exc}", flush=True)
            skipped += 1
            continue
        if len(samples) == 0:
            skipped += 1
            continue

        rmsds = np.array([_rmsd(s, crystal_coords) for s in samples])
        names_out.append(pdb_id)
        min_rmsds.append(rmsds.min())
        fracs.append([(rmsds < t).mean() for t in thresholds])

    if verbose:
        print(f"Done. {len(names_out)} evaluated, {skipped} skipped.")
        if names_out:
            mr = np.array(min_rmsds)
            for j, t in enumerate(thresholds):
                f = np.array(fracs)[:, j]
                print(f"  Top-1 acc (<{t:.0f}Å): {(mr < t).mean():.3f} | "
                      f"Any-sample acc (<{t:.0f}Å): {f.mean():.3f}")

    return (
        np.array(names_out),
        np.array(min_rmsds, dtype=float),
        np.array(fracs, dtype=float),
    )


def load_poses(pdb_id: str, results_index: dict, data_dir: str) -> list:
    """Load heavy-atom coordinate arrays for all DiffDock samples of one complex.

    Args:
        pdb_id: PDB identifier string.
        results_index: dict from build_results_index().
        data_dir: root data directory (unused here, kept for API symmetry).

    Returns:
        list of (N_atoms, 3) numpy arrays, one per sample.
    """
    return load_sample_coords(pdb_id, results_index)


def compute_mira_one_complex(
    crystal: np.ndarray,
    samples: list,
    num_runs: int = 100,
    device=None,
    metric: str = "rmsd",
) -> float:
    """Compute MIRA score for a single complex.

    Args:
        crystal: (N_atoms, 3) crystal ligand coordinates.
        samples: list of (N_atoms, 3) predicted coordinate arrays.
        num_runs: Monte Carlo center draws.
        device: torch device (used for euclidean/rmsd metrics).
        metric: "euclidean" or "rmsd".

    Returns:
        MIRA score as a float (nan if computation fails).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(samples) < 2:
        return float("nan")
    try:
        return float(_mira_euclidean(crystal, samples, num_runs, device, metric))
    except Exception:
        return float("nan")
