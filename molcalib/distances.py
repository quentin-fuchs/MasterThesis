"""Distance metrics for molecular pose comparison.

Provides symmetry-corrected RMSD via spyrmsd and centroid distance.
All functions operate on numpy coordinate arrays and RDKit Mol objects;
no file I/O is performed here.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import numpy as np
from spyrmsd import rmsd as spyrmsd_rmsd, molecule as spyrmsd_molecule

warnings.filterwarnings("ignore")


def _spyrmsd_mol(mol):
    """Convert an RDKit Mol to a spyrmsd Molecule."""
    return spyrmsd_molecule.Molecule.from_rdkit(mol)


def compute_rmsd_symmetry(mol, ref_coords, query_coords_list, timeout=4):
    """Symmetry-corrected heavy-atom RMSD between ref_coords and each query.

    Uses spyrmsd's Hungarian matching over molecular graph automorphisms so
    that equivalent atoms (e.g. in symmetric rings) are correctly permuted
    before computing RMSD.

    Args:
        mol: RDKit Mol defining the heavy-atom graph (no Hs).
        ref_coords: numpy array of shape (N_atoms, 3).
        query_coords_list: list of numpy arrays of shape (N_atoms, 3).
        timeout: seconds allowed per symmrmsd call. Molecules with large
            symmetry groups can cause the Hungarian matching to run for
            minutes; calls exceeding this limit return NaN.

    Returns:
        numpy array of shape (len(query_coords_list),). NaN for failed calls.
    """
    spy_mol = _spyrmsd_mol(mol)
    adjacency = spy_mol.adjacency_matrix
    atomicnums = spy_mol.atomicnums
    results = []
    for qc in query_coords_list:
        if qc.shape != ref_coords.shape:
            results.append(np.nan)
            continue
        ex = ThreadPoolExecutor(max_workers=1)
        future = ex.submit(
            spyrmsd_rmsd.symmrmsd,
            ref_coords, qc,
            atomicnums, atomicnums,
            adjacency, adjacency,
        )
        try:
            results.append(future.result(timeout=timeout))
        except FuturesTimeout:
            warnings.warn(
                f"spyrmsd timed out after {timeout}s — likely high symmetry; returning NaN"
            )
            results.append(np.nan)
        except Exception as exc:
            warnings.warn(f"spyrmsd failed: {exc}")
            results.append(np.nan)
        finally:
            ex.shutdown(wait=False)
    return np.array(results)


def compute_rmsd_symmetry_multi(mol, all_ref_coords, query_coords_list, timeout=4):
    """Symmetry-corrected RMSD taking the minimum over multiple crystal conformers.

    Calls compute_rmsd_symmetry once per crystal conformer and returns the
    element-wise minimum. When all_ref_coords has length 1, the result is
    identical to calling compute_rmsd_symmetry directly.

    Args:
        mol: RDKit Mol defining the heavy-atom graph.
        all_ref_coords: list of numpy arrays (N_atoms, 3), one per conformer.
        query_coords_list: list of numpy arrays (N_atoms, 3).
        timeout: passed through to compute_rmsd_symmetry.

    Returns:
        numpy array of shape (len(query_coords_list),).
    """
    per_ref = np.stack([
        compute_rmsd_symmetry(mol, ref, query_coords_list, timeout=timeout)
        for ref in all_ref_coords
    ])
    return np.nanmin(per_ref, axis=0)


def compute_centroid_distance(ref_coords, query_coords_list):
    """Euclidean distance between ligand centroids.

    Args:
        ref_coords: numpy array (N_atoms, 3).
        query_coords_list: list of numpy arrays (N_atoms, 3).

    Returns:
        numpy array of shape (len(query_coords_list),).
    """
    ref_c = ref_coords.mean(axis=0)
    return np.array([np.linalg.norm(q.mean(axis=0) - ref_c) for q in query_coords_list])


def rmsd(coords1, coords2):
    """Plain unaligned RMSD between two (N, 3) coordinate arrays.

    Args:
        coords1: numpy array (N, 3).
        coords2: numpy array (N, 3).

    Returns:
        Scalar RMSD value.
    """
    return float(np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1))))
