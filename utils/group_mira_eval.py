"""
Per-group (translation / rotation / torsion) MIRA calibration scores for
DiffDock.

MIRA (Sharief et al. 2026, arXiv:2605.02014) is a scalar calibration metric.
For each complex it draws T random centers from the prior, picks a random
reference sample as the radius, counts how many other samples lie inside that
ball, and scores whether the crystal pose is also inside. Under perfect
calibration the expected score is:

    mira_null(S) = (2/3) × (S + 1) / S  ≈ 0.683 for S = 40

Score > null → over-dispersed (samples too spread out).
Score < null → mode-collapsed (samples too tightly clustered).

This module reimplements the random-radius MIRA estimator for each of
DiffDock's three diffusion groups using manifold-appropriate distance metrics
and prior distributions that match DiffDock's randomize_position at t = 1:

  Translation  N(Cα_COM, σ² I)    σ = std_ca × prop / 1.73    metric: L2 (Å)
  Rotation     Haar-uniform SO(3)                               metric: geodesic (rad)
  Torsion      Uniform([-π, π])^k independent per bond          metric: RMS wrapped (rad)

Because the random-radius estimator has the same combinatorial structure
regardless of the metric or prior, the null reference mira_null(S) applies to
all three groups.
"""

import warnings
from multiprocessing import Pool

import numpy as np

from utils.tarp_eval import (
    _INITIAL_NOISE_STD_PROPORTION,
    _random_rotation_matrix,
    load_crystal_coords,
    load_sample_coords,
    load_protein_ca_coords,
)
from utils.group_eval import (
    get_rotatable_bonds,
    _kabsch_rotation,
    _geodesic_angle,
    _geodesic_distance_so3,
    extract_torsion_angles,
    _spyrmsd_mol,
    _get_sym_permutations,
    _apply_permutation,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Null reference
# ---------------------------------------------------------------------------

def mira_null(S: int) -> float:
    """Expected MIRA score under perfect calibration with S posterior samples.

    Derived from the Beta(2,1) null of the random-radius estimator.

    Args:
        S: number of posterior samples per complex.

    Returns:
        Null reference score ≈ 0.683 for S = 40.
    """
    return (2 / 3) * (S + 1) / S


# ---------------------------------------------------------------------------
# Per-group MIRA score functions (one complex at a time)
# ---------------------------------------------------------------------------

def _mira_score_translation(
    crystal_centroid: np.ndarray,
    sample_centroids: np.ndarray,
    ca_coords: np.ndarray,
    num_runs: int,
    rng,
) -> float:
    """MIRA score for the translation group (vectorised over runs).

    Draws random centers from DiffDock's translation prior at t = 1:
    N(Cα_COM, σ² I) with σ = std_ca × prop / 1.73.

    Args:
        crystal_centroid: (3,) crystal ligand centroid.
        sample_centroids: (S, 3) sample centroid positions.
        ca_coords: (N_res, 3) protein Cα coordinates.
        num_runs: number of Monte Carlo center draws.
        rng: numpy Generator.

    Returns:
        MIRA score (float), or nan if S < 2.
    """
    S = sample_centroids.shape[0]
    if S < 2:
        return float("nan")

    N = S - 1
    max_val = (N + 1) / (N + 2)

    protein_com = ca_coords.mean(axis=0)
    ca_centered = ca_coords - protein_com
    std_rec = np.sqrt(np.mean(np.sum(ca_centered ** 2, axis=1)))
    tr_std = std_rec * _INITIAL_NOISE_STD_PROPORTION / 1.73

    # Draw all centers in one shot: (num_runs, 3)
    centers = rng.normal(protein_com, tr_std, size=(num_runs, 3))

    # Distances: (num_runs,) and (num_runs, S)
    d_crystal = np.linalg.norm(centers - crystal_centroid[None, :], axis=1)
    d_samples = np.linalg.norm(centers[:, None, :] - sample_centroids[None, :, :], axis=2)

    # Random reference sample per run
    j_idx = rng.integers(0, S, size=num_runs)
    radii = d_samples[np.arange(num_runs), j_idx]

    # Masked distances: set selected sample to inf so it is excluded from count
    col_idx = np.arange(S)[None, :]
    masked = np.where(col_idx != j_idx[:, None], d_samples, np.inf)
    counts = (masked < radii[:, None]).sum(axis=1)

    k = (d_crystal <= radii).astype(float)
    prob_in = (counts + 1) / (N + 2)
    prob_out = (N - counts + 1) / (N + 2)
    calib = (prob_in * k + prob_out * (1 - k)) / max_val

    return float(np.nanmean(calib))


def _mira_score_rotation(
    sample_rotations: list,
    num_runs: int,
    rng,
) -> float:
    """MIRA score for the rotation group.

    Draws random centers Haar-uniformly from SO(3). Distance metric: geodesic
    angle. The crystal is the reference frame (identity rotation); d(center,
    crystal) = geodesic_angle(center).

    Args:
        sample_rotations: list of (3, 3) Kabsch rotation matrices, one per
            sample. None entries are skipped.
        num_runs: number of Monte Carlo center draws.
        rng: numpy Generator.

    Returns:
        MIRA score (float), or nan if fewer than 2 valid rotations.
    """
    valid = [R for R in sample_rotations if R is not None]
    S = len(valid)
    if S < 2:
        return float("nan")

    N = S - 1
    max_val = (N + 1) / (N + 2)

    total = 0.0
    valid_runs = 0

    for _ in range(num_runs):
        R_center = _random_rotation_matrix(rng)

        # Distance from center to crystal (identity)
        d_crystal = _geodesic_angle(R_center)

        # Distances from center to each sample rotation
        d_samples = np.array([_geodesic_distance_so3(R_center, R) for R in valid])

        finite = np.isfinite(d_samples)
        if finite.sum() < 2 or not np.isfinite(d_crystal):
            continue

        d_f = d_samples[finite]
        S_f = len(d_f)
        N_f = S_f - 1
        mv = (N_f + 1) / (N_f + 2)

        j = rng.integers(0, S_f)
        radius = d_f[j]
        mask = np.ones(S_f, dtype=bool)
        mask[j] = False
        counts = (d_f[mask] < radius).sum()

        k = float(d_crystal <= radius)
        prob = ((counts + 1) / (N_f + 2)) * k + ((N_f - counts + 1) / (N_f + 2)) * (1 - k)
        total += prob / mv
        valid_runs += 1

    return total / valid_runs if valid_runs > 0 else float("nan")


def _mira_score_torsion(
    crystal_torsions: np.ndarray,
    sample_torsions_list: list,
    num_runs: int,
    rng,
) -> float:
    """MIRA score for the torsion group (vectorised over runs).

    Draws random centers uniformly from T^k = [-π, π]^k. Distance metric:
    RMS wrapped angular difference. Returns nan for rigid molecules (k = 0).

    Args:
        crystal_torsions: (k,) crystal dihedral angles.
        sample_torsions_list: list of (k,) arrays, one per sample.
        num_runs: number of Monte Carlo center draws.
        rng: numpy Generator.

    Returns:
        MIRA score (float), or nan if k == 0 or too few valid samples.
    """
    k = len(crystal_torsions)
    if k == 0:
        return float("nan")

    valid = [sa for sa in sample_torsions_list if np.all(np.isfinite(sa)) and len(sa) == k]
    S = len(valid)
    if S < 2:
        return float("nan")

    N = S - 1
    max_val = (N + 1) / (N + 2)

    # Stack: (S, k)
    angles = np.stack(valid, axis=0)

    # Draw all centers: (num_runs, k)
    centers = rng.uniform(-np.pi, np.pi, size=(num_runs, k))

    # Wrapped RMS distance to crystal: (num_runs,)
    diff_c = (centers - crystal_torsions[None, :] + np.pi) % (2 * np.pi) - np.pi
    d_crystal = np.sqrt(np.mean(diff_c ** 2, axis=1))

    # Wrapped RMS distance to each sample: (num_runs, S)
    diff_s = (centers[:, None, :] - angles[None, :, :] + np.pi) % (2 * np.pi) - np.pi
    d_samples = np.sqrt(np.mean(diff_s ** 2, axis=2))

    # Random reference sample per run
    j_idx = rng.integers(0, S, size=num_runs)
    radii = d_samples[np.arange(num_runs), j_idx]

    col_idx = np.arange(S)[None, :]
    masked = np.where(col_idx != j_idx[:, None], d_samples, np.inf)
    counts = (masked < radii[:, None]).sum(axis=1)

    k_ind = (d_crystal <= radii).astype(float)
    prob_in = (counts + 1) / (N + 2)
    prob_out = (N - counts + 1) / (N + 2)
    calib = (prob_in * k_ind + prob_out * (1 - k_ind)) / max_val

    return float(np.nanmean(calib))


# ---------------------------------------------------------------------------
# Multiprocessing worker and batch runner
# ---------------------------------------------------------------------------

def _group_mira_worker(args):
    """Process a single complex for per-group MIRA evaluation.

    Loads crystal and sample coordinates, computes symmetry-corrected rotation
    matrices and torsion angles (same preprocessing as _group_tarp_worker),
    then runs the three per-group MIRA score functions.

    Args:
        args: tuple of (pdb_id, results_index, data_dir, num_runs, seed,
            max_samples).

    Returns:
        (pdb_id, result_dict_or_None, error_str_or_None)
        result_dict keys: 'translation', 'rotation', 'torsion' (float each).
    """
    pdb_id, results_index, data_dir, num_runs, seed, max_samples = args
    warnings.filterwarnings("ignore")

    try:
        crystal_mol, all_crystal_coords = load_crystal_coords(pdb_id, data_dir)
        crystal_coords = all_crystal_coords[0]
        sample_coords = load_sample_coords(pdb_id, results_index)
        ca_coords = load_protein_ca_coords(pdb_id, data_dir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return pdb_id, None, f"load error: {exc}"

    if max_samples is not None:
        sample_coords = sample_coords[:max_samples]
    if len(sample_coords) < 2:
        return pdb_id, None, "too few samples"

    rng = np.random.default_rng(seed)
    rot_bonds = get_rotatable_bonds(crystal_mol)

    # Translation: centroid positions
    crystal_centroid = crystal_coords.mean(axis=0)
    sample_centroids = np.array([sc.mean(axis=0) for sc in sample_coords])

    # Symmetry corrections — Kabsch rotation and torsion angles
    crystal_c = crystal_coords - crystal_centroid
    crystal_torsions = extract_torsion_angles(crystal_mol, crystal_coords, rot_bonds)

    spy_mol = _spyrmsd_mol(crystal_mol)
    sample_centered = [sc - sc.mean(axis=0) for sc in sample_coords]
    perms = _get_sym_permutations(crystal_c, sample_centered, spy_mol)

    sample_rotations = []
    sample_angles = []
    for sc, sc_c, perm in zip(sample_coords, sample_centered, perms):
        idx1, idx2 = perm
        # Rotation matrix via Kabsch
        try:
            sc_c_reordered = _apply_permutation(sc_c, idx1, idx2)
            R = _kabsch_rotation(crystal_c, sc_c_reordered)
            sample_rotations.append(R)
        except Exception:
            sample_rotations.append(None)
        # Torsion angles
        if len(rot_bonds) > 0:
            try:
                sc_reordered = _apply_permutation(sc, idx1, idx2)
                angles = extract_torsion_angles(crystal_mol, sc_reordered, rot_bonds)
            except Exception:
                angles = extract_torsion_angles(crystal_mol, sc, rot_bonds)
        else:
            angles = np.array([], dtype=float)
        sample_angles.append(angles)

    try:
        score_tr = _mira_score_translation(
            crystal_centroid, sample_centroids, ca_coords, num_runs, rng)
        score_rot = _mira_score_rotation(sample_rotations, num_runs, rng)
        score_tor = _mira_score_torsion(crystal_torsions, sample_angles, num_runs, rng)
    except Exception as exc:
        return pdb_id, None, f"compute error: {exc}"

    return pdb_id, {
        "translation": score_tr,
        "rotation":    score_rot,
        "torsion":     score_tor,
    }, None


def run_group_mira_eval(
    complex_names,
    results_index: dict,
    data_dir: str,
    num_runs: int = 100,
    seed: int = 42,
    verbose: bool = True,
    n_workers: int = 1,
    max_samples: int = None,
) -> dict:
    """Compute per-group MIRA scores for a full test set.

    Evaluates MIRA independently for translation, rotation, and torsion.
    Returns per-complex scores and the corresponding PDB IDs for each group
    (some complexes may fail in one group but not another).

    Args:
        complex_names: iterable of PDB ID strings.
        results_index: dict mapping pdb_id → Path from build_results_index().
        data_dir: root data directory (parent of per-complex subdirectories).
        num_runs: Monte Carlo center draws per complex (100 gives stable scores).
        seed: master random seed; per-complex seeds are derived via SeedSequence.
        verbose: print progress every 20 complexes.
        n_workers: parallel worker processes (1 = serial).
        max_samples: if set, cap the number of samples used per complex.

    Returns:
        dict with keys 'translation', 'rotation', 'torsion', each mapping to
        a (names, scores) tuple of numpy arrays of length n_valid for that group.
        n_valid may differ across groups (e.g. torsion excludes rigid ligands).
    """
    complex_names = list(complex_names)
    n = len(complex_names)
    child_seeds = np.random.SeedSequence(seed).spawn(n)

    work = [
        (pdb_id, results_index, data_dir, num_runs, child_seeds[i], max_samples)
        for i, pdb_id in enumerate(complex_names)
    ]

    group_names  = {g: [] for g in ("translation", "rotation", "torsion")}
    group_scores = {g: [] for g in ("translation", "rotation", "torsion")}
    skipped = 0
    n_done = 0

    def _handle(result):
        nonlocal skipped, n_done
        pdb_id, res, err = result
        if verbose and n_done % 20 == 0:
            print(f"  [{n_done}/{n}] {pdb_id} ...", flush=True)
        n_done += 1
        if err is not None:
            if verbose:
                print(f"    Skipping {pdb_id}: {err}", flush=True)
            skipped += 1
            return
        for g in ("translation", "rotation", "torsion"):
            score = res[g]
            if np.isfinite(score):
                group_names[g].append(pdb_id)
                group_scores[g].append(score)

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for result in pool.imap(_group_mira_worker, work):
                _handle(result)
    else:
        for result in map(_group_mira_worker, work):
            _handle(result)

    if verbose:
        print(f"\nDone. {n_done} complexes processed, {skipped} skipped.")
        s_typ = max_samples if max_samples else 40
        null = mira_null(s_typ)
        for g in ("translation", "rotation", "torsion"):
            if group_scores[g]:
                m = np.mean(group_scores[g])
                print(f"  {g:12s}: n={len(group_scores[g]):4d}, "
                      f"mean MIRA = {m:.4f}  (null = {null:.4f})")

    return {
        g: (np.array(group_names[g]), np.array(group_scores[g], dtype=float))
        for g in ("translation", "rotation", "torsion")
    }
