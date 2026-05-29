"""
Per-group (translation / rotation / torsion) TARP evaluation for DiffDock.

Computes TARP coverage fractions separately for each of DiffDock's three
diffusion groups, using manifold-appropriate distance metrics:

  Translation (R³)
    Euclidean L2 distance between ligand centroids.  Centroid = unweighted
    mean of all heavy-atom positions, matching DiffDock's modify_conformer.

  Rotation (SO(3))
    Geodesic angle of the Kabsch best-fit rotation from the crystal pose
    (centroid-centred) to each predicted pose (centroid-centred).  The
    crystal is the reference frame so its "rotation" is always the identity.
    Reference points are uniform random SO(3) rotations (Haar measure), same
    as DiffDock's prior at t=1.  Note: for ligands with many flexible bonds
    this estimate includes some torsional signal; stratify by n_rot_bonds
    when interpreting.

  Torsion (T^k, k = number of rotatable bonds)
    RMS wrapped angular difference between crystal and predicted dihedral
    angles across all k rotatable bonds.  Bonds identified via the same
    graph-connectivity criterion as DiffDock (conformer_matching.py).
    Torsion angles are internal coordinates so they are unaffected by rigid-
    body transformations.  Reference points are k-dimensional uniform draws
    from [−π, π].

TARP reference distributions (match DiffDock's randomize_position at t=1):
  Translation : N(Cα_COM, σ_tr² I)  with  σ_tr = std_ca * prop / 1.73.
  Rotation    : Uniform on SO(3) via random unit quaternion.
  Torsion     : Uniform on [−π, π]^k (one draw per rotatable bond).
"""

import copy
import warnings
from multiprocessing import Pool

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from utils.tarp_eval import (
    _INITIAL_NOISE_STD_PROPORTION,
    _random_rotation_matrix,
    load_crystal_coords,
    load_sample_coords,
    load_protein_ca_coords,
    prepare_reference_template,
)

warnings.filterwarnings("ignore")

_trapz = getattr(np, "trapezoid", np.trapz)


# ---------------------------------------------------------------------------
# Rotatable-bond identification (mirrors DiffDock's get_torsion_angles)
# ---------------------------------------------------------------------------

def get_rotatable_bonds(mol):
    """Identify rotatable bonds using DiffDock's graph-connectivity criterion.

    A bond is rotatable if removing it splits the molecular graph into two
    components each with ≥ 2 atoms (i.e. non-ring, non-terminal). This
    matches datasets/conformer_matching.py:get_torsion_angles and
    utils/torsion.py:get_transformation_mask.

    Args:
        mol: RDKit Mol (heavy atoms, no Hs).

    Returns:
        List of (n0, a, b, n1) atom-index 4-tuples. Each tuple defines a
        dihedral angle via rdMolTransforms.GetDihedralRad(conf, n0, a, b, n1).
        Empty list for ring-only or fully rigid molecules.
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    torsions = []
    for a, b in list(G.edges()):
        G2 = G.copy()
        G2.remove_edge(a, b)
        if nx.is_connected(G2):
            continue
        smaller = min(nx.connected_components(G2), key=len)
        if len(smaller) < 2:
            continue
        n0 = next(iter(G2.neighbors(a)))
        n1 = next(iter(G2.neighbors(b)))
        torsions.append((n0, a, b, n1))
    return torsions


# ---------------------------------------------------------------------------
# SO(3) utilities
# ---------------------------------------------------------------------------

def _kabsch_rotation(source_centered, target_centered):
    """Kabsch best-fit rotation mapping source to target.

    Args:
        source_centered: (N, 3) zero-centroid point cloud.
        target_centered: (N, 3) zero-centroid point cloud.

    Returns:
        (3, 3) rotation matrix R s.t. target ≈ source @ R.T (least squares).
        Guaranteed det(R) = +1 (proper rotation, not reflection).
    """
    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R


def _geodesic_angle(R):
    """Geodesic distance on SO(3) from the identity to R.

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        Rotation angle in radians ∈ [0, π].
    """
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def _geodesic_distance_so3(R1, R2):
    """Geodesic distance on SO(3) between two rotation matrices.

    Args:
        R1, R2: (3, 3) rotation matrices.

    Returns:
        Angle in radians ∈ [0, π].
    """
    return _geodesic_angle(R1.T @ R2)


# ---------------------------------------------------------------------------
# Torsion angle utilities
# ---------------------------------------------------------------------------

def extract_torsion_angles(mol, coords, rot_bonds):
    """Read dihedral angles for each rotatable bond from a set of coordinates.

    Copies the molecular graph from mol, overwrites the conformer with coords,
    then calls rdMolTransforms.GetDihedralRad for each bond in rot_bonds.
    Torsion angles are internal coordinates so the result is independent of
    the rigid-body transformation applied to the pose.

    Args:
        mol: RDKit Mol (heavy atoms, no Hs) — supplies the atom graph.
        coords: (N_atoms, 3) numpy array of atom positions.
        rot_bonds: list of (n0, a, b, n1) tuples from get_rotatable_bonds().

    Returns:
        numpy array of shape (len(rot_bonds),) with angles in [−π, π].
        Empty array (shape (0,)) when rot_bonds is empty.
        Individual entries are NaN if GetDihedralRad raises for that bond.
    """
    if not rot_bonds:
        return np.array([], dtype=float)

    mol_copy = copy.deepcopy(mol)
    conf = mol_copy.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))

    angles = []
    for (n0, a, b, n1) in rot_bonds:
        try:
            angle = rdMolTransforms.GetDihedralRad(conf, int(n0), int(a), int(b), int(n1))
        except Exception:
            angle = float("nan")
        angles.append(angle)
    return np.array(angles, dtype=float)


def _wrapped_diff(a, b):
    """Element-wise wrapped circular difference (a − b), result in (−π, π].

    Args:
        a, b: numpy arrays of angles in radians.

    Returns:
        numpy array, same shape.
    """
    return ((a - b + np.pi) % (2.0 * np.pi)) - np.pi


def rms_torsion_distance(angles1, angles2):
    """RMS wrapped angular distance across rotatable bonds.

    Args:
        angles1, angles2: numpy arrays of shape (n_bonds,).

    Returns:
        float ≥ 0 in radians. NaN if arrays are empty, shapes differ, or
        no finite pair of values remains after masking.
    """
    if len(angles1) == 0 or len(angles2) == 0 or angles1.shape != angles2.shape:
        return float("nan")
    finite = np.isfinite(angles1) & np.isfinite(angles2)
    if not finite.any():
        return float("nan")
    d = _wrapped_diff(angles1[finite], angles2[finite])
    return float(np.sqrt(np.mean(d ** 2)))


# ---------------------------------------------------------------------------
# Per-group distance extraction
# ---------------------------------------------------------------------------

def compute_group_distances(crystal_mol, crystal_coords, sample_coords_list, rot_bonds):
    """Extract translation, rotation, and torsion distances for all samples.

    Args:
        crystal_mol: RDKit Mol (heavy atoms, no Hs).
        crystal_coords: (N_atoms, 3) numpy array — crystal pose.
        sample_coords_list: list of (N_atoms, 3) arrays — DiffDock samples.
        rot_bonds: list of (n0, a, b, n1) from get_rotatable_bonds().

    Returns:
        dict with keys:
            'translation'     : (S,) L2 centroid distance to crystal (Å).
            'rotation'        : (S,) geodesic angle of Kabsch rotation from
                                crystal-centred frame to sample-centred frame
                                (rad, ∈ [0, π]).
            'torsion_rms'     : (S,) RMS wrapped torsion distance across all
                                rotatable bonds (rad). NaN if n_bonds=0.
            'torsion_per_bond': (S, n_bonds) wrapped per-bond differences (rad).
                                Shape (S, 0) if n_bonds=0.
            'n_rot_bonds'     : int, number of rotatable bonds for this complex.
    """
    S = len(sample_coords_list)
    n_bonds = len(rot_bonds)

    crystal_c = crystal_coords.mean(axis=0)
    crystal_centered = crystal_coords - crystal_c
    crystal_torsions = extract_torsion_angles(crystal_mol, crystal_coords, rot_bonds)

    tr_dists = np.empty(S)
    rot_angles = np.empty(S)
    tor_rms = np.empty(S)
    tor_per_bond = np.full((S, n_bonds), np.nan) if n_bonds > 0 else np.empty((S, 0))

    for i, sc in enumerate(sample_coords_list):
        # Translation
        tr_dists[i] = np.linalg.norm(sc.mean(axis=0) - crystal_c)

        # Rotation: Kabsch angle relative to crystal
        sc_centered = sc - sc.mean(axis=0)
        try:
            R = _kabsch_rotation(crystal_centered, sc_centered)
            rot_angles[i] = _geodesic_angle(R)
        except Exception:
            rot_angles[i] = float("nan")

        # Torsion: intrinsic dihedral angles
        sample_torsions = extract_torsion_angles(crystal_mol, sc, rot_bonds)
        tor_rms[i] = rms_torsion_distance(crystal_torsions, sample_torsions)
        if n_bonds > 0:
            diffs = _wrapped_diff(sample_torsions, crystal_torsions)
            tor_per_bond[i] = diffs

    return {
        "translation": tr_dists,
        "rotation": rot_angles,
        "torsion_rms": tor_rms,
        "torsion_per_bond": tor_per_bond,
        "n_rot_bonds": n_bonds,
    }


# ---------------------------------------------------------------------------
# Per-group TARP fraction computation
# ---------------------------------------------------------------------------

def _translation_reference_centroid(ca_coords, rng):
    """Draw a random centroid from DiffDock's translation prior at t=1.

    Mirrors the initial_noise_std_proportion ≥ 0 branch of
    utils/sampling.py:randomize_position.

    Args:
        ca_coords: (N_res, 3) Cα coordinates.
        rng: numpy Generator.

    Returns:
        (3,) centroid position.
    """
    protein_com = ca_coords.mean(axis=0)
    ca_centered = ca_coords - protein_com
    std_rec = np.sqrt(np.mean(np.sum(ca_centered ** 2, axis=1)))
    tr_std = std_rec * _INITIAL_NOISE_STD_PROPORTION / 1.73
    return rng.normal(loc=protein_com, scale=tr_std, size=3)


def compute_tarp_fractions_translation(crystal_coords, sample_coords_list,
                                        ca_coords, K, rng):
    """TARP coverage fractions for the translation group.

    Reference: N(Cα_COM, σ_tr² I).
    Distance: L2 between centroids.

    Args:
        crystal_coords: (N_atoms, 3) crystal pose.
        sample_coords_list: list of (N_atoms, 3) DiffDock samples.
        ca_coords: (N_res, 3) protein Cα coordinates.
        K: number of random reference draws.
        rng: numpy Generator.

    Returns:
        numpy array of shape (≤K,) with fractions in [0, 1].
    """
    crystal_c = crystal_coords.mean(axis=0)
    sample_cs = np.array([sc.mean(axis=0) for sc in sample_coords_list])  # (S, 3)

    fracs = []
    for _ in range(K):
        ref_c = _translation_reference_centroid(ca_coords, rng)
        d_crystal = float(np.linalg.norm(crystal_c - ref_c))
        if not np.isfinite(d_crystal):
            continue
        d_samples = np.linalg.norm(sample_cs - ref_c, axis=1)
        finite = np.isfinite(d_samples)
        if not finite.any():
            continue
        fracs.append(float((d_samples[finite] < d_crystal).mean()))
    return np.array(fracs)


def compute_tarp_fractions_rotation(crystal_coords, sample_coords_list, K, rng):
    """TARP coverage fractions for the rotation group.

    Reference: uniform SO(3) rotation (Haar measure).
    Distance: geodesic angle between the Kabsch rotation (crystal-centred →
    sample-centred) and the reference rotation.  The crystal is treated as
    the identity (rotation = 0 from itself), so d_crystal = geodesic angle
    of the reference rotation from the identity.

    Args:
        crystal_coords: (N_atoms, 3) crystal pose.
        sample_coords_list: list of (N_atoms, 3) DiffDock samples.
        K: number of random reference draws.
        rng: numpy Generator.

    Returns:
        numpy array of shape (≤K,) with fractions in [0, 1].
        Empty if no finite rotation angles could be computed.
    """
    crystal_centered = crystal_coords - crystal_coords.mean(axis=0)

    # Kabsch rotation of each sample relative to the crystal frame
    sample_rotations = []
    for sc in sample_coords_list:
        sc_centered = sc - sc.mean(axis=0)
        try:
            R_sample = _kabsch_rotation(crystal_centered, sc_centered)
            sample_rotations.append(R_sample)
        except Exception:
            sample_rotations.append(None)

    fracs = []
    for _ in range(K):
        R_ref = _random_rotation_matrix(rng)          # uniform SO(3)
        # d_crystal = geodesic(Identity, R_ref) = angle of R_ref itself
        d_crystal = _geodesic_angle(R_ref)
        if not np.isfinite(d_crystal):
            continue
        d_samples = []
        for R_sample in sample_rotations:
            if R_sample is None:
                d_samples.append(float("nan"))
            else:
                d_samples.append(_geodesic_distance_so3(R_sample, R_ref))
        d_samples = np.array(d_samples)
        finite = np.isfinite(d_samples)
        if not finite.any():
            continue
        fracs.append(float((d_samples[finite] < d_crystal).mean()))
    return np.array(fracs)


def compute_tarp_fractions_torsion(crystal_mol, crystal_coords, sample_coords_list,
                                    rot_bonds, K, rng):
    """TARP coverage fractions for the torsion group.

    Reference: uniform on [−π, π]^k (one independent draw per rotatable bond).
    Distance: RMS wrapped angular difference across all k bonds.
    Complexes with 0 rotatable bonds return an empty array.

    Args:
        crystal_mol: RDKit Mol (heavy atoms, no Hs).
        crystal_coords: (N_atoms, 3) crystal pose.
        sample_coords_list: list of (N_atoms, 3) DiffDock samples.
        rot_bonds: list of (n0, a, b, n1) from get_rotatable_bonds().
        K: number of random reference draws.
        rng: numpy Generator.

    Returns:
        numpy array of shape (≤K,) with fractions in [0, 1].
        Empty array if n_bonds == 0.
    """
    n_bonds = len(rot_bonds)
    if n_bonds == 0:
        return np.array([])

    crystal_angles = extract_torsion_angles(crystal_mol, crystal_coords, rot_bonds)
    sample_angles_list = [
        extract_torsion_angles(crystal_mol, sc, rot_bonds)
        for sc in sample_coords_list
    ]

    fracs = []
    for _ in range(K):
        ref_angles = rng.uniform(-np.pi, np.pi, size=n_bonds)
        d_crystal = rms_torsion_distance(crystal_angles, ref_angles)
        if not np.isfinite(d_crystal):
            continue
        d_samples = np.array([
            rms_torsion_distance(sa, ref_angles) for sa in sample_angles_list
        ])
        finite = np.isfinite(d_samples)
        if not finite.any():
            continue
        fracs.append(float((d_samples[finite] < d_crystal).mean()))
    return np.array(fracs)


# ---------------------------------------------------------------------------
# Multiprocessing worker and batch runner
# ---------------------------------------------------------------------------

def _group_tarp_worker(args):
    """Process a single complex for per-group TARP evaluation.

    Top-level function required for multiprocessing pickle compatibility.

    Args:
        args: tuple of
            (pdb_id, results_index, data_dir, K, seed, max_samples)

    Returns:
        (pdb_id, result_dict_or_None, error_str_or_None)
        result_dict has keys 'translation', 'rotation', 'torsion',
        'n_rot_bonds', and values that are numpy arrays of shape (≤K,).
    """
    pdb_id, results_index, data_dir, K, seed, max_samples = args
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
    if len(sample_coords) == 0:
        return pdb_id, None, "no valid samples"

    rng = np.random.default_rng(seed)
    rot_bonds = get_rotatable_bonds(crystal_mol)
    n_bonds = len(rot_bonds)

    try:
        fracs_tr = compute_tarp_fractions_translation(
            crystal_coords, sample_coords, ca_coords, K, rng
        )
        fracs_rot = compute_tarp_fractions_rotation(
            crystal_coords, sample_coords, K, rng
        )
        fracs_tor = compute_tarp_fractions_torsion(
            crystal_mol, crystal_coords, sample_coords, rot_bonds, K, rng
        )
    except Exception as exc:
        return pdb_id, None, f"compute error: {exc}"

    result = {
        "translation": fracs_tr,
        "rotation": fracs_rot,
        "torsion": fracs_tor,
        "n_rot_bonds": n_bonds,
    }
    return pdb_id, result, None


def run_group_tarp_eval(
    complex_names,
    results_index,
    data_dir,
    K=100,
    seed=42,
    verbose=True,
    n_workers=1,
    max_samples=None,
):
    """Run per-group TARP evaluation over all complexes.

    Args:
        complex_names: iterable of PDB ID strings.
        results_index: dict mapping pdb_id → Path from build_results_index().
        data_dir: root directory (parent of per-complex subdirectories).
        K: number of random reference draws per complex per group.
        seed: master random seed; per-complex seeds are derived via
            SeedSequence so results are stable regardless of n_workers.
        verbose: print progress every 20 complexes.
        n_workers: parallel worker processes (1 = serial).
        max_samples: if set, truncate each complex to this many samples.

    Returns:
        dict with keys 'translation', 'rotation', 'torsion', each mapping
        to a numpy array of shape (n_valid_complexes, K) with coverage
        fractions. A separate 'names' key holds the (n_valid_complexes,)
        array of PDB IDs in the same order. 'n_rot_bonds' is a (n_valid,)
        int array with the rotatable-bond count per complex.
    """
    complex_names = list(complex_names)
    n = len(complex_names)
    child_seeds = np.random.SeedSequence(seed).spawn(n)

    work = [
        (pdb_id, results_index, data_dir, K, child_seeds[i], max_samples)
        for i, pdb_id in enumerate(complex_names)
    ]

    rows = {"translation": [], "rotation": [], "torsion": []}
    names_out = []
    n_rot_bonds_out = []
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
        for grp in ("translation", "rotation", "torsion"):
            fracs = res[grp]
            # Pad or trim to exactly K entries (use NaN for empties from torsion)
            if len(fracs) == 0:
                rows[grp].append(np.full(K, np.nan))
            else:
                row = np.full(K, np.nan)
                row[:min(len(fracs), K)] = fracs[:K]
                rows[grp].append(row)
        names_out.append(pdb_id)
        n_rot_bonds_out.append(res["n_rot_bonds"])

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for result in pool.imap(_group_tarp_worker, work):
                _handle(result)
    else:
        for result in map(_group_tarp_worker, work):
            _handle(result)

    if verbose:
        print(
            f"Done. {len(names_out)} complexes processed, {skipped} skipped.",
            flush=True,
        )

    out = {
        "names": np.array(names_out),
        "n_rot_bonds": np.array(n_rot_bonds_out, dtype=int),
    }
    for grp in ("translation", "rotation", "torsion"):
        if rows[grp]:
            out[grp] = np.vstack(rows[grp])
        else:
            out[grp] = np.empty((0, K))
    return out


# ---------------------------------------------------------------------------
# Raw group-distance extraction (for MIRA and other analyses)
# ---------------------------------------------------------------------------

def _group_distances_worker(args):
    """Compute per-sample group distances for a single complex.

    Args:
        args: tuple of (pdb_id, results_index, data_dir, max_samples)

    Returns:
        (pdb_id, distances_dict_or_None, error_str_or_None)
        distances_dict has keys from compute_group_distances plus 'n_samples'.
    """
    pdb_id, results_index, data_dir, max_samples = args
    warnings.filterwarnings("ignore")

    try:
        crystal_mol, all_crystal_coords = load_crystal_coords(pdb_id, data_dir)
        crystal_coords = all_crystal_coords[0]
        sample_coords = load_sample_coords(pdb_id, results_index)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return pdb_id, None, f"load error: {exc}"

    if max_samples is not None:
        sample_coords = sample_coords[:max_samples]
    if len(sample_coords) == 0:
        return pdb_id, None, "no valid samples"

    rot_bonds = get_rotatable_bonds(crystal_mol)
    try:
        dists = compute_group_distances(
            crystal_mol, crystal_coords, sample_coords, rot_bonds
        )
    except Exception as exc:
        return pdb_id, None, f"compute error: {exc}"

    dists["n_samples"] = len(sample_coords)
    return pdb_id, dists, None


def run_group_distances(
    complex_names,
    results_index,
    data_dir,
    verbose=True,
    n_workers=1,
    max_samples=None,
):
    """Compute per-sample per-group distances for all complexes.

    Args:
        complex_names: iterable of PDB ID strings.
        results_index: dict from build_results_index().
        data_dir: root directory.
        verbose: print progress every 20 complexes.
        n_workers: parallel workers.
        max_samples: if set, cap samples per complex.

    Returns:
        dict with keys:
            'names'          : (n_valid,) PDB ID strings.
            'n_rot_bonds'    : (n_valid,) int.
            'translation'    : (n_valid, S) in Å — NaN for missing samples.
            'rotation'       : (n_valid, S) in rad.
            'torsion_rms'    : (n_valid, S) in rad — NaN if n_bonds=0.
        where S = max samples across valid complexes.
    """
    complex_names = list(complex_names)
    n = len(complex_names)
    work = [
        (pdb_id, results_index, data_dir, max_samples)
        for pdb_id in complex_names
    ]

    all_results = []
    skipped = 0
    n_done = 0

    def _handle(result):
        nonlocal skipped, n_done
        pdb_id, dists, err = result
        if verbose and n_done % 20 == 0:
            print(f"  [{n_done}/{n}] {pdb_id} ...", flush=True)
        n_done += 1
        if err is not None:
            if verbose:
                print(f"    Skipping {pdb_id}: {err}", flush=True)
            skipped += 1
        else:
            all_results.append((pdb_id, dists))

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for result in pool.imap(_group_distances_worker, work):
                _handle(result)
    else:
        for result in map(_group_distances_worker, work):
            _handle(result)

    if verbose:
        print(
            f"Done. {len(all_results)} complexes, {skipped} skipped.",
            flush=True,
        )

    if not all_results:
        return {"names": np.array([]), "n_rot_bonds": np.array([], dtype=int),
                "translation": np.empty((0, 0)), "rotation": np.empty((0, 0)),
                "torsion_rms": np.empty((0, 0))}

    S_max = max(r[1]["n_samples"] for r in all_results)
    n_valid = len(all_results)

    out = {
        "names": np.array([r[0] for r in all_results]),
        "n_rot_bonds": np.array([r[1]["n_rot_bonds"] for r in all_results], dtype=int),
        "translation": np.full((n_valid, S_max), np.nan),
        "rotation": np.full((n_valid, S_max), np.nan),
        "torsion_rms": np.full((n_valid, S_max), np.nan),
    }
    for i, (_, dists) in enumerate(all_results):
        S = dists["n_samples"]
        out["translation"][i, :S] = dists["translation"]
        out["rotation"][i, :S] = dists["rotation"]
        out["torsion_rms"][i, :S] = dists["torsion_rms"]

    return out
