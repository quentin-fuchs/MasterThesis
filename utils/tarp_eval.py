"""
TARP (Tests of Accuracy with Random Points) evaluation for DiffDock.

Implements the TARP coverage test from Lemos & Coogan et al. 2023
(https://arxiv.org/abs/2302.03026) applied to DiffDock's pose distributions.

For each test complex, TARP checks whether DiffDock's 40 samples are
calibrated by asking: given a random reference pose θ_r, what fraction of
DiffDock samples fall closer to θ_r than the crystal pose does?  If this
fraction, averaged over complexes and references, matches α for all α, the
posterior is perfectly calibrated.

Two modes are supported:
  - centroid: 3D centroid of ligand (tests binding-site localisation).
  - rmsd:     full heavy-atom RMSD using spyrmsd (tests full pose accuracy).

Typical usage
-------------
>>> from utils.tarp_eval import build_results_index, run_tarp_eval, plot_ecp
>>> index = build_results_index("results/testset_eval_full")
>>> names = np.load("results/testset_eval_merged/complex_names.npy", allow_pickle=True)
>>> f_vals = run_tarp_eval(names, index, "data/PDBBind_processed", K=100, mode="rmsd")
>>> ecp, alpha = ecp_from_fractions(f_vals)
>>> plot_ecp(ecp, alpha)
"""

import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from spyrmsd import rmsd as spyrmsd_rmsd, molecule as spyrmsd_molecule
import prody as pr

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_results_index(eval_full_dir):
    """Scan all chunk_* subdirectories and return a mapping from PDB ID to the
    directory that contains its rank*.sdf prediction files.

    Args:
        eval_full_dir: Path to the top-level results directory that contains
            chunk_0/, chunk_1/, ... subdirectories (e.g.
            "results/testset_eval_full").

    Returns:
        dict mapping pdb_id (str) to Path of the complex subdirectory.
    """
    index = {}
    for chunk_dir in sorted(Path(eval_full_dir).glob("chunk_*")):
        for complex_dir in chunk_dir.iterdir():
            if complex_dir.is_dir():
                index[complex_dir.name] = complex_dir
    return index


def load_crystal_coords(pdb_id, data_dir):
    """Load the crystal-pose heavy-atom coordinates and the RDKit mol.

    Args:
        pdb_id: PDB identifier string (e.g. "5ze6").
        data_dir: Root data directory containing PDBBind_processed/

    Returns:
        (mol, coords): RDKit Mol (heavy atoms, no Hs) and numpy array of
            shape (N_atoms, 3).
    """
    sdf_path = os.path.join(data_dir, pdb_id, f"{pdb_id}_ligand.sdf")
    mol2_path = os.path.join(data_dir, pdb_id, f"{pdb_id}_ligand.mol2")
    for path in [sdf_path, mol2_path]:
        if os.path.exists(path):
            supplier = Chem.SDMolSupplier(path, removeHs=True) if path.endswith(".sdf") else None
            if supplier is not None:
                mol = supplier[0]
            else:
                mol = Chem.MolFromMol2File(path, removeHs=True)
            if mol is not None:
                return mol, mol.GetConformer().GetPositions()
    raise FileNotFoundError(f"No ligand file for {pdb_id} in {data_dir}")


def load_sample_coords(pdb_id, results_index):
    """Load the 40 DiffDock predicted heavy-atom coordinate sets (rank*.sdf).

    The rank*.sdf files are sorted by confidence (rank1 = highest confidence).
    All 40 files are loaded regardless of confidence order.

    Args:
        pdb_id: PDB identifier string.
        results_index: Dict from build_results_index().

    Returns:
        List of numpy arrays, each of shape (N_atoms, 3), one per sample.
        Length is typically 40 but may be less if some files are missing.
    """
    complex_dir = results_index[pdb_id]
    coords_list = []
    rank_files = sorted(
        [f for f in complex_dir.iterdir()
         if f.name.startswith("rank") and f.name.endswith(".sdf")
         and "_confidence" not in f.name],
        key=lambda f: int(f.stem.replace("rank", ""))
    )
    for sdf_file in rank_files:
        mol = Chem.SDMolSupplier(str(sdf_file), removeHs=True)[0]
        if mol is not None:
            coords_list.append(mol.GetConformer().GetPositions())
    return coords_list


def load_protein_ca_coords(pdb_id, data_dir):
    """Load the Cα coordinates of the processed protein structure.

    Args:
        pdb_id: PDB identifier string.
        data_dir: Root data directory containing PDBBind_processed/

    Returns:
        numpy array of shape (N_residues, 3).
    """
    pdb_path = os.path.join(data_dir, pdb_id, f"{pdb_id}_protein_processed.pdb")
    prot = pr.parsePDB(pdb_path)
    return prot.ca.getCoords()


# ---------------------------------------------------------------------------
# Reference pose generation
# ---------------------------------------------------------------------------

def _random_rotation_matrix(rng):
    """Sample a uniformly distributed rotation matrix from SO(3) via a
    random unit quaternion (Shoemake 1992).

    Args:
        rng: numpy Generator (from np.random.default_rng).

    Returns:
        (3, 3) numpy rotation matrix.
    """
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ])


def _get_rotatable_bonds(mol):
    """Return a list of (atom_i, atom_j) index pairs for all rotatable bonds.

    Args:
        mol: RDKit Mol (heavy atoms).

    Returns:
        List of 2-tuples of atom indices.
    """
    rot_bond_pattern = Chem.MolFromSmarts(
        "[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]"
    )
    matches = mol.GetSubstructMatches(rot_bond_pattern)
    # deduplicate (a,b) vs (b,a)
    seen = set()
    bonds = []
    for a, b in matches:
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            bonds.append((a, b))
    return bonds


def generate_reference_coords(template_mol, ca_coords, rng, box_buffer=5.0):
    """Generate a random reference ligand pose within the protein bounding box.

    The reference is constructed from the template molecule's topology:
    1. Copy the template conformer (keeps the same bond lengths/angles).
    2. Randomise all rotatable torsion angles uniformly in [0, 360°].
    3. Centre the coordinates at the origin.
    4. Apply a uniformly random SO(3) rotation.
    5. Translate the centroid to a random point within the protein Cα
       bounding box (extended by box_buffer Å on each side).

    Args:
        template_mol: RDKit Mol (heavy atoms, with a conformer).
        ca_coords: numpy array (N_res, 3) of protein Cα coordinates.
        rng: numpy Generator.
        box_buffer: Extra Å to add around the protein bounding box.

    Returns:
        numpy array of shape (N_atoms, 3).
    """
    # Work on an editable copy; reuse the existing conformer (no re-embedding)
    mol = Chem.RWMol(template_mol)
    conf = mol.GetConformer()
    coords = conf.GetPositions().copy()

    # 1. Randomise torsion angles
    rot_bonds = _get_rotatable_bonds(mol)
    for a, b in rot_bonds:
        # find a neighbor of a (not b) and a neighbor of b (not a)
        a_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() != b]
        b_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != a]
        if a_nbrs and b_nbrs:
            angle = rng.uniform(0, 360)
            rdMolTransforms_set_dihedral(conf, a_nbrs[0], a, b, b_nbrs[0], angle)
    coords = conf.GetPositions().copy()

    # 2. Centre
    coords -= coords.mean(axis=0)

    # 3. Random SO(3) rotation
    R = _random_rotation_matrix(rng)
    coords = coords @ R.T

    # 4. Random translation within protein bounding box
    box_min = ca_coords.min(axis=0) - box_buffer
    box_max = ca_coords.max(axis=0) + box_buffer
    centroid = rng.uniform(box_min, box_max)
    coords += centroid

    return coords


def _rdmoltransforms_import():
    """Lazy import of rdMolTransforms to avoid circular import at module level."""
    from rdkit.Chem import rdMolTransforms
    return rdMolTransforms


def rdMolTransforms_set_dihedral(conf, i, j, k, l, angle_deg):
    """Set a dihedral angle in a conformer, silently ignoring failures."""
    try:
        from rdkit.Chem import rdMolTransforms
        rdMolTransforms.SetDihedralDeg(conf, i, j, k, l, angle_deg)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def _spyrmsd_mol(mol):
    """Convert an RDKit Mol to a spyrmsd Molecule for symmetry-RMSD."""
    return spyrmsd_molecule.Molecule.from_rdkit(mol)


def compute_rmsd_symmetry(mol, ref_coords, query_coords_list):
    """Compute symmetry-corrected RMSD between ref_coords and each set of
    query coordinates using spyrmsd (graph-isomorphism symmetry correction).

    This matches the metric used by evaluate.py.

    Args:
        mol: RDKit Mol defining the atom graph.
        ref_coords: numpy array (N_atoms, 3) — reference (e.g. crystal pose).
        query_coords_list: list of numpy arrays (N_atoms, 3).

    Returns:
        numpy array of shape (len(query_coords_list),).
    """
    spy_mol = _spyrmsd_mol(mol)
    adjacency = spy_mol.adjacency_matrix
    atomicnums = spy_mol.atomicnums
    results = []
    for qc in query_coords_list:
        try:
            r = spyrmsd_rmsd.symmrmsd(
                ref_coords, qc,
                atomicnums, atomicnums,
                adjacency, adjacency,
            )
            results.append(r)
        except Exception:
            results.append(np.nan)
    return np.array(results)


def compute_centroid_distance(ref_coords, query_coords_list):
    """Compute Euclidean distance between centroids.

    Args:
        ref_coords: numpy array (N_atoms, 3).
        query_coords_list: list of numpy arrays (N_atoms, 3).

    Returns:
        numpy array of shape (len(query_coords_list),).
    """
    ref_c = ref_coords.mean(axis=0)
    return np.array([
        np.linalg.norm(q.mean(axis=0) - ref_c)
        for q in query_coords_list
    ])


# ---------------------------------------------------------------------------
# Core TARP loop
# ---------------------------------------------------------------------------

def compute_tarp_fractions_one_complex(
    crystal_mol, crystal_coords, sample_coords, ca_coords,
    K, rng, mode="rmsd"
):
    """Compute K TARP coverage fractions f for a single complex.

    For each of K random reference poses θ_r:
        r   = distance(θ*, θ_r)
        d_j = distance(θ_j, θ_r)  for j in 1..S
        f_k = fraction of d_j < r

    Args:
        crystal_mol: RDKit Mol (heavy atoms, no Hs).
        crystal_coords: numpy array (N_atoms, 3) — crystal pose.
        sample_coords: list of numpy arrays (N_atoms, 3) — S DiffDock samples.
        ca_coords: numpy array (N_res, 3) — protein Cα coords for bounding box.
        K: number of reference draws.
        rng: numpy Generator.
        mode: "rmsd" for full symmetry-corrected RMSD, "centroid" for
            centroid-only Euclidean distance.

    Returns:
        numpy array of shape (K,) with values in [0, 1].
    """
    dist_fn = (
        lambda ref, queries: compute_rmsd_symmetry(crystal_mol, ref, queries)
        if mode == "rmsd"
        else compute_centroid_distance(ref, queries)
    )

    fractions = []
    for _ in range(K):
        ref_coords = generate_reference_coords(crystal_mol, ca_coords, rng)
        r = dist_fn(ref_coords, [crystal_coords])[0]
        d_samples = dist_fn(ref_coords, sample_coords)
        finite = np.isfinite(d_samples) & np.isfinite(r)
        if not finite.any() or not np.isfinite(r):
            continue
        f = (d_samples[finite] < r).mean()
        fractions.append(f)
    return np.array(fractions)


def run_tarp_eval(
    complex_names, results_index, data_dir,
    K=100, mode="rmsd", seed=42, verbose=True
):
    """Run TARP evaluation over all complexes and return the raw coverage fractions.

    Args:
        complex_names: iterable of PDB ID strings (e.g. from complex_names.npy).
        results_index: dict from build_results_index().
        data_dir: root data directory (parent of PDBBind_processed/).
        K: number of random reference points per complex.
        mode: "rmsd" or "centroid".
        seed: random seed for reproducibility.
        verbose: if True, print progress every 20 complexes.

    Returns:
        numpy array of shape (n_valid_complexes * K,) containing all per-
        (complex, reference) coverage fractions f ∈ [0, 1].
    """
    rng = np.random.default_rng(seed)
    all_fractions = []
    skipped = 0

    for i, pdb_id in enumerate(complex_names):
        if verbose and i % 20 == 0:
            print(f"  [{i}/{len(complex_names)}] processing {pdb_id} ...")
        try:
            crystal_mol, crystal_coords = load_crystal_coords(pdb_id, data_dir)
            sample_coords = load_sample_coords(pdb_id, results_index)
            ca_coords = load_protein_ca_coords(pdb_id, data_dir)

            if len(sample_coords) == 0:
                skipped += 1
                continue

            fracs = compute_tarp_fractions_one_complex(
                crystal_mol, crystal_coords, sample_coords,
                ca_coords, K, rng, mode=mode
            )
            all_fractions.append(fracs)
        except Exception as e:
            if verbose:
                print(f"    Skipping {pdb_id}: {e}")
            skipped += 1

    if verbose:
        print(f"Done. {len(all_fractions)} complexes processed, {skipped} skipped.")

    return np.concatenate(all_fractions)


# ---------------------------------------------------------------------------
# ECP computation and plotting
# ---------------------------------------------------------------------------

def ecp_from_fractions(f_values, n_bins=50):
    """Compute the Expected Coverage Probability (ECP) curve from raw coverage
    fractions.

    ecp(α) = fraction of f_values ≤ α

    Under perfect calibration ecp(α) = α for all α (diagonal line).

    Args:
        f_values: 1-D numpy array of coverage fractions in [0, 1].
        n_bins: number of α values to evaluate.

    Returns:
        (ecp, alpha): both numpy arrays of shape (n_bins,).
    """
    alpha = np.linspace(0, 1, n_bins)
    ecp = np.array([(f_values <= a).mean() for a in alpha])
    return ecp, alpha


def plot_ecp(ecp, alpha, ax=None, label=None, color=None, bootstrap_ecps=None):
    """Plot an ECP curve against the perfect-calibration diagonal.

    Args:
        ecp: numpy array of ECP values (shape n_bins).
        alpha: numpy array of credibility levels (shape n_bins).
        ax: matplotlib Axes. If None, a new figure is created.
        label: legend label for this curve.
        color: line colour.
        bootstrap_ecps: optional numpy array of shape (n_bootstrap, n_bins)
            for a shaded confidence band.

    Returns:
        matplotlib Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    c = color or "C0"
    ax.plot(alpha, ecp, color=c, lw=2, label=label)
    if bootstrap_ecps is not None:
        lo = np.percentile(bootstrap_ecps, 5, axis=0)
        hi = np.percentile(bootstrap_ecps, 95, axis=0)
        ax.fill_between(alpha, lo, hi, color=c, alpha=0.2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Credibility level α", fontsize=12)
    ax.set_ylabel("Expected coverage probability", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")

    # Annotate direction of deviation
    ax.text(0.05, 0.92, "Over-dispersed ↑", transform=ax.transAxes,
            fontsize=8, color="grey")
    ax.text(0.55, 0.05, "Mode-collapsed ↓", transform=ax.transAxes,
            fontsize=8, color="grey")
    return ax


def atc_score(ecp, alpha):
    """Compute the Area To Calibration (ATC) score: signed area between the
    ECP curve and the diagonal.  Positive = over-dispersed, negative = mode-
    collapsed, zero = perfectly calibrated.

    Args:
        ecp: numpy array of ECP values.
        alpha: numpy array of credibility levels.

    Returns:
        float scalar.
    """
    return float(np.trapz(ecp - alpha, alpha))
