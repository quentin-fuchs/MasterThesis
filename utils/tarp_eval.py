"""
TARP (Tests of Accuracy with Random Points) evaluation for DiffDock.

Implements the TARP coverage test from Lemos & Coogan et al. 2023
(https://arxiv.org/abs/2302.03026) applied to DiffDock's pose distributions.

For each test complex, TARP checks whether DiffDock's 40 samples are
calibrated by asking: given a random reference pose θ_r drawn from DiffDock's
prior, what fraction of DiffDock samples fall closer to θ_r than the crystal
pose does?  If this fraction, averaged over complexes and references, matches
α for all α, the posterior is perfectly calibrated.

Reference pose prior (matches DiffDock's randomize_position at t=1):
  - Torsion angles: uniform on [-π, π] applied to an RDKit ETKDG conformer
    using the same rotatable-bond definition as DiffDock (graph-connectivity).
  - Rotation: uniform on SO(3).
  - Translation centroid: N(protein_Cα_COM, σ_tr² I), where
      σ_tr = std_rec * INITIAL_NOISE_STD_PROPORTION / 1.73
      std_rec = RMS distance of Cα atoms from their own COM
    This replicates the initial_noise_std_proportion >= 0 branch of
    randomize_position in utils/sampling.py.

Two distance modes:
  - centroid: 3D centroid distance (tests binding-site localisation).
  - rmsd:     symmetry-corrected heavy-atom RMSD via spyrmsd.

Typical usage
-------------
>>> from utils.tarp_eval import build_results_index, run_tarp_eval, plot_ecp
>>> index = build_results_index("results/testset_eval_full")
>>> names = np.load("results/testset_eval_merged/complex_names.npy", allow_pickle=True)
>>> f_mat = run_tarp_eval(names, index, "data/PDBBind_processed", K=100, mode="rmsd")
>>> # f_mat has shape (n_complexes, K)
>>> ecp, alpha = ecp_from_fractions(f_mat)
>>> boot_ecps = bootstrap_ecp(f_mat, n_bootstrap=500)
>>> plot_ecp(ecp, alpha, bootstrap_ecps=boot_ecps)
"""

import copy
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from spyrmsd import rmsd as spyrmsd_rmsd, molecule as spyrmsd_molecule
import prody as pr

warnings.filterwarnings("ignore")

# np.trapezoid added in NumPy 2.0; fall back to np.trapz on older installs.
_trapz = getattr(np, "trapezoid", np.trapz)

# From default_inference_args.yaml — the proportion used in randomize_position's
# initial_noise_std_proportion >= 0 branch: std = std_rec * prop / 1.73.
_INITIAL_NOISE_STD_PROPORTION = 1.4601642460337794


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_results_index(eval_full_dir):
    """Scan all chunk_* subdirectories and return a mapping from PDB ID to the
    directory containing its rank*.sdf prediction files.

    Args:
        eval_full_dir: Path to the top-level results directory that contains
            chunk_0/, chunk_1/, ... subdirectories.

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

    Raises:
        FileNotFoundError: if no ligand file is found.
        ValueError: if the ligand file cannot be parsed or has no conformer.
    """
    sdf_path = os.path.join(data_dir, pdb_id, f"{pdb_id}_ligand.sdf")
    mol2_path = os.path.join(data_dir, pdb_id, f"{pdb_id}_ligand.mol2")
    for path in [sdf_path, mol2_path]:
        if not os.path.exists(path):
            continue
        if path.endswith(".sdf"):
            supplier = Chem.SDMolSupplier(path, removeHs=True)
            mol = supplier[0] if supplier else None
        else:
            mol = Chem.MolFromMol2File(path, removeHs=True)
        if mol is None:
            continue
        if mol.GetNumConformers() == 0:
            raise ValueError(f"{pdb_id}: ligand file has no conformer")
        return mol, mol.GetConformer().GetPositions()
    raise FileNotFoundError(f"No readable ligand file for {pdb_id} in {data_dir}")


def load_sample_coords(pdb_id, results_index):
    """Load the DiffDock predicted heavy-atom coordinate sets from rank*.sdf.

    Args:
        pdb_id: PDB identifier string.
        results_index: Dict from build_results_index().

    Returns:
        List of numpy arrays of shape (N_atoms, 3). May be shorter than 40
        if some rank files contain NaN coordinates (failed inference samples).
    """
    complex_dir = results_index[pdb_id]
    rank_files = sorted(
        [f for f in complex_dir.iterdir()
         if f.name.startswith("rank") and f.name.endswith(".sdf")
         and "_confidence" not in f.name],
        key=lambda f: int(f.stem.replace("rank", ""))
    )
    coords_list = []
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
# Reference pose generation — prior matching DiffDock's randomize_position
# ---------------------------------------------------------------------------

def _random_rotation_matrix(rng):
    """Sample a uniform rotation from SO(3) via a random unit quaternion.

    Args:
        rng: numpy Generator.

    Returns:
        (3, 3) rotation matrix.
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
    """Identify rotatable bonds using DiffDock's graph-connectivity criterion.

    A bond is rotatable if removing it disconnects the molecular graph into
    two components each containing at least 2 atoms (i.e. non-ring,
    non-terminal). This matches the definition used in
    datasets/conformer_matching.py:get_torsion_angles and
    utils/torsion.py:get_transformation_mask.

    Args:
        mol: RDKit Mol (heavy atoms, no Hs).

    Returns:
        List of (n0, a, b, n1) atom-index 4-tuples suitable for
        rdMolTransforms.SetDihedralRad.
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    torsions = []
    for a, b in G.edges():
        G2 = G.copy()
        G2.remove_edge(a, b)
        if nx.is_connected(G2):  # ring bond — skip
            continue
        smaller = min(nx.connected_components(G2), key=len)
        if len(smaller) < 2:  # terminal bond — skip
            continue
        n0 = next(iter(G2.neighbors(a)))
        n1 = next(iter(G2.neighbors(b)))
        torsions.append((n0, a, b, n1))
    return torsions


def _embed_etkdg(mol):
    """Generate a fresh ETKDGv2 conformer for a heavy-atom mol.

    Replicates datasets/process_mols.py:generate_conformer.

    Args:
        mol: RDKit Mol (heavy atoms, no Hs, no conformers required).

    Returns:
        A new RDKit Mol with one ETKDG conformer (heavy atoms, no Hs),
        or None if embedding failed.
    """
    mol_h = AllChem.AddHs(copy.deepcopy(mol))
    mol_h.RemoveAllConformers()
    ps = AllChem.ETKDGv2()
    cid, failures = -1, 0
    while cid == -1 and failures < 3:
        cid = AllChem.EmbedMolecule(mol_h, ps)
        failures += 1
    if cid == -1:
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol_h, ps)
        AllChem.MMFFOptimizeMolecule(mol_h, confId=0)
    mol_noh = Chem.RemoveAllHs(mol_h)
    return mol_noh if mol_noh.GetNumConformers() > 0 else None


def prepare_reference_template(crystal_mol):
    """Build the template used for drawing K reference poses for one complex.

    Called once per complex. Returns a fresh ETKDG conformer plus the
    rotatable-bond list so that generate_reference_coords can randomise
    torsions cheaply K times without re-embedding.

    Args:
        crystal_mol: RDKit Mol loaded from the crystal ligand SDF (heavy atoms).

    Returns:
        (template_mol, rot_bonds): template_mol is an RDKit Mol with one ETKDG
        conformer; rot_bonds is a list of (n0, a, b, n1) tuples.
        Falls back to crystal_mol if ETKDG embedding fails.
    """
    template = _embed_etkdg(crystal_mol)
    if template is None:
        warnings.warn("ETKDG embedding failed; falling back to crystal conformer.")
        template = copy.deepcopy(crystal_mol)
    rot_bonds = _get_rotatable_bonds(template)
    return template, rot_bonds


def generate_reference_coords(template_mol, rot_bonds, ca_coords, rng):
    """Draw one reference pose from DiffDock's prior at t = 1.

    Steps (mirroring randomize_position in utils/sampling.py):
      1. Copy the ETKDG template conformer (preserves bond lengths/angles).
      2. Randomise all rotatable torsion angles uniformly in [-π, π].
      3. Centre at origin.
      4. Apply a uniformly random SO(3) rotation.
      5. Translate centroid to N(Cα_COM, σ_tr² I) where
           σ_tr = std_ca * _INITIAL_NOISE_STD_PROPORTION / 1.73
           std_ca = RMS distance of Cα atoms from their COM.

    Args:
        template_mol: RDKit Mol with one ETKDG conformer (heavy atoms).
        rot_bonds: list of (n0, a, b, n1) tuples from prepare_reference_template.
        ca_coords: numpy array (N_res, 3) of protein Cα coordinates.
        rng: numpy Generator.

    Returns:
        numpy array of shape (N_atoms, 3).
    """
    mol = copy.deepcopy(template_mol)
    conf = mol.GetConformer()

    # 1. Randomise torsion angles
    n_failed = 0
    for n0, a, b, n1 in rot_bonds:
        angle = rng.uniform(-np.pi, np.pi)
        try:
            rdMolTransforms.SetDihedralRad(conf, int(n0), int(a), int(b), int(n1), float(angle))
        except Exception:
            n_failed += 1
    if n_failed > 0:
        warnings.warn(
            f"Failed to set {n_failed}/{len(rot_bonds)} torsion angles; "
            "those bonds keep their ETKDG values."
        )

    coords = conf.GetPositions().copy()

    # 2. Centre at origin
    coords -= coords.mean(axis=0)

    # 3. Random SO(3) rotation
    coords = coords @ _random_rotation_matrix(rng).T

    # 4. Gaussian translation around protein Cα COM
    protein_com = ca_coords.mean(axis=0)
    ca_centered = ca_coords - protein_com
    std_rec = np.sqrt(np.mean(np.sum(ca_centered ** 2, axis=1)))
    tr_std = std_rec * _INITIAL_NOISE_STD_PROPORTION / 1.73
    centroid = rng.normal(loc=protein_com, scale=tr_std, size=3)
    coords += centroid

    return coords


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def _spyrmsd_mol(mol):
    """Convert an RDKit Mol to a spyrmsd Molecule for symmetry-RMSD."""
    return spyrmsd_molecule.Molecule.from_rdkit(mol)


def compute_rmsd_symmetry(mol, ref_coords, query_coords_list):
    """Compute symmetry-corrected RMSD between ref_coords and each set of
    query coordinates using spyrmsd.

    Args:
        mol: RDKit Mol defining the atom graph.
        ref_coords: numpy array (N_atoms, 3).
        query_coords_list: list of numpy arrays (N_atoms, 3).

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
        try:
            r = spyrmsd_rmsd.symmrmsd(
                ref_coords, qc,
                atomicnums, atomicnums,
                adjacency, adjacency,
            )
            results.append(r)
        except Exception as exc:
            warnings.warn(f"spyrmsd failed: {exc}")
            results.append(np.nan)
    return np.array(results)


def compute_centroid_distance(ref_coords, query_coords_list):
    """Compute Euclidean centroid-to-centroid distance.

    Args:
        ref_coords: numpy array (N_atoms, 3).
        query_coords_list: list of numpy arrays (N_atoms, 3).

    Returns:
        numpy array of shape (len(query_coords_list),).
    """
    ref_c = ref_coords.mean(axis=0)
    return np.array([np.linalg.norm(q.mean(axis=0) - ref_c) for q in query_coords_list])


# ---------------------------------------------------------------------------
# Core TARP loop
# ---------------------------------------------------------------------------

def compute_tarp_fractions_one_complex(
    crystal_mol, crystal_coords, template_mol, rot_bonds,
    sample_coords, ca_coords, K, rng, mode="rmsd"
):
    """Compute K TARP coverage fractions f for a single complex.

    For each of K random reference poses θ_r:
        r   = distance(θ*, θ_r)   where θ* = crystal pose
        d_j = distance(θ_j, θ_r)  for each DiffDock sample θ_j
        f_k = fraction of d_j < r

    Args:
        crystal_mol: RDKit Mol (heavy atoms) — defines the atom graph for RMSD.
        crystal_coords: numpy array (N_atoms, 3) — crystal pose.
        template_mol: RDKit Mol with ETKDG conformer from prepare_reference_template.
        rot_bonds: list of (n0, a, b, n1) from prepare_reference_template.
        sample_coords: list of numpy arrays (N_atoms, 3) — DiffDock samples.
        ca_coords: numpy array (N_res, 3) — protein Cα coordinates.
        K: number of reference draws.
        rng: numpy Generator.
        mode: "rmsd" or "centroid".

    Returns:
        numpy array of shape (K,) with values in [0, 1]. May be shorter than
        K if reference distances are non-finite.
    """
    if mode == "rmsd":
        def dist_fn(ref, queries):
            return compute_rmsd_symmetry(crystal_mol, ref, queries)
    else:
        def dist_fn(ref, queries):
            return compute_centroid_distance(ref, queries)

    fractions = []
    for _ in range(K):
        ref_coords = generate_reference_coords(template_mol, rot_bonds, ca_coords, rng)
        r = dist_fn(ref_coords, [crystal_coords])[0]
        if not np.isfinite(r):
            continue
        d_samples = dist_fn(ref_coords, sample_coords)
        finite_mask = np.isfinite(d_samples)
        if not finite_mask.any():
            continue
        f = (d_samples[finite_mask] < r).mean()
        fractions.append(f)
    return np.array(fractions)


def _tarp_worker(args):
    """Process a single complex for TARP evaluation.

    Top-level function required for multiprocessing pickling.

    Args:
        args: tuple of (pdb_id, results_index, data_dir, K, mode, seed).

    Returns:
        (pdb_id, fracs_or_None, error_str_or_None)
    """
    pdb_id, results_index, data_dir, K, mode, seed = args
    warnings.filterwarnings("ignore")
    try:
        crystal_mol, crystal_coords = load_crystal_coords(pdb_id, data_dir)
        sample_coords = load_sample_coords(pdb_id, results_index)
        ca_coords = load_protein_ca_coords(pdb_id, data_dir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return pdb_id, None, f"load error: {exc}"

    if len(sample_coords) == 0:
        return pdb_id, None, "no valid samples after SDF parsing"

    try:
        template_mol, rot_bonds = prepare_reference_template(crystal_mol)
        rng = np.random.default_rng(seed)
        fracs = compute_tarp_fractions_one_complex(
            crystal_mol, crystal_coords, template_mol, rot_bonds,
            sample_coords, ca_coords, K, rng, mode=mode
        )
        return pdb_id, fracs, None
    except Exception as exc:
        return pdb_id, None, f"compute error: {exc}"


def run_tarp_eval(
    complex_names, results_index, data_dir,
    K=100, mode="rmsd", seed=42, verbose=True, n_workers=1
):
    """Run TARP evaluation over all complexes.

    Args:
        complex_names: iterable of PDB ID strings.
        results_index: dict from build_results_index().
        data_dir: root data directory (parent of PDBBind_processed/).
        K: number of random reference points per complex.
        mode: "rmsd" or "centroid".
        seed: random seed. Per-complex seeds are derived via SeedSequence.spawn
            so results are identical regardless of n_workers.
        verbose: if True, print progress every 20 complexes.
        n_workers: number of parallel worker processes (1 = serial).

    Returns:
        numpy array of shape (n_valid_complexes, K). Each row is the K
        coverage fractions for one complex. Use ecp_from_fractions to compute
        the ECP curve and bootstrap_ecp for confidence bands.
    """
    complex_names = list(complex_names)
    n = len(complex_names)
    child_seeds = np.random.SeedSequence(seed).spawn(n)

    work = [
        (pdb_id, results_index, data_dir, K, mode, child_seeds[i])
        for i, pdb_id in enumerate(complex_names)
    ]

    rows = []   # list of (idx, fracs) to preserve complex order
    skipped = 0
    n_done = 0

    def _handle(result):
        nonlocal skipped, n_done
        pdb_id, fracs, err = result
        if verbose and n_done % 20 == 0:
            print(f"  [{n_done}/{n}] processed {pdb_id} ...", flush=True)
        n_done += 1
        if err is not None:
            if verbose:
                print(f"    Skipping {pdb_id}: {err}", flush=True)
            skipped += 1
        elif len(fracs) > 0:
            rows.append(fracs[:K])  # trim to K in case of rare early exits

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for result in pool.imap(_tarp_worker, work):
                _handle(result)
    else:
        for result in map(_tarp_worker, work):
            _handle(result)

    if verbose:
        print(
            f"Done. {len(rows)} complexes processed, {skipped} skipped.",
            flush=True
        )

    # Pad rows shorter than K with NaN so the matrix is rectangular.
    max_k = max((len(r) for r in rows), default=0)
    if max_k == 0:
        return np.empty((0, K))
    out = np.full((len(rows), max_k), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


# ---------------------------------------------------------------------------
# ECP computation and plotting
# ---------------------------------------------------------------------------

def ecp_from_fractions(f_matrix, n_bins=50):
    """Compute the Expected Coverage Probability (ECP) curve.

    ecp(α) = fraction of f values ≤ α.  Under perfect calibration ecp = α.

    Args:
        f_matrix: numpy array of shape (n_complexes, K) from run_tarp_eval,
            or a flat 1-D array.
        n_bins: number of α values.

    Returns:
        (ecp, alpha): numpy arrays of shape (n_bins,).
    """
    f_flat = np.asarray(f_matrix).ravel()
    f_flat = f_flat[np.isfinite(f_flat)]
    alpha = np.linspace(0, 1, n_bins)
    ecp = np.array([(f_flat <= a).mean() for a in alpha])
    return ecp, alpha


def bootstrap_ecp(f_matrix, n_bins=50, n_bootstrap=500, rng=None):
    """Bootstrap confidence bands for the ECP by resampling complexes.

    Resamples rows of f_matrix (each row = one complex) with replacement.
    This gives correct uncertainty accounting for between-complex variability,
    as opposed to treating each of the n_complexes × K fractions as
    independent.

    Args:
        f_matrix: numpy array of shape (n_complexes, K) from run_tarp_eval.
        n_bins: number of α values.
        n_bootstrap: number of bootstrap replicates.
        rng: numpy Generator. If None, uses default_rng().

    Returns:
        numpy array of shape (n_bootstrap, n_bins).
    """
    if rng is None:
        rng = np.random.default_rng()
    f_matrix = np.asarray(f_matrix)
    n = len(f_matrix)
    boot_ecps = np.zeros((n_bootstrap, n_bins))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ecp, _ = ecp_from_fractions(f_matrix[idx], n_bins=n_bins)
        boot_ecps[b] = ecp
    return boot_ecps


def plot_ecp(ecp, alpha, ax=None, label=None, color=None, bootstrap_ecps=None):
    """Plot an ECP curve against the perfect-calibration diagonal.

    Args:
        ecp: numpy array of ECP values (shape n_bins).
        alpha: numpy array of credibility levels (shape n_bins).
        ax: matplotlib Axes. If None, a new figure is created.
        label: legend label.
        color: line colour.
        bootstrap_ecps: optional numpy array of shape (n_bootstrap, n_bins)
            from bootstrap_ecp(), used to draw a 90 % confidence band.
            Should be bootstrapped over complexes, not individual fractions.

    Returns:
        matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

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
    ax.text(0.05, 0.92, "Over-dispersed ↑", transform=ax.transAxes,
            fontsize=8, color="grey")
    ax.text(0.55, 0.05, "Mode-collapsed ↓", transform=ax.transAxes,
            fontsize=8, color="grey")
    return ax


def atc_score(ecp, alpha):
    """Signed area between the ECP curve and the diagonal (ATC score).

    Positive = over-dispersed, negative = mode-collapsed, 0 = calibrated.

    Args:
        ecp: numpy array of ECP values.
        alpha: numpy array of credibility levels.

    Returns:
        float.
    """
    return float(_trapz(ecp - alpha, alpha))
