"""Reference pose prior matching DiffDock's randomize_position at t=1.

Generates random molecular conformations placed near the protein binding site,
used as the center distribution p(c) for both MIRA and TARP. The prior is:
  - Torsion angles: uniform on [-π, π] applied to an RDKit ETKDG conformer,
    using the graph-connectivity rotatable-bond definition from DiffDock.
  - Rotation: uniform on SO(3) via a random unit quaternion.
  - Translation centroid: N(Cα_COM, σ_tr² I) where
      σ_tr = std_ca * 1.4601642460337794 / 1.73
    replicating the initial_noise_std_proportion branch of randomize_position.

Typical usage
-------------
>>> template_mol, rot_bonds = prepare_reference_template(crystal_mol)
>>> ca_coords = load_protein_ca_coords(pdb_id, data_dir)
>>> ref_coords = generate_reference_coords(template_mol, rot_bonds, ca_coords, rng)
"""

import copy
import warnings

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

warnings.filterwarnings("ignore")

# From DiffDock's default_inference_args.yaml — initial_noise_std_proportion.
_INITIAL_NOISE_STD_PROPORTION = 1.4601642460337794


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
    two components each with at least 2 atoms (non-ring, non-terminal).

    Args:
        mol: RDKit Mol (heavy atoms, no Hs).

    Returns:
        List of (n0, a, b, n1) atom-index tuples for rdMolTransforms.SetDihedralRad.
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
        if nx.is_connected(G2):
            continue
        smaller = min(nx.connected_components(G2), key=len)
        if len(smaller) < 2:
            continue
        n0 = next(iter(G2.neighbors(a)))
        n1 = next(iter(G2.neighbors(b)))
        torsions.append((n0, a, b, n1))
    return torsions


def _embed_etkdg(mol):
    """Generate a fresh ETKDGv2 conformer for a heavy-atom mol.

    Args:
        mol: RDKit Mol (heavy atoms, no Hs).

    Returns:
        RDKit Mol with one ETKDG conformer, or None if embedding failed.
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
        if AllChem.EmbedMolecule(mol_h, ps) != -1:
            AllChem.MMFFOptimizeMolecule(mol_h, confId=0)
    mol_noh = Chem.RemoveAllHs(mol_h)
    return mol_noh if mol_noh.GetNumConformers() > 0 else None


def prepare_reference_template(crystal_mol):
    """Build the template used for drawing reference poses for one complex.

    Call once per complex. Returns a fresh ETKDG conformer plus the
    rotatable-bond list so that generate_reference_coords can randomise
    torsions cheaply without re-embedding.

    Args:
        crystal_mol: RDKit Mol (heavy atoms, no Hs).

    Returns:
        (template_mol, rot_bonds): template_mol has one ETKDG conformer;
        rot_bonds is a list of (n0, a, b, n1) tuples.
    """
    template = _embed_etkdg(crystal_mol)
    if template is None:
        warnings.warn("ETKDG embedding failed; falling back to crystal conformer.")
        template = copy.deepcopy(crystal_mol)
    rot_bonds = _get_rotatable_bonds(template)
    return template, rot_bonds


def generate_reference_coords(template_mol, rot_bonds, ca_coords, rng):
    """Draw one reference pose from DiffDock's prior at t = 1.

    Steps (mirroring randomize_position in DiffDock's utils/sampling.py):
      1. Copy the ETKDG template conformer.
      2. Randomise all rotatable torsion angles uniformly in [-π, π].
      3. Centre at origin.
      4. Apply a uniformly random SO(3) rotation.
      5. Translate centroid to N(Cα_COM, σ_tr² I).

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

    n_failed = 0
    for n0, a, b, n1 in rot_bonds:
        angle = rng.uniform(-np.pi, np.pi)
        try:
            rdMolTransforms.SetDihedralRad(conf, int(n0), int(a), int(b), int(n1), float(angle))
        except Exception:
            n_failed += 1
    if n_failed > 0:
        warnings.warn(f"Failed to set {n_failed}/{len(rot_bonds)} torsion angles.")

    coords = conf.GetPositions().copy()
    coords -= coords.mean(axis=0)
    coords = coords @ _random_rotation_matrix(rng).T

    protein_com = ca_coords.mean(axis=0)
    ca_centered = ca_coords - protein_com
    std_rec = np.sqrt(np.mean(np.sum(ca_centered ** 2, axis=1)))
    tr_std = std_rec * _INITIAL_NOISE_STD_PROPORTION / 1.73
    centroid = rng.normal(loc=protein_com, scale=tr_std, size=3)
    coords += centroid

    return coords
