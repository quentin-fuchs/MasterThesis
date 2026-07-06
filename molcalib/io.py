"""Generic I/O helpers for loading ligand and protein structures.

These functions are model-agnostic and work with standard SDF, mol2, and PDB
file formats. Model-specific loaders (DiffDock rank*.sdf output, SigmaDock
predictions.pt) live in the respective eval_* directories.
"""

import os
import warnings

import numpy as np
import prody as pr
from rdkit import Chem

warnings.filterwarnings("ignore")


def load_ligand_sdf(ligand_path, remove_hs=True):
    """Load heavy-atom RDKit Mol and coordinates from an SDF or mol2 file.

    If the file contains multiple records (e.g. crystallographic copies), the
    first record is used as the canonical molecule and all conformer coordinates
    are returned.

    Args:
        ligand_path: path to an SDF or mol2 file.
        remove_hs: remove hydrogens (default True).

    Returns:
        (mol, all_coords): RDKit Mol and list of numpy arrays (N_atoms, 3),
        one per conformer in the file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if no valid conformer can be parsed.
    """
    if not os.path.exists(ligand_path):
        raise FileNotFoundError(f"Ligand file not found: {ligand_path}")

    path = str(ligand_path)
    if path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(path, removeHs=remove_hs)
        if mol is None or mol.GetNumConformers() == 0:
            raise ValueError(f"Cannot parse ligand: {path}")
        return mol, [mol.GetConformer().GetPositions()]

    supplier = Chem.SDMolSupplier(path, removeHs=remove_hs)
    canonical_mol = None
    all_coords = []
    for mol in supplier:
        if mol is None or mol.GetNumConformers() == 0:
            continue
        if canonical_mol is None:
            canonical_mol = mol
        all_coords.append(mol.GetConformer().GetPositions())

    if canonical_mol is None or not all_coords:
        raise ValueError(f"No valid conformer in: {path}")
    return canonical_mol, all_coords


def load_protein_ca_coords(protein_pdb_path):
    """Load Cα coordinates from a processed PDB file.

    Args:
        protein_pdb_path: path to a PDB file containing the protein structure.

    Returns:
        numpy array of shape (N_residues, 3).

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    if not os.path.exists(protein_pdb_path):
        raise FileNotFoundError(f"Protein PDB not found: {protein_pdb_path}")
    prot = pr.parsePDB(str(protein_pdb_path))
    return prot.ca.getCoords()
