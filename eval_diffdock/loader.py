"""DiffDock-specific data loading utilities.

Handles the DiffDock output directory structure (chunk_*/complex_name/rank*.sdf)
and the PoseBusters/PDBBind dataset layout used for evaluation.
"""

import contextlib
import os
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem
import prody as pr

warnings.filterwarnings("ignore")


def build_results_index(eval_full_dir):
    """Scan all chunk_* subdirectories and return a mapping from complex name
    to the directory containing its rank*.sdf prediction files.

    Args:
        eval_full_dir: path to the top-level results directory containing
            chunk_0/, chunk_1/, ... subdirectories.

    Returns:
        dict mapping complex_name (str) to Path of the complex subdirectory.
    """
    index = {}
    for chunk_dir in sorted(Path(eval_full_dir).glob("chunk_*")):
        for complex_dir in chunk_dir.iterdir():
            if complex_dir.is_dir():
                index[complex_dir.name] = complex_dir
    return index


def load_crystal_coords(pdb_id, data_dir):
    """Load crystal-pose heavy-atom coordinates and RDKit mol.

    Tries {pdb_id}_ligands.sdf (plural, PoseBusters style — may contain
    multiple crystallographic copies) then falls back to {pdb_id}_ligand.sdf
    and {pdb_id}_ligand.mol2 (PDBBind style).

    Args:
        pdb_id: PDB identifier string (e.g. "5ze6").
        data_dir: root data directory containing per-complex subdirectories.

    Returns:
        (mol, all_coords): RDKit Mol (heavy atoms) and list of numpy arrays
        (N_atoms, 3), one per crystal conformer.

    Raises:
        FileNotFoundError: if no ligand file is found.
        ValueError: if the file cannot be parsed or has no conformer.
    """
    ligands_path = os.path.join(data_dir, pdb_id, f"{pdb_id}_ligands.sdf")
    if os.path.exists(ligands_path):
        supplier = Chem.SDMolSupplier(ligands_path, removeHs=True)
        canonical_mol, all_coords = None, []
        for mol in supplier:
            if mol is None or mol.GetNumConformers() == 0:
                continue
            if canonical_mol is None:
                canonical_mol = mol
            all_coords.append(mol.GetConformer().GetPositions())
        if canonical_mol is not None and all_coords:
            return canonical_mol, all_coords

    for path in [
        os.path.join(data_dir, pdb_id, f"{pdb_id}_ligand.sdf"),
        os.path.join(data_dir, pdb_id, f"{pdb_id}_ligand.mol2"),
    ]:
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
        return mol, [mol.GetConformer().GetPositions()]

    raise FileNotFoundError(f"No readable ligand file for {pdb_id} in {data_dir}")


def load_sample_coords(pdb_id, results_index):
    """Load DiffDock predicted heavy-atom coordinates from rank*.sdf files.

    Args:
        pdb_id: PDB identifier string.
        results_index: dict from build_results_index().

    Returns:
        List of numpy arrays of shape (N_atoms, 3). May be shorter than 40
        if some rank files contain NaN coordinates.
    """
    complex_dir = results_index[pdb_id]

    plain = [f for f in complex_dir.iterdir()
             if f.name.startswith("rank") and f.name.endswith(".sdf")
             and "_confidence" not in f.name]
    if len(plain) > 1:
        rank_files = sorted(plain, key=lambda f: int(f.stem.replace("rank", "")))
    else:
        rank_files = sorted(
            [f for f in complex_dir.iterdir()
             if f.name.startswith("rank") and "_confidence" in f.name
             and f.name.endswith(".sdf")],
            key=lambda f: int(f.name.split("_confidence")[0].replace("rank", ""))
        )

    coords_list = []
    for sdf_file in rank_files:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            mol = Chem.SDMolSupplier(str(sdf_file), removeHs=True)[0]
        if mol is not None:
            coords_list.append(mol.GetConformer().GetPositions())
    return coords_list


def load_protein_ca_coords(pdb_id, data_dir):
    """Load Cα coordinates from the processed protein PDB file.

    Args:
        pdb_id: PDB identifier string.
        data_dir: root data directory containing per-complex subdirectories.

    Returns:
        numpy array of shape (N_residues, 3).
    """
    processed = os.path.join(data_dir, pdb_id, f"{pdb_id}_protein_processed.pdb")
    fallback  = os.path.join(data_dir, pdb_id, f"{pdb_id}_protein.pdb")
    pdb_path  = processed if os.path.exists(processed) else fallback
    prot = pr.parsePDB(pdb_path)
    return prot.ca.getCoords()
