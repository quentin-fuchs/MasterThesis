#!/usr/bin/env python3
"""
prepare_inputs.py

Download PDB structures from RCSB and extract protein / ligand files
ready for DiffDock inference.

Usage examples:
  # Explicit PDB IDs
  python prepare_inputs.py --pdb_ids 6jap 6w70 1a0q

  # From the testset CSV (extracts PDB IDs from the protein_path column)
  python prepare_inputs.py --testset_csv data/testset_csv.csv

  # From a CSV whose first column contains PDB IDs
  python prepare_inputs.py --pdb_csv my_targets.csv --id_col pdb_id

All outputs land in --out_dir (default: data/PDBBind_processed) and an
inference-ready CSV is written to --out_csv (default: data/inference_ready.csv).
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem, RDLogger
from rdkit.Chem import SDWriter

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ── Crystallisation artefacts / solvents / ions to exclude ───────────────────
SOLVENTS = {
    "HOH", "WAT", "DOD",                               # water
    "GOL", "EDO", "MPD", "PGE", "1PE", "P6G", "PEG",  # glycols / PEG
    "SO4", "SUL", "PO4", "ACT", "ACY", "FMT",          # sulfate, phosphate, acetate
    "CIT", "TRS", "MES", "EPE", "HEP", "BIS",          # buffers
    "BME", "DTT",                                       # reducing agents
    "DMS", "IPA", "IMD", "TLA",                         # misc solvents
    "NO3", "NO2", "SCN", "NH4",
    "LMT", "BOG", "OES",                               # detergents
    # single-atom ions
    "IOD", "CL", "NA", "MG", "ZN", "CA", "FE",
    "MN", "NI", "CU", "CO", "K", "CS", "RB",
}

RCSB_URL = "https://files.rcsb.org/download/{}.pdb"


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def download_pdb(pdb_id: str, dest: str) -> bool:
    url = RCSB_URL.format(pdb_id.upper())
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as e:
        print(f"  [ERROR] network error for {pdb_id}: {e}")
        return False
    if r.status_code != 200:
        print(f"  [ERROR] HTTP {r.status_code} for {pdb_id}")
        return False
    with open(dest, "w") as f:
        f.write(r.text)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Protein extraction
# ─────────────────────────────────────────────────────────────────────────────

class _ProteinOnly(Select):
    """Keep only standard amino-acid residues (ATOM records)."""
    def accept_residue(self, residue):
        return residue.id[0] == " "


def extract_protein(structure, dest: str) -> None:
    io = PDBIO()
    io.set_structure(structure)
    io.save(dest, _ProteinOnly())


# ─────────────────────────────────────────────────────────────────────────────
# Ligand extraction
# ─────────────────────────────────────────────────────────────────────────────

def _non_solvent_residues(structure) -> Dict[str, List]:
    """
    Collect non-solvent HETATM residues, grouped by chain_id so that
    multi-residue ligands (e.g. sucrose = GLC + FRU) are kept together.

    Returns {chain_id: [biopython_residue, ...]}
    """
    by_chain: Dict[str, list] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0].strip()
                if not hetflag or hetflag == "W":
                    continue
                if residue.resname.strip() in SOLVENTS:
                    continue
                by_chain.setdefault(chain.id, []).append(residue)
    return by_chain


def pick_ligand_residues(structure) -> Tuple[Optional[str], list]:
    """
    Pick the primary ligand as the chain whose non-solvent HETATM residues
    have the most atoms in total.  Returns (chain_id, [residues]).
    """
    by_chain = _non_solvent_residues(structure)
    if not by_chain:
        return None, []

    best_chain = max(
        by_chain,
        key=lambda c: sum(len(list(r.get_atoms())) for r in by_chain[c]),
    )
    residues = by_chain[best_chain]
    names = ", ".join(sorted({r.resname.strip() for r in residues}))
    atom_count = sum(len(list(r.get_atoms())) for r in residues)
    print(f"    ligand chain {best_chain}: {names}  ({len(residues)} residue(s), {atom_count} atoms)")
    return best_chain, residues


def residues_to_sdf(residues: list, raw_pdb: str, dest: str) -> bool:
    """
    Extract the given BioPython residues from the raw PDB file and write an SDF.
    Falls back to sanitize=False so unusual valences don't abort the write.
    """
    # Build a minimal PDB block containing only the target residue lines
    target = {
        (r.resname.strip(), r.get_parent().id, r.id[1])
        for r in residues
    }

    hetatm_lines = []
    conect_lines = []
    with open(raw_pdb) as f:
        for line in f:
            if line.startswith("HETATM"):
                resname = line[17:20].strip()
                chain   = line[21]
                resseq  = int(line[22:26].strip())
                if (resname, chain, resseq) in target:
                    hetatm_lines.append(line.rstrip())
            elif line.startswith("CONECT"):
                conect_lines.append(line.rstrip())

    pdb_block = "\n".join(hetatm_lines + conect_lines + ["END"])
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=False)
    if mol is None:
        return False

    Chem.SanitizeMol(mol, catchErrors=True)
    writer = SDWriter(dest)
    writer.write(mol)
    writer.close()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Per-complex pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_complex(pdb_id: str, out_dir: str) -> Optional[dict]:
    """
    Download and prepare one complex.
    Returns a dict with DiffDock CSV columns, or None on failure.
    """
    pdb_id = pdb_id.lower()
    complex_dir = os.path.join(out_dir, pdb_id)
    os.makedirs(complex_dir, exist_ok=True)

    raw_pdb      = os.path.join(complex_dir, f"{pdb_id}_full.pdb")
    protein_path = os.path.join(complex_dir, f"{pdb_id}_protein_processed.pdb")
    ligand_path  = os.path.join(complex_dir, f"{pdb_id}_ligand.sdf")

    if os.path.exists(protein_path) and os.path.exists(ligand_path):
        print(f"  {pdb_id}: already prepared — skipping.")
        return _record(pdb_id, protein_path, ligand_path)

    # Download
    if not os.path.exists(raw_pdb):
        print(f"  {pdb_id}: downloading from RCSB …")
        if not download_pdb(pdb_id, raw_pdb):
            return None
    else:
        print(f"  {pdb_id}: PDB already on disk.")

    # Parse
    structure = PDBParser(QUIET=True).get_structure(pdb_id, raw_pdb)

    # Protein
    extract_protein(structure, protein_path)
    print(f"  {pdb_id}: protein → {protein_path}")

    # Ligand
    chain_id, lig_residues = pick_ligand_residues(structure)
    if not lig_residues:
        print(f"  {pdb_id}: [WARN] no non-solvent ligand found.")
        return None

    if not residues_to_sdf(lig_residues, raw_pdb, ligand_path):
        print(f"  {pdb_id}: [ERROR] RDKit could not write ligand SDF.")
        return None

    print(f"  {pdb_id}: ligand  → {ligand_path}")
    return _record(pdb_id, protein_path, ligand_path)


def _record(pdb_id: str, protein_path: str, ligand_path: str) -> dict:
    return {
        "complex_name":       pdb_id,
        "protein_path":       protein_path,
        "ligand_description": ligand_path,
        "protein_sequence":   "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prepare DiffDock inputs from PDB IDs")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb_ids",     nargs="+", help="One or more PDB IDs")
    src.add_argument("--testset_csv", type=str,  help="testset_csv.csv — PDB IDs extracted from protein_path column")
    src.add_argument("--pdb_csv",     type=str,  help="CSV file with a column of PDB IDs")
    p.add_argument("--id_col",  type=str, default=None,
                   help="Column name in --pdb_csv with PDB IDs (default: first column)")
    p.add_argument("--out_dir", type=str, default="data/PDBBind_processed",
                   help="Root output directory (default: data/PDBBind_processed)")
    p.add_argument("--out_csv", type=str, default="data/inference_ready.csv",
                   help="Path for the generated inference CSV (default: data/inference_ready.csv)")
    return p.parse_args()


def collect_pdb_ids(args) -> List[str]:
    if args.pdb_ids:
        return args.pdb_ids
    if args.testset_csv:
        df = pd.read_csv(args.testset_csv)
        # Extract PDB ID from paths like data/PDBBind_processed/6jap/6jap_protein_processed.pdb
        return df["protein_path"].apply(lambda p: Path(p).parent.name).tolist()
    # --pdb_csv
    df = pd.read_csv(args.pdb_csv)
    col = args.id_col or df.columns[0]
    return df[col].tolist()


def main():
    args = parse_args()
    pdb_ids = [p.lower() for p in collect_pdb_ids(args)]
    print(f"Preparing {len(pdb_ids)} complex(es) → {args.out_dir}\n")

    records = []
    for pdb_id in pdb_ids:
        print(f"[{pdb_id.upper()}]")
        result = prepare_complex(pdb_id, args.out_dir)
        if result:
            records.append(result)

    if not records:
        print("\nNo complexes prepared successfully.")
        sys.exit(1)

    out_csv_dir = os.path.dirname(args.out_csv)
    if out_csv_dir:
        os.makedirs(out_csv_dir, exist_ok=True)
    pd.DataFrame(records).to_csv(args.out_csv, index=False)

    print(f"\nDone: {len(records)}/{len(pdb_ids)} complexes prepared.")
    print(f"Inference CSV: {args.out_csv}")
    print(f"\nNext step:")
    print(f"  python inference.py --protein_ligand_csv {args.out_csv} --out_dir results/batch_inference")


if __name__ == "__main__":
    main()
