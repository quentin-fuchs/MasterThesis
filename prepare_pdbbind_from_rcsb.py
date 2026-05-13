"""
Reconstruct DiffDock's PDBBind_processed layout from the official RCSB PDB.

For each test complex this script:
  1. Downloads the PDB structure from RCSB (the canonical source for all PDB data,
     including PDBBind which cites RCSB as its primary source).
  2. Identifies the primary biological ligand using the RCSB REST API (returns the
     same nonpolymer entity information PDBBind uses).
  3. Retrieves the correct bond orders for that ligand from RCSB's Chemical Component
     Dictionary (CCD) and applies them to the bound conformation via RDKit's
     AssignBondOrdersFromTemplate — the same procedure used by EquiBind/DiffDock's
     original data-preparation pipeline.
  4. Writes:
       {pdb_id}_protein_processed.pdb  — ATOM records only (no waters/HETATM)
       {pdb_id}_ligand.sdf             — bound conformation with correct bond orders

Output mirrors what DiffDock expects:
  data/PDBBind_processed/{pdb_id}/{pdb_id}_protein_processed.pdb
  data/PDBBind_processed/{pdb_id}/{pdb_id}_ligand.sdf

Usage (run from project root on CSD3):
  python prepare_pdbbind_from_rcsb.py \
      --csv data/testset_csv.csv \
      --out data/PDBBind_processed \
      --workers 4
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, MolToMolBlock, MolFromSmiles
from rdkit.Chem import rdmolops

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ constants
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_NONPOLY_URL = "https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{entity_id}"
RCSB_CCD_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{ccd_id}"

# Crystallographic additives, buffers, and ions to exclude from ligand selection
BUFFER_SET = {
    "HOH","DOD","WAT",
    "SO4","PO4","NO3","ACT","GOL","EDO","PEG","MPD","BTB",
    "DMS","ACE","ACY","FMT","IMD","TRS","MES","EPE","PIP",
    "BOG","LMT","BCN","P6G","PE4","PE5","PE7","PE8",
    "1PE","2PE","3PE","4PE","EOH","IPA","PGE","PGO","PG4","PG6",
    "SPM","SPK","SPS",
    "MG","ZN","CA","NA","CL","K","FE","MN","CU","CO","NI","SE",
    "BR","I","F","AU","AG","PT","HG","CD","PB","BA","SR","CS","RB",
    # modified AA residues that appear as HETATM
    "MSE","MLY","CSO","CME","OCS","KCX","LLP","SEB","TPO","SEP","PTR","PCA",
    "HYP","FME","CSD","SEC","PYL",
}
MIN_HEAVY_ATOMS = 7


# ------------------------------------------------------------------ helpers
def fetch_json(url: str, retries: int = 3, delay: float = 1.0):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return None


def fetch_text(url: str, retries: int = 3, delay: float = 1.0) -> Optional[str]:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return None


def download_pdb(pdb_id: str, path: Path, retries: int = 3) -> bool:
    if path.exists():
        return True
    content = fetch_text(RCSB_PDB_URL.format(pdb_id=pdb_id.upper()), retries=retries)
    if content is None:
        return False
    path.write_text(content)
    return True


# ------------------------------------------------------------------ protein
def write_protein(raw_pdb: Path, out: Path) -> None:
    """Write ATOM-only records from the raw PDB file."""
    lines = []
    with open(raw_pdb) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec in ("ATOM", "TER", "END"):
                lines.append(line)
    with open(out, "w") as fh:
        fh.writelines(lines)
        if not lines or lines[-1].strip() != "END":
            fh.write("END\n")


# ------------------------------------------------------------------ ligand
def get_primary_ligand_ccd(pdb_id: str):
    """
    Use the RCSB REST API to find the primary nonpolymer entity (ligand).
    Returns (ccd_id, entity_id) or (None, None).
    """
    entry = fetch_json(RCSB_ENTRY_URL.format(pdb_id=pdb_id.upper()))
    if entry is None:
        return None, None

    entity_ids = (entry.get("rcsb_entry_container_identifiers", {})
                      .get("non_polymer_entity_ids", []) or [])
    if not entity_ids:
        return None, None

    best_ccd, best_eid, best_natoms = None, None, -1
    for eid in entity_ids:
        np_data = fetch_json(RCSB_NONPOLY_URL.format(pdb_id=pdb_id.upper(), entity_id=eid))
        if np_data is None:
            continue
        ccd_id = (np_data.get("pdbx_entity_nonpoly", {}) or {}).get("comp_id", "")
        if not ccd_id or ccd_id in BUFFER_SET:
            continue
        # Count heavy atoms via CCD
        ccd = fetch_json(RCSB_CCD_URL.format(ccd_id=ccd_id))
        if ccd is None:
            continue
        formula = (ccd.get("chem_comp", {}) or {}).get("formula", "")
        # crude heavy-atom count from formula (just C+N+O+S+P+halides)
        import re
        heavy = sum(int(n or 1) for n in re.findall(r"[A-Z][a-z]?\s*(\d*)", formula)
                    if True)  # count all elements
        if heavy >= MIN_HEAVY_ATOMS and heavy > best_natoms:
            best_ccd, best_eid, best_natoms = ccd_id, eid, heavy

    return best_ccd, best_eid


def get_ccd_smiles(ccd_id: str) -> Optional[str]:
    """Return canonical SMILES from the RCSB CCD for a chemical component."""
    ccd = fetch_json(RCSB_CCD_URL.format(ccd_id=ccd_id))
    if ccd is None:
        return None
    # Prefer the canonical SMILES with stereo
    for key in ("pdbx_smiles", "smiles"):
        descriptors = ccd.get("pdbx_chem_comp_descriptor", []) or []
        if not isinstance(descriptors, list):
            descriptors = [descriptors]
        for d in descriptors:
            if isinstance(d, dict) and d.get("type", "").lower() in ("smiles_canonical", "smiles"):
                smi = d.get("descriptor", "")
                if smi:
                    return smi
    # Fallback: formula-based (will miss stereo)
    return None


def extract_hetatm_block(raw_pdb: Path, ccd_id: str) -> Optional[str]:
    """
    Extract the first occurrence of ccd_id HETATM residue as a mini-PDB block.
    Returns None if not found.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    with open(raw_pdb) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            rn = line[17:20].strip()
            ch = line[21].strip()
            rs = line[22:26].strip()
            if rn == ccd_id:
                groups[(ch, rs)].append(line)

    if not groups:
        return None

    # Pick the group with most heavy atoms
    def heavy_count(lines):
        return sum(1 for l in lines
                   if len(l) >= 78 and l[76:78].strip() not in ("H", "D", ""))

    best = max(groups.values(), key=heavy_count)
    return "".join(best) + "END\n"


def make_ligand_sdf(pdb_id: str, raw_pdb: Path, ccd_id: str) -> Optional[str]:
    """
    Extract the bound ligand conformation and assign bond orders from RCSB CCD SMILES.
    Uses RDKit AssignBondOrdersFromTemplate for correct bond orders.
    """
    hetatm_block = extract_hetatm_block(raw_pdb, ccd_id)
    if hetatm_block is None:
        return None

    # Parse bound conformation from PDB block
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        tmp.write(hetatm_block)
        tmp_path = tmp.name
    try:
        raw_mol = Chem.MolFromPDBFile(tmp_path, sanitize=False, removeHs=True)
    finally:
        os.unlink(tmp_path)

    if raw_mol is None:
        return None

    # Try to get CCD SMILES for bond-order template
    smiles = get_ccd_smiles(ccd_id)
    if smiles:
        try:
            template = MolFromSmiles(smiles)
            if template is not None:
                mol = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)
                try:
                    Chem.SanitizeMol(mol)
                    return MolToMolBlock(mol)
                except Exception:
                    pass
        except Exception:
            pass

    # Fallback: sanitize the raw PDB-parsed molecule directly
    try:
        Chem.SanitizeMol(raw_mol)
        return MolToMolBlock(raw_mol)
    except Exception:
        return None


# ------------------------------------------------------------------ fallback ligand (no API)
def fallback_ligand_by_size(raw_pdb: Path):
    """Pick largest non-buffer HETATM residue when API is unavailable."""
    from collections import defaultdict
    groups = defaultdict(list)
    with open(raw_pdb) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            rn = line[17:20].strip()
            ch = line[21].strip()
            rs = line[22:26].strip()
            el = line[76:78].strip() if len(line) >= 78 else ""
            groups[(rn, ch, rs)].append((line, el))

    candidates = []
    for (rn, ch, rs), atoms in groups.items():
        if rn in BUFFER_SET:
            continue
        heavy = sum(1 for _, el in atoms if el not in ("H", "D", ""))
        if heavy >= MIN_HEAVY_ATOMS:
            candidates.append((rn, ch, rs, heavy, atoms))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[3], reverse=True)
    rn, ch, rs, _, atoms = candidates[0]
    return rn, "".join(a for a, _ in atoms) + "END\n"


# ------------------------------------------------------------------ per-complex
def process_one(args):
    pdb_id, out_root, tmp_dir = args
    out_dir = Path(out_root) / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)

    protein_out = out_dir / f"{pdb_id}_protein_processed.pdb"
    ligand_out  = out_dir / f"{pdb_id}_ligand.sdf"

    if protein_out.exists() and ligand_out.exists():
        return pdb_id, "skipped"

    raw_pdb = Path(tmp_dir) / f"{pdb_id}.pdb"
    if not download_pdb(pdb_id, raw_pdb):
        return pdb_id, "download_failed"

    # Protein
    try:
        write_protein(raw_pdb, protein_out)
    except Exception as e:
        return pdb_id, f"protein_error: {e}"

    # Ligand — try API first, fall back to size heuristic
    try:
        ccd_id, _ = get_primary_ligand_ccd(pdb_id)
        if ccd_id:
            sdf = make_ligand_sdf(pdb_id, raw_pdb, ccd_id)
            if sdf:
                ligand_out.write_text(sdf)
                return pdb_id, f"ok via API ({ccd_id})"

        # Fallback
        ccd_id, hetatm_block = fallback_ligand_by_size(raw_pdb)
        if hetatm_block is None:
            return pdb_id, "no_ligand"
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            tmp.write(hetatm_block)
            tmp_path = tmp.name
        try:
            mol = Chem.MolFromPDBFile(tmp_path, sanitize=True, removeHs=True)
        finally:
            os.unlink(tmp_path)
        if mol is None:
            return pdb_id, f"rdkit_failed ({ccd_id})"
        sdf = MolToMolBlock(mol)
        ligand_out.write_text(sdf)
        return pdb_id, f"ok via fallback ({ccd_id})"

    except Exception as e:
        return pdb_id, f"ligand_error: {e}"


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="data/testset_csv.csv")
    parser.add_argument("--out",     default="data/PDBBind_processed")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tmp",     default="/tmp/pdbbind_raw")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, index_col=0)
    pdb_ids = df["protein_path"].apply(lambda p: p.split("/")[-2]).tolist()
    print(f"Processing {len(pdb_ids)} complexes → {args.out}")

    os.makedirs(args.tmp, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    job_args = [(pid, args.out, args.tmp) for pid in pdb_ids]

    results = {}
    with Pool(args.workers) as pool:
        for pdb_id, status in tqdm(pool.imap_unordered(process_one, job_args),
                                   total=len(pdb_ids), desc="Preprocessing"):
            results[pdb_id] = status

    ok      = sum(1 for s in results.values() if s.startswith("ok"))
    skipped = sum(1 for s in results.values() if s == "skipped")
    failed  = {k: v for k, v in results.items()
               if not v.startswith("ok") and v != "skipped"}

    print(f"\nDone: {ok} ok, {skipped} skipped, {len(failed)} failed")
    if failed:
        print("\nFailed:")
        for k, v in failed.items():
            print(f"  {k}: {v}")

    pd.DataFrame(list(results.items()), columns=["pdb_id","status"]).to_csv(
        os.path.join(args.out, "preprocessing_summary.csv"), index=False)


if __name__ == "__main__":
    main()
