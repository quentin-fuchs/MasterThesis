"""
Preprocess PoseBusters benchmark proteins: keep only chains whose closest
Cα atom to any ligand heavy atom is within --cutoff Å (default 10).

Writes {name}_protein_chaincut.pdb alongside the existing files.
Also writes a CSV for inference.py with 100 randomly sampled complexes.

Usage:
    python analysis/pb_preprocess_chaincut.py \
        --data_dir data/posebusters_benchmark_set \
        --split    data/posebusters_pdb_set_correct.txt \
        --cutoff   10.0 \
        --csv_out  data/pb_chaincut_100.csv \
        --n_sample 100
"""

import argparse, os, random, numpy as np
from rdkit import Chem
from rdkit.Chem import RemoveAllHs

# ─────────────────────────────────────────────────────────────────────────────

def parse_pdb_atoms(pdb_path):
    """Return dict: chain_id → list of (x,y,z) for all Cα atoms."""
    ca_by_chain = {}
    all_lines   = []
    with open(pdb_path) as f:
        for line in f:
            all_lines.append(line)
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            atom_name = line[12:16].strip()
            chain     = line[21]
            if atom_name == "CA":
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    ca_by_chain.setdefault(chain, []).append((x, y, z))
                except ValueError:
                    pass
    return all_lines, ca_by_chain


def ligand_heavy_coords(sdf_path):
    """Return (N,3) array of heavy-atom coordinates from first mol in SDF."""
    supp = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
    mol  = supp[0]
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    mol = RemoveAllHs(mol)
    if mol.GetNumConformers() == 0:
        return None
    return mol.GetConformer().GetPositions()          # (N_heavy, 3)


def chains_within_cutoff(ca_by_chain, lig_coords, cutoff):
    """
    Return set of chain IDs where any Cα is within `cutoff` Å of any ligand atom.
    """
    kept = set()
    lig  = np.array(lig_coords)                       # (N_lig, 3)
    for chain, ca_list in ca_by_chain.items():
        ca = np.array(ca_list)                        # (N_ca, 3)
        # pairwise distance matrix (N_ca, N_lig)
        diffs = ca[:, None, :] - lig[None, :, :]     # (N_ca, N_lig, 3)
        dists = np.linalg.norm(diffs, axis=-1)        # (N_ca, N_lig)
        if dists.min() <= cutoff:
            kept.add(chain)
    return kept


def write_filtered_pdb(all_lines, kept_chains, out_path):
    """Write only lines belonging to kept_chains (plus headers/END)."""
    with open(out_path, "w") as f:
        for line in all_lines:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                chain = line[21]
                if chain not in kept_chains:
                    continue
            f.write(line)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/posebusters_benchmark_set")
    ap.add_argument("--split",    default="data/posebusters_pdb_set_correct.txt")
    ap.add_argument("--cutoff",   type=float, default=10.0)
    ap.add_argument("--csv_out",  default="data/pb_chaincut_100.csv")
    ap.add_argument("--n_sample", type=int,   default=100)
    ap.add_argument("--seed",     type=int,   default=42)
    args = ap.parse_args()

    with open(args.split) as f:
        ids = [l.strip() for l in f if l.strip()]
    print(f"Processing {len(ids)} complexes with chain_cutoff={args.cutoff}Å")

    stats = {"ok": 0, "no_lig": 0, "no_ca": 0, "all_kept": 0, "reduced": 0}
    ok_ids = []         # complexes where preprocessing succeeded
    chain_log = []

    for name in ids:
        d       = os.path.join(args.data_dir, name)
        pdb_in  = os.path.join(d, f"{name}_protein.pdb")
        lig_sdf = os.path.join(d, f"{name}_ligand.sdf")   # single crystal copy
        pdb_out = os.path.join(d, f"{name}_protein_chaincut.pdb")

        if not os.path.exists(pdb_in) or not os.path.exists(lig_sdf):
            stats["no_lig"] += 1
            continue

        lig_coords = ligand_heavy_coords(lig_sdf)
        if lig_coords is None:
            stats["no_lig"] += 1
            print(f"  SKIP (no ligand coords): {name}")
            continue

        all_lines, ca_by_chain = parse_pdb_atoms(pdb_in)
        if not ca_by_chain:
            stats["no_ca"] += 1
            print(f"  SKIP (no Cα atoms): {name}")
            continue

        kept = chains_within_cutoff(ca_by_chain, lig_coords, args.cutoff)

        n_orig = len(ca_by_chain)
        n_kept = len(kept)
        if n_kept == 0:
            # Fallback: keep chain with closest Cα to ligand centroid
            lig_cen = lig_coords.mean(axis=0)
            best_chain, best_d = None, np.inf
            for chain, ca_list in ca_by_chain.items():
                d = np.linalg.norm(np.array(ca_list) - lig_cen, axis=1).min()
                if d < best_d:
                    best_d, best_chain = d, chain
            kept = {best_chain}
            print(f"  FALLBACK to closest chain {best_chain} ({best_d:.1f}Å): {name}")

        write_filtered_pdb(all_lines, kept, pdb_out)
        chain_log.append(f"{name}: {sorted(ca_by_chain.keys())} → {sorted(kept)}")

        if n_kept == n_orig:
            stats["all_kept"] += 1
        else:
            stats["reduced"] += 1
            print(f"  {name}: {n_orig} chains → {n_kept} ({sorted(kept)}, {args.cutoff}Å cutoff)")

        stats["ok"] += 1
        ok_ids.append(name)

    print(f"\n--- Summary ---")
    print(f"  Processed: {stats['ok']}")
    print(f"  Chains reduced (cutoff applied): {stats['reduced']}")
    print(f"  Chains unchanged (all within cutoff): {stats['all_kept']}")
    print(f"  Skipped (no ligand/Cα): {stats['no_lig'] + stats['no_ca']}")

    # ── write chain log ────────────────────────────────────────────────────────
    log_path = args.csv_out.replace(".csv", "_chain_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(chain_log))
    print(f"\nChain log: {log_path}")

    # ── sample 100 for inference ───────────────────────────────────────────────
    random.seed(args.seed)
    sample = random.sample(ok_ids, min(args.n_sample, len(ok_ids)))

    rows = ["protein_path,ligand_description,complex_name"]
    for name in sample:
        d       = os.path.join(args.data_dir, name)
        prot    = os.path.join(d, f"{name}_protein_chaincut.pdb")
        lig     = os.path.join(d, f"{name}_ligand_start_conf.sdf")
        if not os.path.exists(lig):
            lig = os.path.join(d, f"{name}_ligand.sdf")
        rows.append(f"{prot},{lig},{name}")

    with open(args.csv_out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Inference CSV ({len(rows)-1} complexes): {args.csv_out}")


if __name__ == "__main__":
    main()
