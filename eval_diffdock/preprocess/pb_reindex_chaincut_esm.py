"""
Reindex posebusters_esm2_embeddings.pt for chain-cut proteins.

The original embeddings use keys {name}_chain_{i} where i is the 0-based index
of the i-th unique chain ID (sorted alphabetically, including segment prefix)
in the FULL protein PDB. When we trim proteins to keep only nearby chains, the
embedding indices must be remapped so that index i refers to the i-th kept chain
(not the i-th chain of the full protein).

Without this fix, evaluate.py skips any complex where chains were removed
because the concatenated ESM embeddings have more rows than the Cα count of
the trimmed protein, causing a shape mismatch in torch.cat.

Usage:
    python analysis/pb_reindex_chaincut_esm.py \
        --split_path data/pb_chaincut_100_split.txt \
        --data_dir   data/posebusters_benchmark_set \
        --esm_in     data/posebusters_esm2_embeddings.pt \
        --esm_out    data/posebusters_esm2_embeddings_chaincut.pt
"""

import argparse, os
import torch
import numpy as np


def get_sorted_chain_ids(pdb_path):
    """
    Return sorted unique chain IDs from a PDB file in the same way that
    moad_extract_receptor_structure does: seg_name + chain_letter, sorted.

    Returns [] if the file is unreadable or has no Cα atoms.
    """
    import prody as pr
    pdb = pr.parsePDB(pdb_path, verbosity='none')
    if pdb is None:
        return []
    ca = pdb.ca
    if ca is None or len(ca) == 0:
        return []
    chain_ids = np.asarray([s + c for s, c in zip(ca.getSegnames(), ca.getChids())])
    return list(np.unique(chain_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_path",   default="data/pb_chaincut_100_split.txt")
    ap.add_argument("--data_dir",     default="data/posebusters_benchmark_set")
    ap.add_argument("--esm_in",       default="data/posebusters_esm2_embeddings.pt")
    ap.add_argument("--esm_out",      default="data/posebusters_esm2_embeddings_chaincut.pt")
    ap.add_argument("--protein_orig", default="protein")
    ap.add_argument("--protein_cut",  default="protein_chaincut")
    args = ap.parse_args()

    with open(args.split_path) as f:
        names = [l.strip() for l in f if l.strip()]
    print(f"Reindexing ESM embeddings for {len(names)} complexes")

    orig_emb = torch.load(args.esm_in, map_location='cpu')
    print(f"Loaded {len(orig_emb)} embedding keys from {args.esm_in}")

    new_emb = {}
    unchanged, remapped, skipped = 0, 0, 0

    for name in names:
        orig_pdb = os.path.join(args.data_dir, name, f"{name}_{args.protein_orig}.pdb")
        cut_pdb  = os.path.join(args.data_dir, name, f"{name}_{args.protein_cut}.pdb")

        if not os.path.exists(orig_pdb) or not os.path.exists(cut_pdb):
            print(f"  SKIP (missing pdb): {name}")
            skipped += 1
            continue

        orig_chains = get_sorted_chain_ids(orig_pdb)
        cut_chains  = get_sorted_chain_ids(cut_pdb)

        if not orig_chains or not cut_chains:
            print(f"  SKIP (no Cα): {name}")
            skipped += 1
            continue

        for new_idx, chain_id in enumerate(cut_chains):
            if chain_id not in orig_chains:
                print(f"  WARNING: chain '{chain_id}' not in original for {name}")
                orig_idx = new_idx
            else:
                orig_idx = orig_chains.index(chain_id)

            orig_key = f"{name}_chain_{orig_idx}"
            new_key  = f"{name}_chain_{new_idx}"

            if orig_key not in orig_emb:
                print(f"  WARNING: key {orig_key} missing from embeddings")
                continue
            new_emb[new_key] = orig_emb[orig_key]

        if cut_chains == orig_chains:
            unchanged += 1
        else:
            remapped += 1
            print(f"  {name}: {orig_chains} → {cut_chains}")

    print(f"\nUnchanged: {unchanged}  Remapped: {remapped}  Skipped: {skipped}")
    print(f"Writing {len(new_emb)} keys to {args.esm_out}")
    torch.save(new_emb, args.esm_out)
    print("Done.")


if __name__ == "__main__":
    main()
