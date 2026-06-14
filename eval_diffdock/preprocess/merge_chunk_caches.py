"""
Merge the 6 chunk-level PDBBind caches into a single timesplit_test cache.

The full testset evaluation was run as 6 parallel chunk jobs. This script
combines their cached graphs into the single cache directory that
diffdock_top1_array.sh expects, so the array job uses the exact same set of
complexes as the original evaluation (enabling a fair comparison).

Both the torsion cache (score model) and the allatoms cache (confidence model)
are merged. Run directly on the login node — no GPU needed.

Usage:
    python eval_diffdock/preprocess/merge_chunk_caches.py
"""

import os
import pickle
from pathlib import Path

DIFFDOCK_DIR = str(Path(__file__).resolve().parents[2] / "diffdock")

TORSION_BASE = os.path.join(DIFFDOCK_DIR, "data/cache_torsion")
ALLATOMS_BASE = os.path.join(DIFFDOCK_DIR, "data/cache_torsion_allatoms")

TORSION_CHUNK_SUFFIX = (
    "_maxLigSizeNone_H0_recRad15.0_recMax24_chainCutoffNone"
    "_esmEmbeddings_full_fixedKNNonly_chainOrd"
)
ALLATOMS_CHUNK_SUFFIX = (
    "_maxLigSizeNone_H0_recRad15.0_recMax24_chainCutoffNone"
    "_atomRad5_atomMax8_esmEmbeddings_full_fixedKNNonly_chainOrd"
)

TORSION_TARGET = os.path.join(
    TORSION_BASE,
    f"pdbbind3_limit0_INDEXtimesplit_test{TORSION_CHUNK_SUFFIX}",
)
ALLATOMS_TARGET = os.path.join(
    ALLATOMS_BASE,
    f"pdbbind3_limit0_INDEXtimesplit_test{ALLATOMS_CHUNK_SUFFIX}",
)

N_CHUNKS = 6


def merge_cache(base_dir, chunk_suffix, target_dir, cache_label):
    """Load all chunk caches and write a merged timesplit_test cache.

    Args:
        base_dir: Root cache directory (cache_torsion or cache_torsion_allatoms).
        chunk_suffix: Filename suffix shared by all chunk cache dirs.
        target_dir: Destination cache directory for the merged output.
        cache_label: Human-readable label for progress messages.
    """
    if os.path.exists(os.path.join(target_dir, "heterographs0.pkl")):
        print(f"  [{cache_label}] already exists at {target_dir}, skipping.")
        return

    all_graphs = []
    all_ligands = []
    all_names = []

    for i in range(N_CHUNKS):
        chunk_dir = os.path.join(base_dir, f"pdbbind3_limit0_INDEXchunk_{i}{chunk_suffix}")
        graphs_path = os.path.join(chunk_dir, "heterographs0.pkl")
        ligands_path = os.path.join(chunk_dir, "rdkit_ligands0.pkl")
        # names file uses first 3 chars of the split filename: "chunk_N" → "chu"
        names_path = os.path.join(chunk_dir, "pdbbind_chu_names.txt")

        if not os.path.exists(graphs_path):
            print(f"  [{cache_label}] WARNING: chunk {i} not found at {chunk_dir}, skipping.")
            continue

        with open(graphs_path, "rb") as f:
            graphs = pickle.load(f)
        with open(ligands_path, "rb") as f:
            ligands = pickle.load(f)
        with open(names_path) as f:
            names = [line.strip() for line in f if line.strip()]

        print(f"  [{cache_label}] chunk {i}: {len(graphs)} complexes")
        all_graphs.extend(graphs)
        all_ligands.extend(ligands)
        all_names.extend(names)

    print(f"  [{cache_label}] total: {len(all_graphs)} complexes → {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    with open(os.path.join(target_dir, "heterographs0.pkl"), "wb") as f:
        pickle.dump(all_graphs, f)
    with open(os.path.join(target_dir, "rdkit_ligands0.pkl"), "wb") as f:
        pickle.dump(all_ligands, f)
    # names file uses first 3 chars of "timesplit_test" → "tim"
    with open(os.path.join(target_dir, "pdbbind_tim_names.txt"), "w") as f:
        f.write("\n".join(all_names))


if __name__ == "__main__":
    print("Merging torsion caches (score model)...")
    merge_cache(TORSION_BASE, TORSION_CHUNK_SUFFIX, TORSION_TARGET, "torsion")

    print("\nMerging allatoms caches (confidence model)...")
    merge_cache(ALLATOMS_BASE, ALLATOMS_CHUNK_SUFFIX, ALLATOMS_TARGET, "allatoms")

    print("\nDone. You can now submit diffdock_top1_array.sh.")
