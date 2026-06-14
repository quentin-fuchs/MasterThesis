"""
Compute top-1 and oracle RMSD from an inference.py output directory,
comparing against _ligand.sdf (single crystal copy, symmRMSD).

Usage:
    python analysis/pb_rmsd_from_inference.py \
        --results_dir results/pb_chaincut_100 \
        --data_dir    data/posebusters_benchmark_set \
        --label       "chain_cut=10"
"""

import argparse, os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import RemoveAllHs
from spyrmsd import rmsd as spyrmsd_rmsd, molecule as spyrmsd_mol

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def symmrmsd(ref_mol, ref_pos, pred_pos):
    spy = spyrmsd_mol.Molecule.from_rdkit(ref_mol)
    try:
        r = spyrmsd_rmsd.symmrmsd(
            ref_pos, pred_pos,
            spy.atomicnums, spy.atomicnums,
            spy.adjacency_matrix, spy.adjacency_matrix,
        )
        return float(r) if np.isscalar(r) else float(r[0])
    except Exception:
        return float(np.sqrt(((pred_pos - ref_pos)**2).sum(axis=1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--data_dir",    default="data/posebusters_benchmark_set")
    ap.add_argument("--n_samples",   type=int, default=40)
    ap.add_argument("--label",       default="chaincut")
    args = ap.parse_args()

    complexes = sorted(
        d for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d))
    )
    print(f"Found {len(complexes)} complexes in {args.results_dir}")

    top1_rmsds, oracle_rmsds = [], []
    failed = []

    for name in complexes:
        cdir = os.path.join(args.results_dir, name)

        # load crystal reference (_ligand.sdf, single copy)
        crystal_sdf = os.path.join(args.data_dir, name, f"{name}_ligand.sdf")
        if not os.path.exists(crystal_sdf):
            failed.append((name, "no crystal sdf")); continue
        ref_mol = Chem.SDMolSupplier(crystal_sdf, sanitize=True, removeHs=True)[0]
        if ref_mol is None:
            failed.append((name, "crystal parse fail")); continue
        ref_pos = ref_mol.GetConformer().GetPositions()

        # load all ranked poses
        pose_rmsds = []
        for rank in range(1, args.n_samples + 1):
            sdf = os.path.join(cdir, f"rank{rank}.sdf")
            if not os.path.exists(sdf):
                break
            pred_mol = Chem.SDMolSupplier(sdf, sanitize=True, removeHs=True)[0]
            if pred_mol is None or pred_mol.GetNumConformers() == 0:
                pose_rmsds.append(np.inf); continue
            pred_pos = pred_mol.GetConformer().GetPositions()
            if pred_pos.shape != ref_pos.shape:
                pose_rmsds.append(np.inf); continue
            pose_rmsds.append(symmrmsd(ref_mol, ref_pos, pred_pos))

        if not pose_rmsds:
            failed.append((name, "no poses")); continue

        top1_rmsds.append(pose_rmsds[0])          # rank1 = confidence-ranked top
        oracle_rmsds.append(min(r for r in pose_rmsds if np.isfinite(r))
                            if any(np.isfinite(r) for r in pose_rmsds) else np.inf)

    top1   = np.array(top1_rmsds)
    oracle = np.array(oracle_rmsds)
    N = len(top1)

    print(f"\n=== {args.label} ({N} complexes) ===")
    print(f"  Top-1  < 2Å : {100*(top1<2).mean():.2f}%  ({(top1<2).sum()}/{N})")
    print(f"  Top-1  < 5Å : {100*(top1<5).mean():.2f}%")
    print(f"  Oracle < 2Å : {100*(oracle<2).mean():.2f}%  ({(oracle<2).sum()}/{N})")
    print(f"  Median top1 : {np.nanmedian(top1):.2f}Å")
    if failed:
        print(f"  Failed: {len(failed)}")
        for n, r in failed[:10]:
            print(f"    {n}: {r}")

    # per-complex table
    print(f"\n{'Complex':15s} {'top1':>7s} {'oracle':>7s}")
    order = np.argsort(top1)
    for i in order:
        mark = "✓" if top1[i] < 2 else " "
        print(f"  {mark} {complexes[i]:13s}  {top1[i]:6.2f}  {oracle[i]:6.2f}")


if __name__ == "__main__":
    main()
