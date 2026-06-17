"""MIRA + TARP evaluation for SigmaDock PoseBusters results.

Loads predictions from seed_*/predictions.pt files, then uses molcalib
for the core MIRA and TARP computations.

Args (CLI):
    results_dir:     model directory containing seed_*/predictions.pt
    --data-dir:      PoseBusters dataset root (required for TARP and symRMSD MIRA)
    --metric:        MIRA distance metric: euclidean, rmsd, or symrmsd
    --num-runs:      MIRA Monte Carlo draws per complex (default 100)
    --K:             TARP reference draws per complex (default 20)
    --mode:          TARP distance: centroid (fast) or rmsd
    --n-bootstrap:   ECP bootstrap replicates (default 200)
    --output:        output .npz path (default: <results_dir>/mira_tarp.npz)
    --skip-mira:     skip MIRA and load scores from existing mira_tarp.npz
"""

import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from eval_sigmadock.loader import load_sigmadock_poses
from molcalib.mira import mira_null, mira_score, _mira_euclidean
from molcalib.tarp import tarp_fractions, ecp_from_fractions, bootstrap_ecp, plot_ecp, atc_score
from molcalib.prior import prepare_reference_template


def run_mira(complex_data, num_runs=20, metric="euclidean", seed=42, data_dir=None):
    """Compute per-complex MIRA scores for SigmaDock predictions.

    Args:
        complex_data: output of load_sigmadock_poses().
        num_runs: Monte Carlo center draws per complex.
        metric: "euclidean", "rmsd", or "symrmsd".
        seed: master random seed (symrmsd only).
        data_dir: PoseBusters dataset root. Required when metric="symrmsd".

    Returns:
        (names, scores): numpy arrays of shape (n_valid,).
    """
    if metric == "symrmsd" and data_dir is None:
        raise ValueError("data_dir required when metric='symrmsd'")

    use_symrmsd = metric == "symrmsd"
    if not use_symrmsd:
        from mira_score import get_device
        device = get_device()
    else:
        device = None
        from molcalib.io import load_protein_ca_coords as _load_ca

    complex_ids = list(complex_data)
    n = len(complex_ids)
    child_seeds = np.random.SeedSequence(seed).spawn(n) if use_symrmsd else None
    names, scores = [], []

    for i, cid in enumerate(complex_ids):
        if i % 20 == 0:
            print(f"  MIRA [{i}/{n}] {cid} ...", flush=True)

        lig_ref, sample_coords = complex_data[cid]
        crystal = lig_ref.GetConformer().GetPositions()

        if use_symrmsd:
            pdb_id = cid.split("::")[0]
            import os
            pdb_path_processed = os.path.join(data_dir, pdb_id, f"{pdb_id}_protein_processed.pdb")
            pdb_path_fallback  = os.path.join(data_dir, pdb_id, f"{pdb_id}_protein.pdb")
            pdb_path = pdb_path_processed if os.path.exists(pdb_path_processed) else pdb_path_fallback
            try:
                ca_coords = _load_ca(pdb_path)
                template_mol, rot_bonds = prepare_reference_template(lig_ref)
            except Exception as exc:
                print(f"    Skipping {cid} (setup): {exc}", flush=True)
                continue
            rng = np.random.default_rng(child_seeds[i])
            score = mira_score(
                lig_ref, crystal, sample_coords,
                template_mol, rot_bonds, ca_coords,
                num_runs=num_runs, rng=rng,
            )
        else:
            score = _mira_euclidean(crystal, sample_coords, num_runs, device, metric)

        if not np.isnan(score):
            names.append(cid)
            scores.append(score)

    S = len(next(iter(complex_data.values()))[1])
    print(f"MIRA done: {len(scores)}/{n} complexes.")
    print(f"  Mean MIRA = {np.mean(scores):.4f}  (null S={S}: {mira_null(S):.4f})")
    return np.array(names), np.array(scores, dtype=float)


def run_tarp(complex_data, data_dir, K=20, mode="centroid", seed=42):
    """Compute TARP coverage fractions for SigmaDock predictions.

    Args:
        complex_data: output of load_sigmadock_poses().
        data_dir: PoseBusters dataset root (for protein Cα coords).
        K: random reference draws per complex.
        mode: "centroid" (fast) or "rmsd".
        seed: master random seed.

    Returns:
        (names, f_matrix): names array and float matrix (n_valid, K).
    """
    import os
    from molcalib.io import load_protein_ca_coords as _load_ca

    complex_ids = list(complex_data)
    n = len(complex_ids)
    child_seeds = np.random.SeedSequence(seed).spawn(n)
    names_out, rows = [], []

    for i, cid in enumerate(complex_ids):
        if i % 20 == 0:
            print(f"  TARP [{i}/{n}] {cid} ...", flush=True)

        lig_ref, sample_coords = complex_data[cid]
        crystal = lig_ref.GetConformer().GetPositions()
        pdb_id = cid.split("::")[0]
        pdb_path_processed = os.path.join(data_dir, pdb_id, f"{pdb_id}_protein_processed.pdb")
        pdb_path_fallback  = os.path.join(data_dir, pdb_id, f"{pdb_id}_protein.pdb")
        pdb_path = pdb_path_processed if os.path.exists(pdb_path_processed) else pdb_path_fallback
        try:
            ca_coords = _load_ca(pdb_path)
        except Exception as exc:
            print(f"    Skipping {cid} (protein load): {exc}", flush=True)
            continue

        template_mol, rot_bonds = prepare_reference_template(lig_ref)
        rng = np.random.default_rng(child_seeds[i])
        fracs = tarp_fractions(
            lig_ref, crystal, template_mol, rot_bonds,
            sample_coords, ca_coords, K=K, rng=rng, mode=mode,
        )
        if len(fracs) > 0:
            names_out.append(cid)
            rows.append(fracs[:K])

    max_k = max((len(r) for r in rows), default=0)
    if max_k == 0:
        return np.array(names_out), np.empty((0, K))
    f_matrix = np.full((len(rows), max_k), np.nan)
    for j, r in enumerate(rows):
        f_matrix[j, :len(r)] = r

    print(f"TARP done: {len(rows)}/{n} complexes.")
    return np.array(names_out), f_matrix


def main(args):
    model_dir = Path(args.results_dir)
    out_path  = Path(args.output) if args.output else model_dir / "mira_tarp.npz"
    plot_path = out_path.with_suffix(".png")

    print(f"Loading poses from {model_dir} ...", flush=True)
    complex_data = load_sigmadock_poses(model_dir)
    S = len(next(iter(complex_data.values()))[1])
    print(f"Loaded {len(complex_data)} complexes, {S} seeds each.\n")

    if args.skip_mira:
        print("=== MIRA skipped (--skip-mira) ===")
        existing = np.load(str(model_dir / "mira_tarp.npz"), allow_pickle=True)
        mira_names  = existing["mira_names"]
        mira_scores = existing["mira_scores"]
    else:
        print("=== MIRA ===")
        mira_names, mira_scores = run_mira(
            complex_data, num_runs=args.num_runs, metric=args.metric,
            seed=42, data_dir=args.data_dir,
        )

    tarp_names = np.array([], dtype=object)
    f_matrix   = np.empty((0, args.K))
    ecp = alpha = boot_ecps = np.array([])

    if args.data_dir:
        print("\n=== TARP ===")
        tarp_names, f_matrix = run_tarp(
            complex_data, data_dir=args.data_dir,
            K=args.K, mode=args.mode, seed=42,
        )
        if f_matrix.shape[0] > 0:
            ecp, alpha = ecp_from_fractions(f_matrix)
            boot_ecps  = bootstrap_ecp(f_matrix, n_bootstrap=args.n_bootstrap)
            print(f"\n  TARP AUC = {np.trapz(ecp, alpha):.4f}  (perfect = 0.5000)")

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            plot_ecp(ecp, alpha, ax=ax,
                     label=f"SigmaDock (N={len(tarp_names)})",
                     bootstrap_ecps=boot_ecps)
            ax.set_title(f"TARP ECP — SigmaDock ({args.mode} mode)")
            fig.tight_layout()
            fig.savefig(str(plot_path), dpi=150)
            print(f"  ECP plot → {plot_path}")
    else:
        print("\n(TARP skipped — pass --data-dir to enable)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path),
             mira_names=mira_names, mira_scores=mira_scores,
             tarp_names=tarp_names, tarp_f_matrix=f_matrix,
             tarp_ecp=ecp, tarp_alpha=alpha, tarp_boot_ecps=boot_ecps)
    print(f"\nResults → {out_path}")

    print("\n" + "=" * 50)
    print(f"  Complexes : {len(complex_data)}  |  Seeds : {S}")
    print(f"  MIRA mean : {mira_scores.mean():.4f}  (null S={S}: {mira_null(S):.4f})")
    delta = mira_scores.mean() - mira_null(S)
    print(f"  MIRA Δnull: {delta:+.4f}  "
          f"({'over-dispersed' if delta > 0 else 'mode-collapsed'})")
    if f_matrix.shape[0] > 0:
        print(f"  TARP AUC  : {np.trapz(ecp, alpha):.4f}  (perfect = 0.5000)")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MIRA + TARP evaluation for SigmaDock PoseBusters results."
    )
    parser.add_argument("results_dir")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--metric", choices=["euclidean", "rmsd", "symrmsd"],
                        default="euclidean")
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--mode", choices=["centroid", "rmsd"], default="centroid")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-mira", action="store_true")
    main(parser.parse_args())
