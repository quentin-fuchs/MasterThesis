"""MIRA score and TARP coverage curve evaluation for SigmaDock PoseBusters results.

Loads predictions from SigmaDock's seed_*/predictions.pt files, then calls
DiffDock's MIRA and TARP core functions directly.

- MIRA (Sharief et al. 2026): measures posterior calibration — are the 40
  samples spread the right amount around the crystal pose?
  Score ~ 0.683 = perfect calibration (S=40).  >0.683 = over-dispersed,
  <0.683 = mode-collapsed.

- TARP (Lemos et al. 2023): coverage test — does the fraction of samples
  closer to a random reference than the crystal lie along the diagonal?
  ECP curve above diagonal = over-confident, below = over-dispersed.

Both metrics operate on the full unranked set of 40 poses per complex.

Args (CLI):
    results_dir:  model directory containing seed_*/predictions.pt
    --data-dir:   root of PoseBusters dataset (parent of complex subdirs).
                  Required for TARP (protein Cα). If omitted, TARP is skipped.
    --K:          TARP reference draws per complex (default 100).
    --mode:       TARP distance — "centroid" (default, fast) or "rmsd" (slow).
    --n-workers:  parallel TARP workers (default 1).
    --n-bootstrap: ECP bootstrap replicates (default 200).
    --output:     output .npz path (default: <results_dir>/mira_tarp.npz).
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SIGMADOCK_SRC = _HERE.parent / "src"
_DIFFDOCK_DIR  = _HERE.parents[1] / "DiffDock"

sys.path.insert(0, str(_SIGMADOCK_SRC))
sys.path.insert(0, str(_DIFFDOCK_DIR))

from sigmadock.chem.statistics import get_mol_from_coords

from utils.mira_eval import _mira_one_complex, mira_null
from utils.tarp_eval import (
    load_protein_ca_coords,
    prepare_reference_template,
    compute_tarp_fractions_one_complex,
    ecp_from_fractions,
    bootstrap_ecp,
    plot_ecp,
)


# ── data loading ──────────────────────────────────────────────────────────────

def load_sigmadock_poses(model_dir: Path) -> dict:
    """Load all predicted poses from seed_*/predictions.pt files.

    Args:
        model_dir: path containing seed_* subdirectories with predictions.pt.

    Returns:
        Dict mapping complex_id → (lig_ref_mol, List[np.ndarray shape (N,3)]).
        crystal coordinates are in lig_ref.GetConformer().GetPositions().
    """
    seed_dirs = sorted(
        [p for p in model_dir.glob("seed_*") if (p / "predictions.pt").exists()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_*/predictions.pt in {model_dir}")

    poses: dict[str, list] = {}
    ref_mols: dict[str, object] = {}

    for seed_dir in seed_dirs:
        pt = torch.load(seed_dir / "predictions.pt", weights_only=False)
        for complex_id, samples in pt["results"].items():
            sample = samples[0]
            lig_ref = sample["lig_ref"]
            x0_hat  = sample["x0_hat"]      # (N_atoms, 3) CPU tensor

            pred_mol = get_mol_from_coords(x0_hat, lig_ref)
            coords   = pred_mol.GetConformer().GetPositions()  # (N, 3) float64

            if complex_id not in poses:
                poses[complex_id]    = []
                ref_mols[complex_id] = lig_ref

            poses[complex_id].append(coords)

    return {cid: (ref_mols[cid], poses[cid]) for cid in sorted(poses)}


# ── MIRA ──────────────────────────────────────────────────────────────────────

def run_mira(complex_data: dict, num_runs: int = 100, metric: str = "euclidean") -> tuple:
    """Compute per-complex MIRA scores.

    Args:
        complex_data: output of load_sigmadock_poses().
        num_runs: Monte Carlo center draws per complex.
        metric: "euclidean" (default) or "rmsd".

    Returns:
        (names, scores): numpy arrays of shape (n_valid,).
    """
    from mira_score import get_device
    device = get_device()

    names, scores = [], []
    n = len(complex_data)

    for i, (cid, (lig_ref, sample_coords)) in enumerate(complex_data.items()):
        if i % 20 == 0:
            print(f"  MIRA [{i}/{n}] {cid} ...", flush=True)

        crystal = lig_ref.GetConformer().GetPositions()  # (N, 3)

        score = _mira_one_complex(
            crystal, sample_coords,
            num_runs=num_runs, device=device, metric=metric,
        )
        if not np.isnan(score):
            names.append(cid)
            scores.append(score)

    S = len(next(iter(complex_data.values()))[1])
    print(f"MIRA done: {len(scores)}/{n} complexes.")
    print(f"  Mean MIRA = {np.mean(scores):.4f}   "
          f"(reference for S={S}: {mira_null(S):.4f})")
    return np.array(names), np.array(scores, dtype=float)


# ── TARP ──────────────────────────────────────────────────────────────────────

def run_tarp(
    complex_data: dict,
    data_dir: str,
    K: int = 100,
    mode: str = "centroid",
    seed: int = 42,
    n_workers: int = 1,
) -> tuple:
    """Compute TARP coverage fractions for all complexes.

    Args:
        complex_data: output of load_sigmadock_poses().
        data_dir: PoseBusters dataset root — used to load protein Cα coords.
        K: random reference draws per complex.
        mode: "centroid" (fast) or "rmsd" (slow, symmetry-corrected).
        seed: master random seed; per-complex seeds derived via SeedSequence.
        n_workers: parallel workers (TARP is embarrassingly parallel).

    Returns:
        (names, f_matrix): names array and float matrix (n_valid, K).
    """
    complex_ids = list(complex_data)
    n = len(complex_ids)
    child_seeds = np.random.SeedSequence(seed).spawn(n)

    names_out, rows = [], []
    skipped = 0

    def _process_one(i: int, cid: str):
        nonlocal skipped
        lig_ref, sample_coords = complex_data[cid]
        crystal = lig_ref.GetConformer().GetPositions()

        pdb_id = cid.split("::")[0]
        try:
            ca_coords = load_protein_ca_coords(pdb_id, data_dir)
        except Exception as exc:
            print(f"    Skipping {cid} (protein load error): {exc}", flush=True)
            skipped += 1
            return

        template_mol, rot_bonds = prepare_reference_template(lig_ref)
        rng = np.random.default_rng(child_seeds[i])

        fracs = compute_tarp_fractions_one_complex(
            lig_ref, crystal, template_mol, rot_bonds,
            sample_coords, ca_coords,
            K=K, rng=rng, mode=mode,
        )
        if len(fracs) > 0:
            names_out.append(cid)
            rows.append(fracs[:K])

    if n_workers > 1:
        from multiprocessing import Pool
        # Multiprocessing requires picklable args; run serially for simplicity
        # if data is already loaded in-memory. Fall back to serial.
        print("Note: n_workers>1 not supported with in-memory data; running serially.")

    for i, cid in enumerate(complex_ids):
        if i % 20 == 0:
            print(f"  TARP [{i}/{n}] {cid} ...", flush=True)
        _process_one(i, cid)

    max_k = max((len(r) for r in rows), default=0)
    if max_k == 0:
        return np.array(names_out), np.empty((0, K))

    f_matrix = np.full((len(rows), max_k), np.nan)
    for j, r in enumerate(rows):
        f_matrix[j, :len(r)] = r

    print(f"TARP done: {len(rows)}/{n} complexes, {skipped} skipped.")
    return np.array(names_out), f_matrix


# ── main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    model_dir = Path(args.results_dir)
    out_path  = Path(args.output) if args.output else model_dir / "mira_tarp.npz"
    plot_path = out_path.with_suffix(".png")

    print(f"Loading poses from {model_dir} ...", flush=True)
    complex_data = load_sigmadock_poses(model_dir)
    print(f"Loaded {len(complex_data)} complexes, "
          f"{len(next(iter(complex_data.values()))[1])} seeds each.\n")

    # ── MIRA ──────────────────────────────────────────────────────────────────
    print("=== MIRA ===")
    mira_names, mira_scores = run_mira(
        complex_data, num_runs=args.num_runs, metric=args.metric
    )

    # ── TARP ──────────────────────────────────────────────────────────────────
    tarp_names = np.array([], dtype=object)
    f_matrix   = np.empty((0, args.K))
    ecp        = np.array([])
    alpha      = np.array([])
    boot_ecps  = np.array([])

    if args.data_dir:
        print("\n=== TARP ===")
        tarp_names, f_matrix = run_tarp(
            complex_data,
            data_dir=args.data_dir,
            K=args.K,
            mode=args.mode,
            n_workers=args.n_workers,
        )
        if f_matrix.shape[0] > 0:
            ecp, alpha   = ecp_from_fractions(f_matrix)
            boot_ecps    = bootstrap_ecp(f_matrix, n_bootstrap=args.n_bootstrap)
            auc = np.trapz(ecp, alpha)
            print(f"\n  TARP AUC = {auc:.4f}  (perfect = 0.5000)")
            print(f"  ECP at α=0.5: {np.interp(0.5, alpha, ecp):.3f}")

            # ── plot ──────────────────────────────────────────────────────────
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            plot_ecp(ecp, alpha, ax=ax,
                     label=f"SigmaDock (N={len(tarp_names)})",
                     bootstrap_ecps=boot_ecps)
            ax.set_title(f"TARP ECP — SigmaDock PoseBusters ({args.mode} mode)")
            fig.tight_layout()
            fig.savefig(str(plot_path), dpi=150)
            print(f"  ECP plot → {plot_path}")
    else:
        print("\n(TARP skipped — pass --data-dir to enable)")

    # ── save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        mira_names=mira_names,
        mira_scores=mira_scores,
        tarp_names=tarp_names,
        tarp_f_matrix=f_matrix,
        tarp_ecp=ecp,
        tarp_alpha=alpha,
        tarp_boot_ecps=boot_ecps,
    )
    print(f"\nResults → {out_path}")

    # ── summary ───────────────────────────────────────────────────────────────
    S = len(next(iter(complex_data.values()))[1])
    print("\n" + "=" * 50)
    print(f"  Complexes evaluated : {len(complex_data)}")
    print(f"  Seeds per complex   : {S}")
    print(f"  MIRA mean           : {mira_scores.mean():.4f}")
    print(f"  MIRA reference (S={S}): {mira_null(S):.4f}")
    delta = mira_scores.mean() - mira_null(S)
    direction = "over-dispersed" if delta > 0 else "mode-collapsed"
    print(f"  MIRA deviation      : {delta:+.4f}  ({direction})")
    if f_matrix.shape[0] > 0:
        print(f"  TARP AUC            : {np.trapz(ecp, alpha):.4f}  (perfect = 0.5000)")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MIRA + TARP evaluation for SigmaDock PoseBusters results."
    )
    parser.add_argument("results_dir",
                        help="Model dir containing seed_*/predictions.pt")
    parser.add_argument("--data-dir", default=None,
                        help="PoseBusters dataset root (required for TARP)")
    parser.add_argument("--K", type=int, default=100,
                        help="TARP reference draws per complex (default 100)")
    parser.add_argument("--mode", choices=["centroid", "rmsd"], default="centroid",
                        help="TARP distance mode: centroid (fast) or rmsd (slow)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel TARP workers")
    parser.add_argument("--n-bootstrap", type=int, default=200,
                        help="ECP bootstrap replicates (default 200)")
    parser.add_argument("--num-runs", type=int, default=100,
                        help="MIRA Monte Carlo draws per complex (default 100)")
    parser.add_argument("--metric", choices=["euclidean", "rmsd"], default="euclidean",
                        help="MIRA distance metric (default euclidean)")
    parser.add_argument("--output", default=None,
                        help="Output .npz path (default: <results_dir>/mira_tarp.npz)")
    main(parser.parse_args())
