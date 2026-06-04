"""Per-group (translation / rotation / torsion) TARP and MIRA evaluation for SigmaDock.

Adapts DiffDock's per-group evaluation utilities to SigmaDock's .pt output format.
Crystal and sample coordinates come from seed_*/predictions.pt; protein Cα coords
are loaded from the PoseBusters dataset directory.

Output layout (<results_dir>/group_eval/):
    complex_names.npy               (N,)      complex ID strings
    n_rot_bonds.npy                 (N,)      rotatable bond counts
    tarp_fractions_translation.npy  (N, K)    translation TARP fractions
    tarp_fractions_rotation.npy     (N, K)    rotation TARP fractions
    tarp_fractions_torsion.npy      (N, K)    torsion TARP fractions (NaN for rigid)
    mira_names_translation.npy      (N,)
    mira_scores_translation.npy     (N,)
    mira_names_rotation.npy         (N,)
    mira_scores_rotation.npy        (N,)
    mira_names_torsion.npy          (M,)      M ≤ N (rigid ligands excluded)
    mira_scores_torsion.npy         (M,)
    distances_translation.npy       (N, S)    L2 centroid distances (Å)
    distances_rotation.npy          (N, S)    geodesic Kabsch angles (rad)
    distances_torsion_rms.npy       (N, S)    RMS torsion angle differences (rad)
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parents[1] / "DiffDock"))

from sigmadock.chem.statistics import get_mol_from_coords
from utils.tarp_eval import load_protein_ca_coords
from utils.group_eval import (
    get_rotatable_bonds,
    _kabsch_rotation,
    _geodesic_angle,
    extract_torsion_angles,
    _spyrmsd_mol,
    _get_sym_permutations,
    _apply_permutation,
    compute_tarp_fractions_translation,
    compute_tarp_fractions_rotation,
    compute_tarp_fractions_torsion,
)
from utils.group_mira_eval import (
    mira_null,
    _mira_score_translation,
    _mira_score_rotation,
    _mira_score_torsion,
)


def load_sigmadock_poses(model_dir: Path) -> dict:
    """Load all predicted poses from seed_*/predictions.pt files.

    Args:
        model_dir: directory containing seed_* subdirectories with predictions.pt.

    Returns:
        Dict mapping complex_id → (lig_ref_mol, list_of_(N_atoms, 3)_arrays).
        Crystal coords are in lig_ref.GetConformer().GetPositions().
    """
    seed_dirs = sorted(
        [p for p in model_dir.glob("seed_*") if (p / "predictions.pt").exists()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_*/predictions.pt in {model_dir}")

    poses: dict = {}
    ref_mols: dict = {}

    for seed_dir in seed_dirs:
        pt = torch.load(seed_dir / "predictions.pt", weights_only=False)
        for complex_id, samples in pt["results"].items():
            sample  = samples[0]
            lig_ref = sample["lig_ref"]
            x0_hat  = sample["x0_hat"]
            pred_mol = get_mol_from_coords(x0_hat, lig_ref)
            coords   = pred_mol.GetConformer().GetPositions()

            if complex_id not in poses:
                poses[complex_id]    = []
                ref_mols[complex_id] = lig_ref
            poses[complex_id].append(coords)

    return {cid: (ref_mols[cid], poses[cid]) for cid in sorted(poses)}


def _process_complex(cid, lig_ref, sample_coords_list, data_dir, K, num_runs, rng):
    """Compute per-group TARP fractions, MIRA scores, and distances for one complex.

    Args:
        cid: complex ID string (format: <pdb_id>::<ligand_name>).
        lig_ref: RDKit Mol — crystal/reference ligand with a conformer.
        sample_coords_list: list of (N_atoms, 3) predicted coordinate arrays.
        data_dir: PoseBusters dataset root (for loading protein Cα coords).
        K: TARP reference draws per group.
        num_runs: MIRA Monte Carlo center draws per group.
        rng: numpy Generator.

    Returns:
        dict with keys:
            'translation', 'rotation', 'torsion'  — (K,) TARP fraction arrays
            'mira_translation', 'mira_rotation', 'mira_torsion'  — scalar floats
            'dist_translation', 'dist_rotation', 'dist_torsion_rms'  — (S,) arrays
            'n_rot_bonds'  — int
    """
    pdb_id         = cid.split("::")[0]
    crystal_coords = np.array(lig_ref.GetConformer().GetPositions())
    S              = len(sample_coords_list)

    ca_coords = load_protein_ca_coords(pdb_id, data_dir)
    rot_bonds = get_rotatable_bonds(lig_ref)
    n_bonds   = len(rot_bonds)

    # ── Symmetry-corrected preprocessing ─────────────────────────────────────
    crystal_c        = crystal_coords.mean(axis=0)
    crystal_centred  = crystal_coords - crystal_c
    crystal_torsions = extract_torsion_angles(lig_ref, crystal_coords, rot_bonds)

    spy_mol             = _spyrmsd_mol(lig_ref)
    sample_centred_list = [sc - sc.mean(axis=0) for sc in sample_coords_list]
    perms               = _get_sym_permutations(crystal_centred, sample_centred_list, spy_mol)

    sample_rotations   = []
    sample_torsions    = []
    sample_centroids   = []
    dist_translation   = np.empty(S)
    dist_rotation      = np.full(S, np.nan)
    dist_torsion_rms   = np.full(S, np.nan)

    for i, (sc, sc_c, perm) in enumerate(zip(sample_coords_list, sample_centred_list, perms)):
        idx1, idx2 = perm

        dist_translation[i] = float(np.linalg.norm(sc.mean(axis=0) - crystal_c))
        sample_centroids.append(sc.mean(axis=0))

        try:
            sc_c_reord = _apply_permutation(sc_c, idx1, idx2)
            R = _kabsch_rotation(crystal_centred, sc_c_reord)
            sample_rotations.append(R)
            dist_rotation[i] = _geodesic_angle(R)
        except Exception:
            sample_rotations.append(None)

        if n_bonds > 0:
            try:
                sc_reord = _apply_permutation(sc, idx1, idx2)
                sa = extract_torsion_angles(lig_ref, sc_reord, rot_bonds)
            except Exception:
                sa = extract_torsion_angles(lig_ref, sc, rot_bonds)
            sample_torsions.append(sa)
            diffs  = ((sa - crystal_torsions + np.pi) % (2 * np.pi)) - np.pi
            finite = np.isfinite(diffs)
            dist_torsion_rms[i] = float(np.sqrt(np.mean(diffs[finite] ** 2))) if finite.any() else np.nan
        else:
            sample_torsions.append(np.array([], dtype=float))

    # ── Per-group TARP fractions ──────────────────────────────────────────────
    fracs_tr  = compute_tarp_fractions_translation(
        crystal_coords, sample_coords_list, ca_coords, K, rng)
    fracs_rot = compute_tarp_fractions_rotation(sample_rotations, K, rng)
    fracs_tor = compute_tarp_fractions_torsion(crystal_torsions, sample_torsions, K, rng)

    # ── Per-group MIRA scores ─────────────────────────────────────────────────
    sample_centroids_arr = np.array(sample_centroids)
    mira_tr  = _mira_score_translation(crystal_c, sample_centroids_arr, ca_coords, num_runs, rng)
    mira_rot = _mira_score_rotation(sample_rotations, num_runs, rng)
    mira_tor = _mira_score_torsion(crystal_torsions, sample_torsions, num_runs, rng)

    def _pad(arr):
        row = np.full(K, np.nan)
        row[:min(len(arr), K)] = arr[:K]
        return row

    return {
        "translation":      _pad(fracs_tr),
        "rotation":         _pad(fracs_rot),
        "torsion":          _pad(fracs_tor),
        "mira_translation": mira_tr,
        "mira_rotation":    mira_rot,
        "mira_torsion":     mira_tor,
        "dist_translation": dist_translation,
        "dist_rotation":    dist_rotation,
        "dist_torsion_rms": dist_torsion_rms,
        "n_rot_bonds":      n_bonds,
    }


def main(args):
    model_dir = Path(args.results_dir)
    out_dir   = model_dir / "group_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading poses from {model_dir} ...", flush=True)
    complex_data = load_sigmadock_poses(model_dir)
    n = len(complex_data)
    S = max(len(v[1]) for v in complex_data.values())
    print(f"  {n} complexes, {S} seeds each.\n", flush=True)

    child_seeds = np.random.SeedSequence(args.seed).spawn(n)
    K, num_runs = args.K, args.num_runs

    names_out     = []
    nrot_out      = []
    tarp_rows     = {g: [] for g in ("translation", "rotation", "torsion")}
    mira_names    = {g: [] for g in ("translation", "rotation", "torsion")}
    mira_scores   = {g: [] for g in ("translation", "rotation", "torsion")}
    dist_tr_rows  = []
    dist_rot_rows = []
    dist_tor_rows = []
    skipped = 0

    for i, (cid, (lig_ref, sample_coords_list)) in enumerate(complex_data.items()):
        if i % 20 == 0:
            print(f"  [{i}/{n}] {cid} ...", flush=True)
        rng = np.random.default_rng(child_seeds[i])
        try:
            res = _process_complex(
                cid, lig_ref, sample_coords_list, args.data_dir, K, num_runs, rng)
        except Exception as exc:
            print(f"    Skipping {cid}: {exc}", flush=True)
            skipped += 1
            continue

        names_out.append(cid)
        nrot_out.append(res["n_rot_bonds"])
        for g in ("translation", "rotation", "torsion"):
            tarp_rows[g].append(res[g])
        for g, key in [("translation", "mira_translation"),
                       ("rotation",    "mira_rotation"),
                       ("torsion",     "mira_torsion")]:
            score = res[key]
            if np.isfinite(score):
                mira_names[g].append(cid)
                mira_scores[g].append(score)
        dist_tr_rows.append(res["dist_translation"])
        dist_rot_rows.append(res["dist_rotation"])
        dist_tor_rows.append(res["dist_torsion_rms"])

    print(f"\nDone: {len(names_out)} processed, {skipped} skipped.", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    N     = len(names_out)
    S_max = max((len(r) for r in dist_tr_rows), default=0)

    np.save(f"{out_dir}/complex_names.npy", np.array(names_out))
    np.save(f"{out_dir}/n_rot_bonds.npy",   np.array(nrot_out, dtype=int))

    for g in ("translation", "rotation", "torsion"):
        np.save(f"{out_dir}/tarp_fractions_{g}.npy",
                np.vstack(tarp_rows[g]) if tarp_rows[g] else np.empty((0, K)))
        np.save(f"{out_dir}/mira_names_{g}.npy",  np.array(mira_names[g]))
        np.save(f"{out_dir}/mira_scores_{g}.npy", np.array(mira_scores[g], dtype=float))

    def _pad_dist(rows):
        M = np.full((N, S_max), np.nan)
        for k, r in enumerate(rows):
            M[k, :len(r)] = r
        return M

    np.save(f"{out_dir}/distances_translation.npy",  _pad_dist(dist_tr_rows))
    np.save(f"{out_dir}/distances_rotation.npy",      _pad_dist(dist_rot_rows))
    np.save(f"{out_dir}/distances_torsion_rms.npy",  _pad_dist(dist_tor_rows))

    null = mira_null(S)
    print(f"\nGroup MIRA summary  (null = {null:.4f}):")
    for g in ("translation", "rotation", "torsion"):
        sc = np.array(mira_scores[g])
        if len(sc):
            print(f"  {g:12s}: n={len(sc):3d}  mean={sc.mean():.4f}  "
                  f"dev={sc.mean() - null:+.4f}")
    print(f"\nResults saved → {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Per-group TARP & MIRA evaluation for SigmaDock.")
    ap.add_argument("results_dir",
                    help="Model dir containing seed_*/predictions.pt")
    ap.add_argument("--data-dir",
                    default="/home/qf226/rds/hpc-work/data/posebusters_benchmark_set",
                    help="PoseBusters dataset root (for protein Cα coords)")
    ap.add_argument("--K",        type=int, default=100,
                    help="TARP reference draws per complex (default 100)")
    ap.add_argument("--num-runs", type=int, default=100,
                    help="MIRA Monte Carlo draws per complex (default 100)")
    ap.add_argument("--seed",     type=int, default=42)
    main(ap.parse_args())
