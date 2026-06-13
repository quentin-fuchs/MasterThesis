"""
MIRA score evaluation for DiffDock.

Applies the MIRA calibration score (Sharief et al. 2026, arXiv:2605.02014) to
DiffDock's predicted pose distributions. MIRA draws random centers in the pose
coordinate space and checks whether the crystal pose falls inside the same ball
as the predicted samples. Under perfect calibration the score converges to
~0.683 for S=40 samples (derived from the Beta(2,1) null distribution).

Score > 0.683: over-dispersed (samples spread too wide)
Score < 0.683: mode-collapsed (samples clustered too tightly)

Distance metric: Euclidean distance in flattened all-atom coordinate space
(q = 3 × n_heavy_atoms), equivalent to scaled unaligned RMSD. Poses are
already in the protein frame so no alignment is needed.

Because each complex has a different number of heavy atoms (different q), MIRA
is run per-complex with T=1 and aggregated externally. Group-level uncertainty
is estimated by bootstrapping over per-complex scores.
"""

import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# Lazy import to avoid circular dependency at module load time.
def _get_compute_rmsd_symmetry():
    from utils.tarp_eval import compute_rmsd_symmetry
    return compute_rmsd_symmetry

# Reference MIRA score under perfect calibration for S samples.
# E[calib] = (2/3) / ((S-1+1)/(S-1+2)) = (2/3) * (S+1)/S
# For S=40: (2/3) * 41/40 ≈ 0.6833
def mira_null(S: int) -> float:
    return (2 / 3) * (S + 1) / S


def _load_poses(pdb_id: str, results_index: dict, data_dir: str):
    """Load crystal and predicted heavy-atom coordinates for one complex.

    Args:
        pdb_id: PDB identifier.
        results_index: dict from build_results_index() in tarp_eval.
        data_dir: root directory containing PDBBind_processed/.

    Returns:
        (crystal_coords, sample_coords_list) where crystal_coords is (n_atoms, 3)
        and sample_coords_list is a list of (n_atoms, 3) arrays.

    Raises:
        FileNotFoundError, ValueError: if files are missing or unparseable.
    """
    from utils.tarp_eval import load_crystal_coords, load_sample_coords
    _, all_crystal_coords = load_crystal_coords(pdb_id, data_dir)
    crystal_coords = all_crystal_coords[0]
    sample_coords = load_sample_coords(pdb_id, results_index)
    return crystal_coords, sample_coords


def _mira_one_complex_symrmsd(
    crystal_mol,
    crystal_coords: np.ndarray,
    sample_coords: list,
    num_runs: int,
    rng,
    timeout: int = 4,
) -> float:
    """MIRA score using symmetry-corrected pairwise RMSD (spyrmsd).

    Replaces the flat-Euclidean distance used by the `mira_score` library with
    symmetry-corrected heavy-atom RMSD. Bypasses `mira_score.mira` entirely and
    re-implements the random-radius MIRA estimator directly, following the same
    pattern as `group_mira_eval._mira_score_translation`.

    Precomputes the full (1+S) × (1+S) pairwise symRMSD matrix D (index 0 =
    crystal, 1..S = samples), then for each of `num_runs` iterations:
      - draws a random posterior sample as the center,
      - draws a distinct reference sample to set the ball radius,
      - checks what fraction of remaining samples and the crystal fall inside.

    The null reference `mira_null(S)` applies because under perfect calibration
    (crystal exchangeable with samples) the rank of d(center, crystal) among
    d(center, sample_j) is uniform on {1..S} for any distance metric.

    Args:
        crystal_mol: RDKit Mol (heavy atoms) defining the molecular graph for
            the symRMSD atom-permutation search.
        crystal_coords: (N_atoms, 3) crystal pose coordinates.
        sample_coords: list of (N_atoms, 3) predicted pose coordinates.
        num_runs: number of Monte Carlo center draws.
        rng: numpy Generator.
        timeout: per-call timeout in seconds passed to `compute_rmsd_symmetry`.
            High-symmetry molecules that exceed this limit return NaN.

    Returns:
        MIRA score (float), or NaN if fewer than 2 samples or all runs fail.
    """
    S = len(sample_coords)
    if S < 2:
        return float("nan")

    compute_rmsd_symmetry = _get_compute_rmsd_symmetry()
    all_coords = [crystal_coords] + list(sample_coords)  # length 1 + S

    # Build symmetric (1+S, 1+S) pairwise symRMSD matrix.
    D = np.full((1 + S, 1 + S), np.nan)
    np.fill_diagonal(D, 0.0)
    for i in range(1 + S):
        targets = all_coords[i + 1:]
        if not targets:
            continue
        row = compute_rmsd_symmetry(crystal_mol, all_coords[i], targets, timeout=timeout)
        for offset, val in enumerate(row):
            j = i + 1 + offset
            D[i, j] = val
            D[j, i] = val

    N = S - 1
    scores = []
    sample_indices = np.arange(1, 1 + S)  # indices 1..S in D

    for _ in range(num_runs):
        # Random center: a posterior sample
        i = int(rng.choice(sample_indices))
        # Random reference: another posterior sample
        other_samples = sample_indices[sample_indices != i]
        j = int(rng.choice(other_samples))

        r = D[i, j]
        if not np.isfinite(r):
            continue

        d_crystal = D[0, i]

        # Distances from center to all remaining samples (exclude i and j)
        rest = sample_indices[(sample_indices != i) & (sample_indices != j)]
        d_rest = D[i, rest]
        finite_mask = np.isfinite(d_rest)
        d_finite = d_rest[finite_mask]
        N_f = len(d_finite)

        counts = int((d_finite < r).sum())
        k_flag = float(np.isfinite(d_crystal) and d_crystal <= r)
        prob_in  = (counts + 1) / (N_f + 2)
        prob_out = (N_f - counts + 1) / (N_f + 2)
        mv = (N_f + 1) / (N_f + 2)
        scores.append((prob_in * k_flag + prob_out * (1 - k_flag)) / mv)

    return float(np.nanmean(scores)) if scores else float("nan")


def _mira_one_complex(crystal: np.ndarray, samples: list, num_runs: int,
                      device: torch.device,
                      metric: str = "euclidean",
                      mol=None,
                      rng=None) -> float:
    """Run MIRA for a single complex (T=1).

    Normalises all poses jointly (subtract joint mean, divide by joint std)
    so that the U[0,1]^q random centers are meaningful regardless of the
    absolute coordinate frame.

    Args:
        crystal: (n_atoms, 3) crystal pose.
        samples: list of (n_atoms, 3) predicted poses.
        num_runs: number of Monte Carlo center draws.
        device: torch device (ignored for metric="symrmsd").
        metric: distance metric.
            "euclidean" — Euclidean distance in flat q-space, equivalent to
                          scaled RMSD (√n_atoms × RMSD). Default.
            "rmsd"      — divides the normalised flat vectors by √n_atoms so
                          the Euclidean distance equals per-atom-average RMSD.
            "symrmsd"   — symmetry-corrected heavy-atom RMSD via spyrmsd.
                          Bypasses mira_score.mira; mol and rng must be provided.
        mol: RDKit Mol (heavy atoms). Required when metric="symrmsd".
        rng: numpy Generator. Required when metric="symrmsd".

    Returns:
        MIRA score (float), or NaN if fewer than 2 samples.
    """
    if metric == "symrmsd":
        if mol is None:
            raise ValueError("mol must be provided when metric='symrmsd'")
        if rng is None:
            rng = np.random.default_rng()
        return _mira_one_complex_symrmsd(mol, crystal, samples, num_runs, rng)

    from mira_score import mira

    if metric not in ("euclidean", "rmsd"):
        raise ValueError(f"metric must be 'euclidean', 'rmsd', or 'symrmsd', got {metric!r}")

    if len(samples) < 2:
        return float("nan")

    n_atoms = crystal.shape[0]
    S = len(samples)

    # Stack and flatten: (1+S, q)
    all_poses = np.stack([crystal] + samples, axis=0).reshape(1 + S, -1)

    # Joint normalisation
    mu = all_poses.mean(axis=0)
    sigma = all_poses.std() + 1e-8
    all_poses = (all_poses - mu) / sigma

    # Scale by 1/√n_atoms so Euclidean distance equals per-atom-average RMSD.
    if metric == "rmsd":
        all_poses = all_poses / np.sqrt(n_atoms)

    truth_t = torch.tensor(all_poses[:1], dtype=torch.float32, device=device)        # (1, q)
    post_t  = torch.tensor(all_poses[1:], dtype=torch.float32, device=device)        # (S, q)
    post_t  = post_t.unsqueeze(0).unsqueeze(0)                                       # (1, 1, S, q)

    score, _ = mira(truth_t, post_t, num_runs=num_runs, norm=False,
                    disable_tqdm=True, device=device)
    return float(score[0].cpu())


def _mira_symrmsd_worker(args):
    """Multiprocessing worker for symRMSD MIRA evaluation of one complex.

    Top-level function required for pickling by multiprocessing.Pool.

    Args:
        args: tuple of (pdb_id, results_index, data_dir, num_runs, seed).

    Returns:
        (pdb_id, score_or_nan, error_str_or_None)
    """
    pdb_id, results_index, data_dir, num_runs, seed = args
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from utils.tarp_eval import load_crystal_coords, load_sample_coords
        crystal_mol, all_crystal = load_crystal_coords(pdb_id, data_dir)
        crystal_coords = all_crystal[0]
        samples = load_sample_coords(pdb_id, results_index)
    except Exception as exc:
        return pdb_id, float("nan"), f"load error: {exc}"
    if len(samples) < 2:
        return pdb_id, float("nan"), "too few samples"
    try:
        rng = np.random.default_rng(seed)
        score = _mira_one_complex_symrmsd(crystal_mol, crystal_coords, samples,
                                          num_runs, rng)
        return pdb_id, score, None
    except Exception as exc:
        return pdb_id, float("nan"), f"compute error: {exc}"


def compute_mira_scores(
    complex_names,
    results_index: dict,
    data_dir: str,
    num_runs: int = 100,
    verbose: bool = True,
    device: torch.device = None,
    metric: str = "euclidean",
    seed: int = 42,
    n_workers: int = 1,
) -> tuple:
    """Compute per-complex MIRA scores over the full test set.

    Runs MIRA independently for each complex (T=1 per call) and aggregates
    the resulting scalars. Group-level statistics should be computed by
    bootstrapping over the returned per-complex scores.

    Args:
        complex_names: iterable of PDB ID strings.
        results_index: dict from build_results_index().
        data_dir: root directory containing PDBBind_processed/.
        num_runs: Monte Carlo runs per complex (controls center-placement
            variance; 100 gives stable per-complex scores).
        verbose: print progress every 20 complexes.
        device: torch device; auto-detected if None (ignored for symrmsd).
        metric: "euclidean" (default, scaled RMSD), "rmsd" (per-atom RMSD),
            or "symrmsd" (symmetry-corrected RMSD via spyrmsd).
            See _mira_one_complex for details.
        seed: master random seed used for per-complex rngs when
            metric="symrmsd". Ignored for other metrics.
        n_workers: number of parallel worker processes. Only used when
            metric="symrmsd"; ignored otherwise (mira_score.mira is not
            fork-safe). 1 = serial.

    Returns:
        (names_out, scores): two numpy arrays of length n_valid. names_out
        contains the PDB IDs of successfully evaluated complexes; scores
        contains the corresponding MIRA scores.
    """
    from utils.tarp_eval import load_crystal_coords as _load_crystal_mol

    use_symrmsd = metric == "symrmsd"
    if not use_symrmsd:
        from mira_score import get_device
        if device is None:
            device = get_device()

    complex_names = list(complex_names)
    n = len(complex_names)
    child_seeds = np.random.SeedSequence(seed).spawn(n) if use_symrmsd else None
    names_out, scores = [], []
    skipped = 0

    if use_symrmsd and n_workers > 1:
        from multiprocessing import Pool
        work = [
            (pdb_id, results_index, data_dir, num_runs, child_seeds[i])
            for i, pdb_id in enumerate(complex_names)
        ]
        n_done = 0
        with Pool(processes=n_workers) as pool:
            for pdb_id, score, err in pool.imap_unordered(_mira_symrmsd_worker, work):
                if verbose and n_done % 20 == 0:
                    print(f"  [{n_done}/{n}] {pdb_id} ...", flush=True)
                n_done += 1
                if err is not None:
                    if verbose:
                        print(f"    Skipping {pdb_id}: {err}", flush=True)
                    skipped += 1
                elif np.isfinite(score):
                    names_out.append(pdb_id)
                    scores.append(score)
                else:
                    skipped += 1
    else:
        for i, pdb_id in enumerate(complex_names):
            if verbose and i % 20 == 0:
                print(f"  [{i}/{n}] {pdb_id} ...", flush=True)
            try:
                crystal, samples = _load_poses(pdb_id, results_index, data_dir)
            except Exception as exc:
                if verbose:
                    print(f"    Skipping {pdb_id}: {exc}", flush=True)
                skipped += 1
                continue

            if use_symrmsd:
                try:
                    crystal_mol, _ = _load_crystal_mol(pdb_id, data_dir)
                except Exception as exc:
                    if verbose:
                        print(f"    Skipping {pdb_id} (mol load): {exc}", flush=True)
                    skipped += 1
                    continue
                rng = np.random.default_rng(child_seeds[i])
                score = _mira_one_complex(crystal, samples, num_runs=num_runs,
                                          device=device, metric=metric,
                                          mol=crystal_mol, rng=rng)
            else:
                score = _mira_one_complex(crystal, samples, num_runs=num_runs,
                                          device=device, metric=metric)

            if not np.isnan(score):
                names_out.append(pdb_id)
                scores.append(score)
            else:
                skipped += 1

    if verbose:
        S_typical = 40
        print(f"Done. {len(scores)} complexes evaluated, {skipped} skipped.")
        print(f"Reference (perfect calibration, S={S_typical}): "
              f"{mira_null(S_typical):.4f}")

    return np.array(names_out), np.array(scores, dtype=float)


def _rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Plain unaligned RMSD between two (N, 3) coordinate arrays."""
    return float(np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1))))


def compute_rmsd_accuracy(
    complex_names,
    results_index: dict,
    data_dir: str,
    thresholds: tuple = (2.0, 5.0),
    verbose: bool = True,
) -> tuple:
    """Compute per-complex RMSD accuracy metrics over the full test set.

    Uses plain (unaligned, unsymmetry-corrected) RMSD, consistent with the
    Euclidean distance MIRA uses internally. Atom ordering is preserved from
    the source SDF so no symmetry correction is needed.

    Args:
        complex_names: iterable of PDB ID strings.
        results_index: dict from build_results_index().
        data_dir: root directory containing PDBBind_processed/.
        thresholds: RMSD thresholds in Angstroms (default: 2.0 and 5.0).
        verbose: print progress every 20 complexes.

    Returns:
        (names_out, min_rmsds, fracs): three arrays of length n_valid.
          names_out: (n_valid,) PDB IDs of successfully evaluated complexes.
          min_rmsds: (n_valid,) minimum RMSD over all samples per complex.
          fracs: (n_valid, len(thresholds)) fraction of samples within each
            threshold, in the same order as thresholds.
    """
    complex_names = list(complex_names)
    n = len(complex_names)
    names_out, min_rmsds, fracs = [], [], []
    skipped = 0

    for i, pdb_id in enumerate(complex_names):
        if verbose and i % 20 == 0:
            print(f"  [{i}/{n}] {pdb_id} ...", flush=True)
        try:
            crystal, samples = _load_poses(pdb_id, results_index, data_dir)
        except Exception as exc:
            if verbose:
                print(f"    Skipping {pdb_id}: {exc}", flush=True)
            skipped += 1
            continue

        if len(samples) == 0:
            skipped += 1
            continue

        rmsds = np.array([_rmsd(s, crystal) for s in samples])
        names_out.append(pdb_id)
        min_rmsds.append(rmsds.min())
        fracs.append([(rmsds < t).mean() for t in thresholds])

    if verbose:
        print(f"Done. {len(names_out)} complexes evaluated, {skipped} skipped.")
        if names_out:
            mr = np.array(min_rmsds)
            for j, t in enumerate(thresholds):
                f = np.array(fracs)[:, j]
                print(f"  Top-1 acc (<{t:.0f}Å): {(mr < t).mean():.3f} | "
                      f"Any-sample acc (<{t:.0f}Å): {f.mean():.3f}")

    return (
        np.array(names_out),
        np.array(min_rmsds, dtype=float),
        np.array(fracs, dtype=float),
    )


def bootstrap_mira_groups(
    scores: np.ndarray,
    group_labels: np.ndarray,
    group_names: list,
    n_bootstrap: int = 500,
    rng=None,
) -> dict:
    """Bootstrap per-group MIRA mean and 90% CI by resampling complexes.

    Args:
        scores: (n_complexes,) array of per-complex MIRA scores.
        group_labels: (n_complexes,) array of group name strings.
        group_names: ordered list of group names.
        n_bootstrap: number of bootstrap replicates.
        rng: numpy Generator.

    Returns:
        dict mapping group_name -> {'n', 'mean', 'lo', 'hi', 'boot_means'}
    """
    if rng is None:
        rng = np.random.default_rng(42)

    results = {}
    for grp in group_names:
        mask = group_labels == grp
        g_scores = scores[mask]
        n = len(g_scores)
        if n == 0:
            continue
        boot = np.array([
            rng.choice(g_scores, size=n, replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        results[grp] = {
            "n":          n,
            "mean":       g_scores.mean(),
            "lo":         np.percentile(boot, 5),
            "hi":         np.percentile(boot, 95),
            "boot_means": boot,
        }
    return results
