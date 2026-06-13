"""MIRA (Model-Independent Random-radius Assessment) calibration score.

Implements the MIRA score (Sharief et al. 2026, arXiv:2605.02014) for
evaluating whether a docking model's predicted pose distribution is
well-calibrated.

For each complex, MIRA draws random center poses c from an arbitrary
distribution p(c) and checks whether the crystal pose falls inside the same
ball as the predicted samples. Under perfect calibration the score converges
to mira_null(S) ≈ 0.683 for S=40 samples.

Score > null: over-dispersed (samples spread too wide)
Score < null: mode-collapsed (samples clustered too tightly)

Distance metric: symmetry-corrected heavy-atom RMSD via spyrmsd.
Center distribution p(c): same prior as TARP (generate_reference_coords),
following Sharief et al.'s recommendation to use x*-dependent centers so
MIRA is sensitive to uninformative posteriors.

Typical usage
-------------
>>> from molcalib import mira_score, mira_null, prepare_reference_template
>>> template_mol, rot_bonds = prepare_reference_template(crystal_mol)
>>> score = mira_score(
...     crystal_mol, crystal_coords, sample_coords,
...     template_mol, rot_bonds, ca_coords,
...     num_runs=20, rng=rng,
... )
>>> print(f"MIRA = {score:.4f}  (null = {mira_null(len(sample_coords)):.4f})")
"""

import warnings

import numpy as np
import torch

from molcalib.distances import compute_rmsd_symmetry
from molcalib.prior import generate_reference_coords

warnings.filterwarnings("ignore")


def mira_null(S):
    """Expected MIRA score under perfect calibration for S posterior samples.

    Derived from E[calib] = 2/3 under calibration, normalised by
    max_val = (N+1)/(N+2) where N = S-1 (one sample used as yr, excluded).

    Args:
        S: number of posterior samples per complex.

    Returns:
        Float null reference score. For S=40: ≈ 0.6833.
    """
    return (2 / 3) * (S + 1) / S


def mira_score(
    crystal_mol,
    crystal_coords,
    sample_coords,
    template_mol,
    rot_bonds,
    ca_coords,
    num_runs=20,
    rng=None,
    timeout=4,
):
    """MIRA score for a single complex using symmetry-corrected RMSD.

    For each of num_runs regions:
      1. Draw c from DiffDock's prior (same as TARP).
      2. Compute d(c, y*) and d(c, y_j) for all S samples via symRMSD.
      3. Draw yr = one random sample; set ball radius r = d(c, yr).
      4. Count n = #{j ≠ yr : d(c, y_j) < r} (N = S-1 counted samples).
      5. calib = Laplace-smoothed probability of crystal's rank, normalised
         by max_val = S/(S+1).

    The null reference is mira_null(S) = (2/3)*(S+1)/S.

    Args:
        crystal_mol: RDKit Mol (heavy atoms) — atom graph for symRMSD.
        crystal_coords: numpy array (N_atoms, 3) — crystal pose.
        sample_coords: list of numpy arrays (N_atoms, 3) — predicted poses.
        template_mol: RDKit Mol with ETKDG conformer from prepare_reference_template.
        rot_bonds: list of (n0, a, b, n1) from prepare_reference_template.
        ca_coords: numpy array (N_res, 3) — protein Cα coordinates.
        num_runs: number of Monte Carlo center draws (regions).
        rng: numpy Generator. Created fresh if None.
        timeout: per-call timeout for compute_rmsd_symmetry.

    Returns:
        MIRA score (float), or NaN if fewer than 2 samples or all runs fail.
    """
    S = len(sample_coords)
    if S < 2:
        return float("nan")
    if rng is None:
        rng = np.random.default_rng()

    all_targets = [crystal_coords] + list(sample_coords)  # length 1+S

    scores = []
    for _ in range(num_runs):
        c_coords = generate_reference_coords(template_mol, rot_bonds, ca_coords, rng)
        all_dists = compute_rmsd_symmetry(crystal_mol, c_coords, all_targets, timeout=timeout)
        d_crystal = all_dists[0]
        d_samples = all_dists[1:]

        yr_idx = int(rng.integers(S))
        r = d_samples[yr_idx]
        if not np.isfinite(r):
            continue

        mask = np.ones(S, dtype=bool)
        mask[yr_idx] = False
        d_counted = d_samples[mask]
        finite_mask = np.isfinite(d_counted)
        d_finite = d_counted[finite_mask]
        N_f = len(d_finite)
        if N_f == 0:
            continue

        counts = int((d_finite < r).sum())
        k_flag = float(np.isfinite(d_crystal) and d_crystal <= r)
        prob_in  = (counts + 1) / (N_f + 2)
        prob_out = (N_f - counts + 1) / (N_f + 2)
        mv = (N_f + 1) / (N_f + 2)
        scores.append((prob_in * k_flag + prob_out * (1 - k_flag)) / mv)

    return float(np.nanmean(scores)) if scores else float("nan")


def _mira_euclidean(crystal, samples, num_runs, device, metric="euclidean"):
    """MIRA score using the mira_score library (Euclidean or scaled RMSD).

    Internal helper used when metric is 'euclidean' or 'rmsd'. Normalises
    coordinates jointly and calls mira_score.mira.

    Args:
        crystal: (n_atoms, 3) crystal pose.
        samples: list of (n_atoms, 3) predicted poses.
        num_runs: number of center draws.
        device: torch device.
        metric: "euclidean" or "rmsd".

    Returns:
        MIRA score (float), or NaN if fewer than 2 samples.
    """
    from mira_score import mira as _mira_lib

    if len(samples) < 2:
        return float("nan")

    n_atoms = crystal.shape[0]
    all_poses = np.stack([crystal] + samples, axis=0).reshape(len(samples) + 1, -1)
    mu = all_poses.mean(axis=0)
    sigma = all_poses.std() + 1e-8
    all_poses = (all_poses - mu) / sigma
    if metric == "rmsd":
        all_poses = all_poses / np.sqrt(n_atoms)

    truth_t = torch.tensor(all_poses[:1], dtype=torch.float32, device=device)
    post_t  = torch.tensor(all_poses[1:], dtype=torch.float32, device=device)
    post_t  = post_t.unsqueeze(0).unsqueeze(0)

    score, _ = _mira_lib(truth_t, post_t, num_runs=num_runs, norm=False,
                          disable_tqdm=True, device=device)
    return float(score[0].cpu())


def bootstrap_mira_groups(scores, group_labels, group_names, n_bootstrap=500, rng=None):
    """Bootstrap per-group MIRA mean and 90% CI by resampling complexes.

    Args:
        scores: (n_complexes,) array of per-complex MIRA scores.
        group_labels: (n_complexes,) array of group name strings.
        group_names: ordered list of group names.
        n_bootstrap: number of bootstrap replicates.
        rng: numpy Generator.

    Returns:
        dict mapping group_name → {'n', 'mean', 'lo', 'hi', 'boot_means'}.
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
