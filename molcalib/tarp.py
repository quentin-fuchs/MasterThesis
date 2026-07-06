"""TARP (Tests of Accuracy with Random Points) calibration diagnostic.

Implements the TARP coverage test (Lemos & Coogan et al. 2023,
arXiv:2302.03026) for evaluating whether a docking model's predicted pose
distribution is well-calibrated.

For each test complex, TARP asks: given a random reference pose c drawn from
the model's prior, what fraction of predicted samples fall closer to c than
the crystal pose does? If this fraction, averaged over complexes and references,
matches α for all α ∈ [0,1], the posterior is perfectly calibrated.

Typical usage
-------------
>>> from molcalib import prepare_reference_template, generate_reference_coords
>>> from molcalib import tarp_fractions, ecp_from_fractions, plot_ecp
>>> template_mol, rot_bonds = prepare_reference_template(crystal_mol)
>>> fracs = tarp_fractions(
...     crystal_mol, crystal_coords, template_mol, rot_bonds,
...     sample_coords, ca_coords, K=20, rng=rng, mode="rmsd"
... )
>>> f_matrix  # shape (n_complexes, K)
>>> ecp, alpha = ecp_from_fractions(f_matrix)
>>> plot_ecp(ecp, alpha)
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt

from molcalib.distances import compute_rmsd_symmetry, compute_centroid_distance
from molcalib.prior import generate_reference_coords
from molcalib.style import FS

warnings.filterwarnings("ignore")


def tarp_fractions(
    crystal_mol,
    crystal_coords,
    template_mol,
    rot_bonds,
    sample_coords,
    ca_coords,
    K,
    rng,
    mode="rmsd",
):
    """Compute K TARP coverage fractions for a single complex.

    For each of K random reference poses c:
        r   = distance(crystal, c)
        d_j = distance(sample_j, c)   for each predicted sample
        f_k = fraction of d_j < r

    Under perfect calibration, f is uniform on [0,1] and the ECP matches the
    diagonal (Theorem 3, Lemos & Coogan 2023).

    Args:
        crystal_mol: RDKit Mol (heavy atoms) — defines the atom graph for RMSD.
        crystal_coords: numpy array (N_atoms, 3) — crystal pose.
        template_mol: RDKit Mol with ETKDG conformer from prepare_reference_template.
        rot_bonds: list of (n0, a, b, n1) from prepare_reference_template.
        sample_coords: list of numpy arrays (N_atoms, 3) — predicted poses.
        ca_coords: numpy array (N_res, 3) — protein Cα coordinates.
        K: number of random reference draws.
        rng: numpy Generator.
        mode: "rmsd" (symmetry-corrected, default) or "centroid".

    Returns:
        numpy array of shape (≤K,) with values in [0,1]. May be shorter than K
        if reference distances are non-finite.
    """
    if mode == "rmsd":
        def dist_fn(ref, queries):
            return compute_rmsd_symmetry(crystal_mol, ref, queries)
    else:
        def dist_fn(ref, queries):
            return compute_centroid_distance(ref, queries)

    fractions = []
    for _ in range(K):
        ref_coords = generate_reference_coords(template_mol, rot_bonds, ca_coords, rng)
        r = dist_fn(ref_coords, [crystal_coords])[0]
        if not np.isfinite(r):
            continue
        d_samples = dist_fn(ref_coords, sample_coords)
        finite_mask = np.isfinite(d_samples)
        if not finite_mask.any():
            continue
        fractions.append((d_samples[finite_mask] < r).mean())
    return np.array(fractions)


def ecp_from_fractions(f_matrix, n_bins=50):
    """Compute the Expected Coverage Probability (ECP) curve.

    ecp(α) = fraction of f values ≤ α. Under perfect calibration ecp(α) = α.

    Args:
        f_matrix: numpy array of shape (n_complexes, K) or flat 1-D array.
        n_bins: number of α values.

    Returns:
        (ecp, alpha): numpy arrays of shape (n_bins,).
    """
    f_flat = np.asarray(f_matrix).ravel()
    f_flat = f_flat[np.isfinite(f_flat)]
    alpha = np.linspace(0, 1, n_bins)
    ecp = np.array([(f_flat <= a).mean() for a in alpha])
    return ecp, alpha


def bootstrap_ecp(f_matrix, n_bins=50, n_bootstrap=500, rng=None):
    """Bootstrap confidence bands for the ECP by resampling complexes.

    Resamples rows of f_matrix (each row = one complex) with replacement to
    give correct uncertainty accounting for between-complex variability.

    Args:
        f_matrix: numpy array of shape (n_complexes, K).
        n_bins: number of α values.
        n_bootstrap: number of bootstrap replicates.
        rng: numpy Generator. If None, uses default_rng().

    Returns:
        numpy array of shape (n_bootstrap, n_bins).
    """
    if rng is None:
        rng = np.random.default_rng()
    f_matrix = np.asarray(f_matrix)
    n = len(f_matrix)
    boot_ecps = np.zeros((n_bootstrap, n_bins))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ecp, _ = ecp_from_fractions(f_matrix[idx], n_bins=n_bins)
        boot_ecps[b] = ecp
    return boot_ecps


def plot_ecp(ecp, alpha, ax=None, label=None, color=None, bootstrap_ecps=None,
             linestyle="solid"):
    """Plot an ECP curve against the perfect-calibration diagonal.

    Args:
        ecp: numpy array of ECP values (shape n_bins).
        alpha: numpy array of credibility levels (shape n_bins).
        ax: matplotlib Axes. If None, a new figure is created.
        label: legend label.
        color: line colour.
        bootstrap_ecps: optional (n_bootstrap, n_bins) array from bootstrap_ecp()
            for a 90% confidence band.
        linestyle: matplotlib linestyle string.

    Returns:
        matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    c = color or "C0"
    ax.plot(alpha, ecp, color=c, lw=2, label=label, linestyle=linestyle)
    if bootstrap_ecps is not None:
        lo = np.percentile(bootstrap_ecps, 5, axis=0)
        hi = np.percentile(bootstrap_ecps, 95, axis=0)
        ax.fill_between(alpha, lo, hi, color=c, alpha=0.2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Credibility level α", fontsize=FS["label"])
    ax.set_ylabel("Expected coverage probability", fontsize=FS["label"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    return ax
