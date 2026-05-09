"""
Distribution analysis utilities for DiffDock inference results.

Provides tools to characterise the pose distribution output by DiffDock,
supporting the PQMass evaluation goal of testing whether the model's
distribution is well-captured.

Main entry points
-----------------
load_poses          -- load all ranked SDF poses from a results directory
compute_rmsd_matrix -- all-pairs symmetry-corrected RMSD (via spyrmsd)
cluster_poses       -- RMSD-based hierarchical clustering (2 Å cutoff)
saturation_analysis -- diversity vs. number of samples (mode coverage)
plot_rmsd_heatmap   -- clustered RMSD heatmap with dendrogram ordering
torsion_rose_plots  -- circular histograms per rotatable bond
view_poses_colored  -- py3Dmol viewer coloured by confidence or cluster
run_full_analysis   -- convenience wrapper returning all figures + data
"""

import os
import re
import glob
import json
from collections import namedtuple
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdMolTransforms

from utils.visualise import load_results_dir, mol_to_pdb_block

try:
    from spyrmsd import rmsd as _spyrmsd_rmsd
    _SPYRMSD_AVAILABLE = True
except ImportError:
    _SPYRMSD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

PoseRecord = namedtuple(
    "PoseRecord",
    ["rank", "confidence", "mol", "coords_heavy", "heavy_indices"],
)
"""
rank          : int   -- 1-based rank from DiffDock output
confidence    : float -- confidence score (higher = better; often negative)
mol           : rdkit.Chem.Mol
coords_heavy  : ndarray, shape (N_heavy, 3)
heavy_indices : list[int]  -- atom indices of heavy atoms in the full mol
"""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_poses(results_dir: str) -> List[PoseRecord]:
    """Load all ranked poses from a DiffDock output directory.

    Args:
        results_dir: Path to a single complex output directory,
            e.g. ``results/batch_job/6d08``.

    Returns:
        List of PoseRecord namedtuples sorted by rank (1, 2, …), each
        containing rank, confidence score, RDKit Mol, heavy-atom coordinate
        array (N_heavy × 3), and the list of heavy-atom indices.
    """
    info = load_results_dir(results_dir)
    poses = []
    for sdf_path in info["ligand_sdfs"]:
        fname = os.path.basename(sdf_path)
        rank_m = re.search(r"rank(\d+)", fname)
        rank = int(rank_m.group(1)) if rank_m else 0
        conf_m = re.search(r"confidence([+-]?\d+\.?\d*)", fname)
        confidence = float(conf_m.group(1)) if conf_m else float("nan")

        mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
        if mol is None:
            continue

        heavy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        coords = mol.GetConformer().GetPositions()[heavy_indices]
        poses.append(PoseRecord(rank, confidence, mol, coords, heavy_indices))

    poses.sort(key=lambda p: p.rank)

    # Deduplicate by rank: keep the entry with a valid confidence score when both exist
    seen_ranks: Dict[int, PoseRecord] = {}
    for pose in poses:
        if pose.rank not in seen_ranks:
            seen_ranks[pose.rank] = pose
        elif not np.isnan(pose.confidence):
            seen_ranks[pose.rank] = pose
    poses = [seen_ranks[r] for r in sorted(seen_ranks)]

    return poses


# ---------------------------------------------------------------------------
# Pairwise RMSD
# ---------------------------------------------------------------------------

def _mol_to_spyrmsd_args(pose: PoseRecord):
    """Extract spyrmsd inputs (coords, atomic numbers, adjacency) for one pose."""
    h = pose.heavy_indices
    atomnums = np.array([pose.mol.GetAtomWithIdx(i).GetAtomicNum() for i in h])
    adj = Chem.GetAdjacencyMatrix(pose.mol)[np.ix_(h, h)]
    return pose.coords_heavy, atomnums, adj


def compute_rmsd_matrix(poses: List[PoseRecord]) -> np.ndarray:
    """Compute the all-pairs symmetry-corrected RMSD matrix using spyrmsd.

    Uses ``spyrmsd.rmsd.symmrmsd`` which accounts for graph automorphisms
    (symmetry), making it correct for symmetric ligands. Heavy atoms only.

    Args:
        poses: List of PoseRecord as returned by ``load_poses``.

    Returns:
        Symmetric (N, N) float array of pairwise RMSDs in Angstroms.

    Raises:
        ImportError: if spyrmsd is not importable.
    """
    if not _SPYRMSD_AVAILABLE:
        raise ImportError(
            "spyrmsd not found. The bundled version lives in the DiffDock repo root; "
            "ensure the project root is on sys.path."
        )

    n = len(poses)
    matrix = np.zeros((n, n), dtype=float)
    c_ref, an_ref, adj_ref = _mol_to_spyrmsd_args(poses[0])

    for i in range(n):
        ci, ani, adji = _mol_to_spyrmsd_args(poses[i])
        others = [j for j in range(n) if j != i]
        coords_others = [poses[j].coords_heavy for j in others]
        if not coords_others:
            continue
        # symmrmsd(c_ref, c_list, atomnums_ref, atomnums_list, adj_ref, adj_list)
        # All poses are the same molecule, so atomnums and adj are shared.
        rmsds = _spyrmsd_rmsd.symmrmsd(
            ci, coords_others, ani, ani, adji, adji
        )
        for k, j in enumerate(others):
            matrix[i, j] = rmsds[k]

    return matrix


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_poses(
    rmsd_matrix: np.ndarray,
    cutoff: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster poses by RMSD using complete-linkage hierarchical clustering.

    Args:
        rmsd_matrix: Symmetric (N, N) RMSD array from ``compute_rmsd_matrix``.
        cutoff: Distance threshold in Angstroms for cluster assignment.
            2.0 Å is the standard cutoff used in docking benchmarks.

    Returns:
        Tuple of:
        - labels: int array of shape (N,) with 1-based cluster IDs.
        - linkage_matrix: scipy linkage matrix (for plotting dendrograms).
    """
    condensed = squareform(rmsd_matrix, checks=False)
    Z = linkage(condensed, method="complete")
    labels = fcluster(Z, t=cutoff, criterion="distance")
    return labels, Z


# ---------------------------------------------------------------------------
# Saturation analysis
# ---------------------------------------------------------------------------

def saturation_analysis(
    poses: List[PoseRecord],
    rmsd_matrix: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot mean pairwise RMSD vs. number of top-N poses (saturation curve).

    If the curve plateaus, most modes have been discovered. A monotonically
    rising curve indicates more samples are needed.

    Args:
        poses: List of PoseRecord (used only for count here; assumed rank-sorted).
        rmsd_matrix: (N, N) RMSD matrix from ``compute_rmsd_matrix``.
        ax: Optional existing matplotlib Axes to draw on.

    Returns:
        matplotlib Figure containing the saturation plot.
    """
    n = len(poses)
    means = []
    for k in range(1, n + 1):
        sub = rmsd_matrix[:k, :k]
        # upper triangle (excluding diagonal)
        vals = sub[np.triu_indices(k, k=1)]
        means.append(np.mean(vals) if len(vals) > 0 else 0.0)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    ax.plot(range(1, n + 1), means, marker="o", linewidth=2, color="#3b6fd4")
    ax.axhline(means[-1], color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Number of top-N poses")
    ax.set_ylabel("Mean pairwise RMSD (Å)")
    ax.set_title("Saturation analysis")
    ax.set_xticks(range(1, n + 1))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# RMSD heatmap
# ---------------------------------------------------------------------------

def plot_rmsd_heatmap(
    rmsd_matrix: np.ndarray,
    labels: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Clustered RMSD heatmap with dendrogram-based row/column reordering.

    Block structure in the heatmap indicates well-separated modes; a diffuse
    pattern indicates a broad unimodal distribution.

    Args:
        rmsd_matrix: Symmetric (N, N) RMSD array.
        labels: Optional cluster label array from ``cluster_poses``.
            When provided, cluster boundaries are drawn as rectangles.
        ax: Optional existing matplotlib Axes.

    Returns:
        matplotlib Figure.
    """
    n = rmsd_matrix.shape[0]
    condensed = squareform(rmsd_matrix, checks=False)
    Z = linkage(condensed, method="complete")
    order = leaves_list(Z)
    reordered = rmsd_matrix[np.ix_(order, order)]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    im = ax.imshow(reordered, cmap="viridis_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="RMSD (Å)")
    ax.set_title("Pairwise RMSD matrix (clustered)")
    ax.set_xlabel("Pose index (reordered)")
    ax.set_ylabel("Pose index (reordered)")
    tick_labels = [str(order[i] + 1) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=7)

    # Draw cluster boundaries if labels provided
    if labels is not None:
        reordered_labels = labels[order]
        boundaries = []
        for i in range(1, n):
            if reordered_labels[i] != reordered_labels[i - 1]:
                boundaries.append(i - 0.5)
        for b in boundaries:
            ax.axhline(b, color="red", linewidth=1.2)
            ax.axvline(b, color="red", linewidth=1.2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Torsional analysis
# ---------------------------------------------------------------------------

def _get_rotatable_bonds(mol: Chem.Mol) -> List[Tuple[int, int, int, int]]:
    """Return (a, b, c, d) atom-index tuples defining each rotatable-bond dihedral."""
    rot_smarts = Chem.MolFromSmarts(
        "[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]"
    )
    matches = mol.GetSubstructMatches(rot_smarts)
    dihedrals = []
    seen = set()
    for b_idx, c_idx in matches:
        key = tuple(sorted((b_idx, c_idx)))
        if key in seen:
            continue
        seen.add(key)
        b_atom = mol.GetAtomWithIdx(b_idx)
        c_atom = mol.GetAtomWithIdx(c_idx)
        a_neighbors = [n.GetIdx() for n in b_atom.GetNeighbors() if n.GetIdx() != c_idx]
        d_neighbors = [n.GetIdx() for n in c_atom.GetNeighbors() if n.GetIdx() != b_idx]
        if not a_neighbors or not d_neighbors:
            continue
        dihedrals.append((a_neighbors[0], b_idx, c_idx, d_neighbors[0]))
    return dihedrals


def torsion_rose_plots(
    poses: List[PoseRecord],
    ncols: int = 4,
) -> plt.Figure:
    """Circular histograms (rose plots) of dihedral angles per rotatable bond.

    High circular variance for a bond → the model is uncertain about that
    torsion. Sharp peaks → the angle is constrained.

    Args:
        poses: List of PoseRecord as returned by ``load_poses``.
        ncols: Number of columns in the subplot grid.

    Returns:
        matplotlib Figure with one polar subplot per rotatable bond.
    """
    dihedrals = _get_rotatable_bonds(poses[0].mol)
    n_bonds = len(dihedrals)
    if n_bonds == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No rotatable bonds found", ha="center", va="center")
        return fig

    nrows = (n_bonds + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.8, nrows * 2.8),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    axes = np.array(axes).flatten()

    bins = np.linspace(-np.pi, np.pi, 19)  # 18 bins of 20°

    for bond_i, (a, b, c, d) in enumerate(dihedrals):
        angles = []
        for pose in poses:
            conf = pose.mol.GetConformer()
            try:
                angle = rdMolTransforms.GetDihedralRad(conf, a, b, c, d)
            except Exception:
                continue
            angles.append(angle)
        angles = np.array(angles)

        ax = axes[bond_i]
        counts, bin_edges = np.histogram(angles, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = bin_edges[1] - bin_edges[0]
        ax.bar(bin_centers, counts, width=width, color="#3b6fd4", alpha=0.75, edgecolor="white")
        ax.set_title(f"Bond {bond_i + 1}\n({b}–{c})", fontsize=8, pad=2)
        ax.set_xticks([0, np.pi / 2, np.pi, -np.pi / 2])
        ax.set_xticklabels(["0°", "90°", "±180°", "270°"], fontsize=6)
        ax.tick_params(axis="y", labelsize=6)

        # Circular variance: 1 − |mean resultant length|
        R = np.abs(np.mean(np.exp(1j * angles))) if len(angles) > 0 else 0
        circ_var = 1 - R
        ax.set_xlabel(f"CV={circ_var:.2f}", fontsize=7, labelpad=2)

    # Hide unused subplots
    for i in range(n_bonds, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Torsion angle distributions", fontsize=12)
    return fig


# ---------------------------------------------------------------------------
# 3-D viewer coloured by confidence / cluster
# ---------------------------------------------------------------------------

def view_poses_colored(
    poses: List[PoseRecord],
    receptor_pdb: Optional[str] = None,
    results_dir: Optional[str] = None,
    color_by: str = "confidence",
    cluster_labels: Optional[np.ndarray] = None,
    width: int = 900,
    height: int = 600,
    show_surface: bool = False,
    surface_opacity: float = 0.15,
) -> Any:
    """Render all poses in py3Dmol, coloured by confidence or cluster membership.

    Args:
        poses: List of PoseRecord from ``load_poses``.
        receptor_pdb: Path to receptor PDB. If None and results_dir is given,
            inferred from metadata.
        results_dir: DiffDock output directory (used to find receptor PDB).
        color_by: ``'confidence'`` (blue=high, red=low) or ``'cluster'``
            (distinct colour per cluster; requires cluster_labels).
        cluster_labels: Int array of cluster IDs (required for color_by='cluster').
        width, height: Canvas size in pixels.
        show_surface: Draw a translucent VDW surface on the receptor.
        surface_opacity: Opacity of the receptor surface.

    Returns:
        py3Dmol.view interactive viewer.
    """
    try:
        import py3Dmol
    except ImportError as exc:
        raise ImportError("py3Dmol is required; install with `pip install py3Dmol`.") from exc

    if receptor_pdb is None and results_dir is not None:
        info = load_results_dir(results_dir)
        receptor_pdb = info["receptor_pdb"]

    # Build colour list
    if color_by == "confidence":
        confidences = np.array([p.confidence for p in poses], dtype=float)
        # normalise so that high confidence → blue, low → red
        vmin, vmax = confidences.min(), confidences.max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap("RdYlBu")
        colours = [
            mcolors.to_hex(cmap(norm(c))) for c in confidences
        ]
    elif color_by == "cluster":
        if cluster_labels is None:
            raise ValueError("cluster_labels must be provided when color_by='cluster'.")
        palette = [
            "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
            "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        ]
        unique = sorted(set(cluster_labels))
        cmap_cl = {cl: palette[i % len(palette)] for i, cl in enumerate(unique)}
        colours = [cmap_cl[cl] for cl in cluster_labels]
    else:
        raise ValueError(f"color_by must be 'confidence' or 'cluster', got {color_by!r}.")

    viewer = py3Dmol.view(width=width, height=height)
    viewer.setBackgroundColor("white")

    if receptor_pdb is not None:
        with open(receptor_pdb, "r", encoding="utf-8") as fh:
            viewer.addModel(fh.read(), "pdb")
        viewer.setStyle({"model": 0}, {"cartoon": {"color": "spectrum", "opacity": 0.8}})
        if show_surface:
            viewer.addSurface(
                py3Dmol.VDW,
                {"opacity": surface_opacity, "color": "white"},
                {"model": 0},
            )

    for i, (pose, colour) in enumerate(zip(poses, colours)):
        pdb_block = mol_to_pdb_block(pose.mol)
        viewer.addModel(pdb_block, "pdb")
        viewer.setStyle(
            {"model": -1},
            {
                "stick": {"radius": 0.15, "color": colour},
                "sphere": {"radius": 0.35, "color": colour, "opacity": 0.7},
            },
        )
        label = f"r{pose.rank} ({pose.confidence:.2f})"
        viewer.addLabel(
            label,
            {"fontSize": 9, "fontColor": colour, "backgroundColor": "white",
             "backgroundOpacity": 0.6, "showBackground": True},
            {"model": -1},
        )

    viewer.zoomTo({"model": -1})
    return viewer


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_full_analysis(
    results_dir: str,
    cutoff: float = 2.0,
) -> Dict[str, Any]:
    """Run the complete distribution analysis for one DiffDock results directory.

    Computes pairwise RMSD, clusters poses, and generates all diagnostic plots.

    Args:
        results_dir: Path to a DiffDock output directory for one complex.
        cutoff: RMSD threshold (Å) for clustering. Default is 2.0 Å.

    Returns:
        Dictionary with keys:
        - ``poses``          : List[PoseRecord]
        - ``rmsd_matrix``    : (N, N) ndarray
        - ``cluster_labels`` : (N,) int ndarray
        - ``linkage_matrix`` : scipy linkage matrix
        - ``n_clusters``     : int
        - ``saturation_fig`` : matplotlib Figure
        - ``heatmap_fig``    : matplotlib Figure
        - ``torsion_fig``    : matplotlib Figure
    """
    poses = load_poses(results_dir)
    rmsd_matrix = compute_rmsd_matrix(poses)
    labels, Z = cluster_poses(rmsd_matrix, cutoff=cutoff)

    saturation_fig = saturation_analysis(poses, rmsd_matrix)
    heatmap_fig = plot_rmsd_heatmap(rmsd_matrix, labels=labels)
    torsion_fig = torsion_rose_plots(poses)

    n_clusters = len(set(labels))
    print(
        f"Analysed {len(poses)} poses → {n_clusters} cluster(s) at {cutoff} Å cutoff.\n"
        f"Cluster populations: "
        + ", ".join(
            f"C{cl}={list(labels).count(cl)}"
            for cl in sorted(set(labels))
        )
    )

    return {
        "poses": poses,
        "rmsd_matrix": rmsd_matrix,
        "cluster_labels": labels,
        "linkage_matrix": Z,
        "n_clusters": n_clusters,
        "saturation_fig": saturation_fig,
        "heatmap_fig": heatmap_fig,
        "torsion_fig": torsion_fig,
    }
