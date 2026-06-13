# DiffDock-specific evaluation pipeline — uses molcalib for core metrics.

from eval_diffdock.loader import (
    build_results_index,
    load_crystal_coords,
    load_sample_coords,
    load_protein_ca_coords,
)
from eval_diffdock.mira_runner import (
    compute_mira_scores,
    compute_rmsd_accuracy,
    load_poses,
    compute_mira_one_complex,
)
from eval_diffdock.tarp_runner import run_tarp_eval
from eval_diffdock.group_tarp_runner import run_group_tarp_eval, run_group_distances
from eval_diffdock.group_mira_runner import run_group_mira_eval
from eval_diffdock.pb_eval import (
    run_posebusters,
    load_pb_filtered_coords,
    compute_rmsd_accuracy_filtered,
    compute_mira_filtered,
)
from eval_diffdock.distribution_analysis import (
    PoseRecord,
    load_poses as load_poses_from_dir,
    compute_rmsd_matrix,
    compute_rmsd_to_crystal,
    plot_confidence_vs_rmsd,
    cluster_poses,
    saturation_analysis,
    plot_rmsd_heatmap,
    torsion_rose_plots,
    view_poses_colored,
    run_full_analysis,
)

__all__ = [
    # loader
    "build_results_index",
    "load_crystal_coords",
    "load_sample_coords",
    "load_protein_ca_coords",
    # mira_runner
    "compute_mira_scores",
    "compute_rmsd_accuracy",
    "load_poses",
    "compute_mira_one_complex",
    # tarp_runner
    "run_tarp_eval",
    # group_tarp_runner
    "run_group_tarp_eval",
    "run_group_distances",
    # group_mira_runner
    "run_group_mira_eval",
    # pb_eval
    "run_posebusters",
    "load_pb_filtered_coords",
    "compute_rmsd_accuracy_filtered",
    "compute_mira_filtered",
    # distribution_analysis
    "PoseRecord",
    "load_poses_from_dir",
    "compute_rmsd_matrix",
    "compute_rmsd_to_crystal",
    "plot_confidence_vs_rmsd",
    "cluster_poses",
    "saturation_analysis",
    "plot_rmsd_heatmap",
    "torsion_rose_plots",
    "view_poses_colored",
    "run_full_analysis",
]
