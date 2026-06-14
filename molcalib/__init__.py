"""molcalib — MIRA and TARP calibration diagnostics for molecular docking.

Implements the MIRA score (Sharief et al. 2026, arXiv:2605.02014) and TARP
coverage test (Lemos & Coogan et al. 2023, arXiv:2302.03026) for evaluating
whether a docking model's predicted pose distribution is well-calibrated.

Public API
----------
Distances:
    compute_rmsd_symmetry     — symmetry-corrected heavy-atom RMSD (spyrmsd)
    compute_centroid_distance — ligand-centroid Euclidean distance

Prior (reference pose distribution):
    prepare_reference_template — build ETKDG template + rotatable bonds
    generate_reference_coords  — draw one prior pose (random Cα + torsions + SO3)

MIRA:
    mira_null                 — null reference score for S posterior samples
    mira_score                — MIRA score for a single complex (any metric)
    bootstrap_mira_groups     — bootstrap per-group mean and 90% CI

TARP:
    tarp_fractions            — K coverage fractions for a single complex
    ecp_from_fractions        — ECP curve from fraction matrix
    bootstrap_ecp             — bootstrap confidence bands
    plot_ecp                  — plot ECP with calibration diagonal

I/O:
    load_ligand_sdf           — load heavy-atom RDKit mol + coordinates from SDF
    load_protein_ca_coords    — load Cα coordinates from a PDB file
"""

from molcalib.distances import (
    compute_rmsd_symmetry,
    compute_rmsd_symmetry_multi,
    compute_centroid_distance,
)
from molcalib.prior import (
    prepare_reference_template,
    generate_reference_coords,
)
from molcalib.mira import (
    mira_null,
    mira_score,
    bootstrap_mira_groups,
)
from molcalib.tarp import (
    tarp_fractions,
    ecp_from_fractions,
    bootstrap_ecp,
    plot_ecp,
)
from molcalib.io import (
    load_ligand_sdf,
    load_protein_ca_coords,
)

__all__ = [
    "compute_rmsd_symmetry",
    "compute_rmsd_symmetry_multi",
    "compute_centroid_distance",
    "prepare_reference_template",
    "generate_reference_coords",
    "mira_null",
    "mira_score",
    "bootstrap_mira_groups",
    "tarp_fractions",
    "ecp_from_fractions",
    "bootstrap_ecp",
    "plot_ecp",
    "load_ligand_sdf",
    "load_protein_ca_coords",
]
