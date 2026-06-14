"""Implementation check: re-run MIRA + TARP using the new molcalib API.

Runs the full evaluation (global MIRA symrmsd, global TARP centroid, and
per-DOF TARP/MIRA for translation / rotation / torsion) over the
PoseBusters benchmark using the reorganised thesis-repo code, then saves
results to metrics/molcalib_check/ so they can be compared to the
precomputed files in metrics/ without overwriting them.

Usage
-----
Called by the SLURM script diffdock_impl_check.sh; set env vars before running:

    RESULTS_DIR  — top-level DiffDock results directory (chunk_* layout)
    DATA_DIR     — PoseBusters dataset root (one subdir per complex)
    METRICS_DIR  — directory containing complex_names.npy and existing *.npy
    OUT_DIR      — output directory (must not exist to avoid overwriting)
    N_WORKERS    — parallel worker processes
    SEED         — master random seed
    K_TARP       — TARP reference draws per complex

All values can be overridden via environment variables; defaults target the
PoseBusters benchmark.
"""

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
# thesis/   → molcalib + eval_diffdock (pip-installed or on PYTHONPATH)
# thesis/diffdock/   → group_eval / group_mira_eval (need explicit path)

THESIS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if THESIS_DIR not in sys.path:
    sys.path.insert(0, THESIS_DIR)

# ── Configuration ─────────────────────────────────────────────────────────────
RDS          = "/home/qf226/rds/hpc-work"
RESULTS_DIR  = os.environ.get("RESULTS_DIR",  f"{RDS}/results/DiffDock/pb_evaluate_v2_merged/poses")
DATA_DIR     = os.environ.get("DATA_DIR",     f"{RDS}/data/posebusters_benchmark_set")
METRICS_DIR  = os.environ.get("METRICS_DIR",  f"{RDS}/results/DiffDock/pb_evaluate_v2_merged/metrics")
OUT_DIR      = os.environ.get("OUT_DIR",      f"{METRICS_DIR}/molcalib_check")
N_WORKERS    = int(os.environ.get("N_WORKERS", "12"))
SEED         = int(os.environ.get("SEED",     "42"))
K_TARP       = int(os.environ.get("K_TARP",  "100"))

print("=== DiffDock PoseBusters — implementation check ===")
print(f"  RESULTS_DIR : {RESULTS_DIR}")
print(f"  DATA_DIR    : {DATA_DIR}")
print(f"  METRICS_DIR : {METRICS_DIR}")
print(f"  OUT_DIR     : {OUT_DIR}")
print(f"  N_WORKERS   : {N_WORKERS}")
print(f"  SEED        : {SEED}")
print(f"  K_TARP      : {K_TARP}")
print()

# Guard: refuse to overwrite existing output
if os.path.exists(OUT_DIR):
    existing = os.listdir(OUT_DIR)
    if existing:
        raise FileExistsError(
            f"OUT_DIR already contains files: {OUT_DIR!r}\n"
            "Delete it manually if you want to re-run."
        )

os.makedirs(OUT_DIR, exist_ok=True)
group_out = os.path.join(OUT_DIR, "group_eval")
os.makedirs(group_out, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from eval_diffdock.loader import build_results_index
from eval_diffdock.mira_runner import compute_mira_scores
from eval_diffdock.tarp_runner import run_tarp_eval

from eval_diffdock.group_tarp_runner import run_group_tarp_eval, run_group_distances
from eval_diffdock.group_mira_runner import run_group_mira_eval

# ── Load complex names ────────────────────────────────────────────────────────
complex_names_all = np.load(os.path.join(METRICS_DIR, "complex_names.npy"),
                            allow_pickle=True)
results_index = build_results_index(RESULTS_DIR)
complex_names = [n for n in complex_names_all if n in results_index]
print(f"Complexes available : {len(complex_names_all)}")
print(f"Complexes in index  : {len(complex_names)}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Global MIRA — symrmsd metric
# ─────────────────────────────────────────────────────────────────────────────
print("=== 1/4  Global MIRA (symrmsd) ===")
t0 = time.time()
mira_names, mira_scores = compute_mira_scores(
    complex_names,
    results_index,
    DATA_DIR,
    num_runs=20,
    metric="symrmsd",
    seed=SEED,
    verbose=True,
    n_workers=N_WORKERS,
)
print(f"Elapsed: {time.time() - t0:.1f}s")
np.save(os.path.join(OUT_DIR, "mira_names_symrmsd.npy"),  mira_names)
np.save(os.path.join(OUT_DIR, "mira_scores_symrmsd.npy"), mira_scores)
print(f"Saved {len(mira_scores)} scores → {OUT_DIR}/mira_names/scores_symrmsd.npy")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Global TARP — centroid mode, K=100
# ─────────────────────────────────────────────────────────────────────────────
print("=== 2/4  Global TARP (centroid, K=100) ===")
t0 = time.time()
tarp_fracs = run_tarp_eval(
    complex_names,
    results_index,
    DATA_DIR,
    K=K_TARP,
    mode="centroid",
    seed=SEED,
    verbose=True,
    n_workers=N_WORKERS,
)
print(f"Elapsed: {time.time() - t0:.1f}s")
np.save(os.path.join(OUT_DIR, "tarp_fractions_centroid.npy"),  tarp_fracs)
print(f"Saved {tarp_fracs.shape} → {OUT_DIR}/tarp_fractions_centroid.npy")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-DOF TARP — translation / rotation / torsion
# ─────────────────────────────────────────────────────────────────────────────
print("=== 3/4  Per-DOF TARP (K=100) ===")
t0 = time.time()
tarp_dof = run_group_tarp_eval(
    complex_names,
    results_index,
    DATA_DIR,
    K=K_TARP,
    seed=SEED,
    verbose=True,
    n_workers=N_WORKERS,
)
print(f"Elapsed: {time.time() - t0:.1f}s")
for dof in ("translation", "rotation", "torsion"):
    arr = tarp_dof[dof]
    path = os.path.join(group_out, f"tarp_fractions_{dof}.npy")
    np.save(path, arr)
    print(f"  {dof:12s}  {arr.shape}  → {path}")
np.save(os.path.join(group_out, "complex_names.npy"),    tarp_dof["names"])
np.save(os.path.join(group_out, "n_rot_bonds_tarp.npy"), tarp_dof["n_rot_bonds"])
print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-DOF MIRA — translation / rotation / torsion
# ─────────────────────────────────────────────────────────────────────────────
print("=== 4/4  Per-DOF MIRA ===")
t0 = time.time()
mira_dof = run_group_mira_eval(
    complex_names,
    results_index,
    DATA_DIR,
    num_runs=20,
    seed=SEED,
    verbose=True,
    n_workers=N_WORKERS,
)
print(f"Elapsed: {time.time() - t0:.1f}s")
for dof in ("translation", "rotation", "torsion"):
    names_d, scores_d = mira_dof[dof]
    names_path  = os.path.join(group_out, f"mira_names_{dof}.npy")
    scores_path = os.path.join(group_out, f"mira_scores_{dof}.npy")
    np.save(names_path,  names_d)
    np.save(scores_path, scores_d)
    print(f"  {dof:12s}  n={len(scores_d)}  mean={np.nanmean(scores_d):.4f}")
print()

print("=== All done ===")
print(f"Outputs in : {OUT_DIR}")
