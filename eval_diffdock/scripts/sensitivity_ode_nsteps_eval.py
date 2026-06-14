"""
Evaluate MIRA and centroid TARP for the ODE / n_steps sensitivity study.

Conditions compared:
  sde_20  (baseline) — ~/rds/hpc-work/results/DiffDock/pb_evaluate_v2_merged/
  sde_10             — sensitivity_ode_nsteps/sde_10/
  sde_50             — sensitivity_ode_nsteps/sde_50/
  ode_20             — sensitivity_ode_nsteps/ode_20/
  ode_50             — sensitivity_ode_nsteps/ode_50/

Results are filtered to the 100-complex whitelist. The baseline already has
results for 99 of the 100 (7FRX_O88 is missing); the comparison uses the
intersection of available complexes per condition.

Run on a CPU icelake node (MIRA is compute-light; TARP at K=10 takes ~5 min):
  sbatch ~/slurm/DiffDock/diffdock_run_notebook.sh  (or adapt for a script job)

Output: results/sensitivity_ode_nsteps/summary.json  (MIRA table + TARP arrays)
"""

import json
import sys
from pathlib import Path

import numpy as np

THESIS_DIR = Path("/home/qf226/MProject/thesis")
RDS = Path("/home/qf226/rds/hpc-work")
# 100 complexes randomly sampled from all 308 PB complexes (seed=42)
WHITELIST_PATH = THESIS_DIR / "diffdock" / "data" / "splits" / "pb_sensitivity_rand100.txt"
DATA_DIR = str(RDS / "data/posebusters_benchmark_set")

BASELINE_DIR = RDS / "results/DiffDock/pb_evaluate_v2_merged"
SENS_ROOT    = RDS / "results/DiffDock/sensitivity_ode_nsteps_v2"

CONDITIONS = {
    "sde_20": BASELINE_DIR,
    "sde_10": SENS_ROOT / "sde_10",
    "sde_50": SENS_ROOT / "sde_50",
    "ode_20": SENS_ROOT / "ode_20",
}

sys.path.insert(0, str(THESIS_DIR))

from eval_diffdock.mira_runner import compute_mira_scores
from molcalib.mira import mira_null
from eval_diffdock.tarp_runner import run_tarp_eval
from molcalib.tarp import ecp_from_fractions


def build_flat_index(results_dir: Path) -> dict:
    """Index a flat results dir (complex_name/ subdirs directly under results_dir)."""
    return {d.name: d for d in results_dir.iterdir() if d.is_dir()}


def load_whitelist(path: Path) -> list:
    return path.read_text().split()


def run_condition(label: str, results_dir: Path, complex_names: list) -> dict:
    print(f"\n=== {label} ===")
    index = build_flat_index(results_dir)
    available = [n for n in complex_names if n in index]
    print(f"  Whitelist complexes available: {len(available)}/{len(complex_names)}")

    # MIRA
    names_out, mira_scores = compute_mira_scores(
        available, index, DATA_DIR, num_runs=100, verbose=True
    )
    mira_mean = float(np.mean(mira_scores))
    mira_std  = float(np.std(mira_scores))
    null      = mira_null(40)
    print(f"  MIRA: {mira_mean:.4f} ± {mira_std:.4f}  (null={null:.4f}, Δ={mira_mean - null:+.4f})")

    # Centroid TARP at K=10 (fast; upgrade to K=100 for publication)
    fractions = run_tarp_eval(
        available, index, DATA_DIR, K=10, mode="centroid", seed=42, verbose=True
    )
    ecp = ecp_from_fractions(fractions)
    print(f"  TARP centroid ECP computed over {len(available)} complexes, K=10")

    return {
        "n_complexes": len(available),
        "mira_mean": mira_mean,
        "mira_std": mira_std,
        "mira_null": null,
        "mira_delta": mira_mean - null,
        "tarp_centroid_ecp": ecp.tolist(),
        "tarp_fractions": fractions.tolist(),
    }


def main():
    whitelist = load_whitelist(WHITELIST_PATH)
    print(f"Whitelist: {len(whitelist)} complexes")

    out_dir = SENS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, results_dir in CONDITIONS.items():
        if not results_dir.exists():
            print(f"\n[SKIP] {label}: {results_dir} not found")
            continue
        results[label] = run_condition(label, results_dir, whitelist)

    # Summary table
    print("\n\n=== Summary ===")
    print(f"{'Condition':<10}  {'n':>4}  {'MIRA':>7}  {'Δ null':>8}")
    for label, r in results.items():
        print(f"{label:<10}  {r['n_complexes']:>4}  {r['mira_mean']:.4f}  {r['mira_delta']:+.4f}")

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
