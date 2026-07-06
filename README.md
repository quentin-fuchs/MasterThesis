# Thesis: Calibration Evaluation of Molecular Docking Models

MPhil thesis project evaluating whether DiffDock and SigmaDock produce well-calibrated pose distributions using the TARP and MIRA likelihood-free calibration metrics.

## What this repo does

Takes pre-run inference results from DiffDock and SigmaDock and evaluates them on two test sets:

- **PDBBind test set** — 322 protein-ligand complexes, 40 predicted poses each
- **PoseBusters benchmark** — 305 complexes, 40 poses each, with physical validity filtering
- **Top-1 runs PDBBind** — 10 independent DiffDock runs merged into top-1 (10 poses) and top-3 (30 poses) variants

Metrics computed:

- **TARP** (Tests of Accuracy with Random Points) — calibration diagnostic via ECP curves
- **MIRA** (Model-Independent Random-radius Assessment) — scalar calibration score, null ≈ 0.683 for S=40
- **RMSD accuracy** — fraction of complexes with top-1 pose within 2 Å / 5 Å of crystal
- **PoseBusters pass rate** — fraction of poses passing ~20 physical validity checks

---

## Quickstart

```bash
# 1. Clone the repo
git clone <repo-url>
cd SigmaDock-DiffDock-Eval

# 2. Create and activate the analysis environment
conda env create -f envs/analysis.yml
conda activate analysis

# 3. Install the molcalib package and evaluation modules
pip install -e .

# 4. Register the Jupyter kernel
python -m ipykernel install --user --name analysis --display-name "Docking (analysis)"

# 5. Download the data (see Data section below), then open a notebook
jupyter notebook notebooks/tarp_analysis.ipynb
```

---

## Installation

Three conda environments are used. Only `analysis` is needed to reproduce the figures and run the calibration metrics; the inference environments are required only to re-run DiffDock or SigmaDock inference from scratch.

### Analysis (metrics + notebooks)

```bash
conda env create -f envs/analysis.yml
conda run -n analysis pip install -e .
python -m ipykernel install --user --name analysis --display-name "Docking (analysis)"
```

This installs Python 3.10 with RDKit 2025.09.6, spyrmsd 0.9.0, PoseBusters 0.6.5, and Jupyter. The RDKit version is pinned — do not upgrade without re-testing SDF loading and symRMSD calculations.

### DiffDock inference (optional)

```bash
conda env create -f diffdock/environment.yml -n diffdock_inference
conda run -n diffdock_inference pip install \
  "openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307"
```

### SigmaDock inference (optional, CSD3/glibc ≥ 2.28)

```bash
conda create -y -n sigmadock_inference python=3.10
conda activate sigmadock_inference
cd sigmadock
bash install.sh cu118 train,test
```

Both `train` and `test` extras are required — `sample.py` imports `spyrmsd` at module level via `statistics.py`.

For complete setup notes — including the gnina rescoring binary for SigmaDock — see [`envs/README.md`](envs/README.md).

---

## Data

All large data files are hosted on OneDrive:

**[Download data](https://1drv.ms/f/c/9c84199f301a2f33/IgD2iP4y3aylQoSr4Z9py6MGAdffGun1MvAjnXTvDCl072g?e=pM1uFs)**

After downloading, unzip and place the contents so the directory layout matches:

```
data/
  PDBBind_processed/          Crystal structures + processed proteins (PDBBind test set)
  posebusters_benchmark_set/  PoseBusters crystal structures
  inference/                  Inference CSVs and protein annotations

results/DiffDock/
  pdbbind_testset/            40-sample baseline inference results
    metrics/                  Pre-computed .npy arrays (TARP fractions, MIRA scores, RMSD)
  pb_evaluate_v2_merged/      PoseBusters inference results
    poses/                    Per-complex SDF files
    metrics/
  top1_runs_v2_top1_merged/   10-run top-1 merged (10 poses per complex)
    metrics/
  top1_runs_v2_top3_merged/   10-run top-3 merged (30 poses per complex)
    metrics/

results/SigmaDock/
  pdbbind_testset/
    metrics/
```

The notebooks load pre-computed `metrics/` arrays directly — no heavy computation is needed to reproduce the figures.

---

## Repository layout

```
molcalib/                   Core metric implementations (installable package)
  mira.py                   MIRA score (symRMSD-based, Sharief et al. 2026)
  tarp.py                   TARP fractions and ECP curve plotting (Lemos et al. 2023)
  distances.py              Symmetry-corrected RMSD via spyrmsd (with 4 s timeout)
  prior.py                  Reference pose prior (generate_reference_coords)
  io.py                     SDF/PDB loading utilities
  style.py                  Shared matplotlib style configuration

eval_diffdock/              DiffDock-specific data loading and batch evaluation
  loader.py                 Build results index from DiffDock output directories
  tarp_runner.py            Parallel TARP evaluation over a test set
  mira_runner.py            Parallel MIRA evaluation and RMSD accuracy
  pb_eval.py                PoseBusters filtering and PB-filtered metrics
  scripts/                  Entry-point scripts for batch evaluation runs
    run_tarp_mira_pdbbind.py       TARP + MIRA for the PDBBind test set
    run_tarp_mira_pb_benchmark.py  TARP + MIRA for the PoseBusters benchmark
    run_pb_eval_pdbbind.py         PoseBusters filtering for PDBBind
    run_pb_eval_posebusters.py     PoseBusters filtering for PoseBusters benchmark
    run_group_eval.py              Per-protein-family grouped TARP/MIRA
    run_rmsd_eval.py               RMSD accuracy (top-1 and best-of-N)
    ...

eval_sigmadock/             SigmaDock-specific evaluation (mirrors eval_diffdock)

notebooks/                  Analysis and figure generation
  tarp_analysis.ipynb                   Main thesis figure: TARP ECP curves for the PDBBind
                                        test set, with per-protein-family breakdown
  mira_tarp_sigmadock.ipynb             TARP ECP curves and MIRA scores for SigmaDock on
                                        the PDBBind test set; produces the SigmaDock thesis figure
  comparison_diffdock_sigmadock.ipynb   Side-by-side DiffDock vs SigmaDock MIRA and symRMSD
                                        comparison across the shared test set
  posebusters_calibration.ipynb         TARP/MIRA calibration on the 305-complex PoseBusters
                                        benchmark; evaluates effect of physical validity filtering
  pb_filtering_analysis.ipynb           Root-cause breakdown of PoseBusters pass/fail rates:
                                        which validity checks fail and for which pose ranks
  group_tarp_mira.ipynb                 Per-protein-family TARP and MIRA grouped analysis;
                                        identifies whether calibration varies across target classes
  confidence_calibration_analysis.ipynb Relationship between DiffDock confidence scores and
                                        per-complex calibration quality (MIRA)
  distribution_analysis.ipynb           Pose distribution visualisations: RMSD histograms,
                                        centroid spread, and sample diversity
  sensitivity_n_steps.ipynb             ODE solver step-count sensitivity analysis for SigmaDock
  check_implementation_change.ipynb     Validation notebook: verifies that metric results are
                                        stable across implementation changes
  plot_inference_results.ipynb          Generic inference result visualisation (exploratory)
  generate_esm_embeddings.ipynb         ESM protein embedding generation (preprocessing)

slurm/                      Exemplary SLURM job scripts used on Cambridge CSD3 (not portable as-is)
  diffdock/                 DiffDock evaluation jobs (inference, TARP, MIRA, PoseBusters)

docs/                       Package documentation
  molcalib.md               Full molcalib API reference, background, and usage examples

figures/                    Output figures (generated by notebooks)
diffdock/                   DiffDock source (submodule)
sigmadock/                  SigmaDock source (submodule)
envs/                       Conda environment files
  analysis.yml              Main analysis environment (metrics + notebooks)
  README.md                 Environment setup notes including inference envs
```

---

## Usage

### Computing metrics on a single complex (Python API)

The `molcalib` package can be used independently of the eval pipeline. The inputs are RDKit `Mol` objects and NumPy coordinate arrays.
For stability it should be used with benchmarks of at least 100 different complexes.

```python
import numpy as np
from rdkit import Chem
from molcalib.mira import mira_score, mira_null
from molcalib.tarp import tarp_fractions, ecp_from_fractions, plot_ecp

# crystal_mol:    RDKit Mol with a single conformer (the crystal pose)
# crystal_coords: (N_atoms, 3) float array — crystal atom positions
# sample_coords:  (S, N_atoms, 3) float array — S predicted poses
# ca_coords:      (N_CA, 3) float array — protein Cα positions (defines the prior region)

crystal_mol = Chem.SDMolSupplier("crystal.sdf", removeHs=False)[0]
crystal_coords = crystal_mol.GetConformer().GetPositions()   # (N, 3)
sample_coords  = np.stack([m.GetConformer().GetPositions()
                            for m in Chem.SDMolSupplier("poses.sdf", removeHs=False)])  # (S, N, 3)

# MIRA score
score = mira_score(crystal_mol, crystal_coords, sample_coords, ca_coords)
null  = mira_null(S=sample_coords.shape[0])   # ≈ 0.683 for S=40
print(f"MIRA: {score:.3f}  (null ≈ {null:.3f})")

# TARP fractions and ECP curve
fracs = tarp_fractions(crystal_mol, crystal_coords, sample_coords, ca_coords, K=10, seed=42)
ecp   = ecp_from_fractions(fracs[np.newaxis])  # shape (1, K) → ECP curve

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plot_ecp(ecp, alpha=np.linspace(0, 1, 10), ax=ax)
plt.show()
```

### Running batch evaluations

The evaluation scripts in `eval_diffdock/scripts/` run TARP, MIRA, and RMSD accuracy over a full test set. Each script has configurable path constants at the top — edit `DATA_DIR`, `RESULTS_FULL`, and `MERGED` to match your local layout before running.

```bash
# PDBBind test set (322 complexes, ~40 poses each)
# Edit DATA_DIR, RESULTS_FULL, MERGED at the top of the script first
N_WORKERS=8 python eval_diffdock/scripts/run_tarp_mira_pdbbind.py

# PoseBusters benchmark (305 complexes)
N_WORKERS=8 python eval_diffdock/scripts/run_tarp_mira_pb_benchmark.py

# PoseBusters physical-validity filtering
python eval_diffdock/scripts/run_pb_eval_pdbbind.py

# Per-protein-family grouped evaluation
python eval_diffdock/scripts/run_group_eval.py
```

Outputs are `.npy` / `.npz` arrays saved into a `metrics/` subdirectory alongside the results. These are the files the notebooks load.

`N_WORKERS` controls multiprocessing parallelism. For HPC use, the scripts were originally submitted via SLURM; the shell scripts are not included in this repository but the Python entry points are self-contained.

### Batch evaluation — SigmaDock

The `eval_sigmadock/scripts/` directory mirrors `eval_diffdock/scripts/` with SigmaDock-specific data loaders. Usage is identical; adjust `DATA_DIR` and `RESULTS_DIR` accordingly.

---

## Generative AI Declaration

In preparation for this report, I adhered to the coursewide AI policy.
Claude Code assisted in understanding the existing codebase and in writing functions and shell scripts. Gemini and Claude were further used for drafting and proofreading of the report.

---

## molcalib

`molcalib` is the core contribution of this project — a standalone Python package implementing TARP and MIRA calibration diagnostics for molecular docking.

Full documentation including mathematical background, complete API reference, and usage examples: **[docs/molcalib.md](docs/molcalib.md)**

```bash
pip install -e .
```
