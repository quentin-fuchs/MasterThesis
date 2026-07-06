# molcalib

`molcalib` is a Python package implementing two likelihood-free calibration diagnostics for molecular docking: **TARP** (Tests of Accuracy with Random Points) and **MIRA** (Model-Independent Random-radius Assessment). Both metrics test whether a docking model's predicted pose distribution is well-calibrated without requiring a tractable likelihood function.

The package was developed as part of an MPhil thesis evaluating the calibration of DiffDock and SigmaDock on the PDBBind test set and the PoseBusters benchmark.

---

## Contents

- [Background](#background)
  - [Calibration in molecular docking](#calibration-in-molecular-docking)
  - [The reference pose prior](#the-reference-pose-prior)
  - [TARP](#tarp)
  - [MIRA](#mira)
- [Installation](#installation)
- [API reference](#api-reference)
  - [molcalib.prior](#molcalibprior)
  - [molcalib.distances](#molcalibdistances)
  - [molcalib.mira](#molcalibmira)
  - [molcalib.tarp](#molcalibtarp)
  - [molcalib.io](#molcalibio)
- [End-to-end example](#end-to-end-example)

---

## Background

### Calibration in molecular docking

A docking model is **calibrated** if its predicted distribution over poses matches the true conditional distribution given the protein-ligand complex. For diffusion-based models like DiffDock and SigmaDock, which sample a posterior over poses, calibration means the spread and location of S predicted samples faithfully represent the model's uncertainty — not just that one of them is accurate.

Traditional accuracy metrics (e.g. top-1 RMSD < 2 Å) only evaluate the single best prediction and say nothing about the rest of the distribution. The calibration metrics in molcalib evaluate the full S-sample distribution.

### The reference pose prior

Both TARP and MIRA require a reference distribution `p(c | protein)` over pose space that depends only on the protein, not on the crystal pose. Using a protein-dependent prior — rather than a fixed uniform box — makes the test sensitive to uninformative posteriors: a model that always returns prior samples is detected as uncalibrated.

molcalib uses a prior that mirrors DiffDock's forward diffusion process at `t = 1` (`randomize_position` in DiffDock's `utils/sampling.py`):

- **Torsion angles** — uniform on [−π, π] applied to an ETKDGv2 conformer of the ligand
- **Global rotation** — uniform on SO(3) via a random unit quaternion
- **Translation** — centroid drawn from a Gaussian centred at the protein Cα centre-of-mass:

  ```
  centroid ~ N(Cα_COM,  σ² I)
  where  σ = std(‖Cα − Cα_COM‖) × 1.4602 / 1.73
  ```

  The constant 1.4602 is DiffDock's `initial_noise_std_proportion` from `default_inference_args.yaml`.

### TARP

*Lemos et al. 2023 — arXiv:2302.03026*

For a complex with crystal pose `y*` and S posterior samples `{y₁, …, yₛ}`, TARP draws K random reference poses `{c₁, …, c_K}` from the prior and computes a **coverage fraction** for each:

```
f_k = (1/S) Σⱼ  1[ d(cₖ, yⱼ) < d(cₖ, y*) ]
```

where `d` is either symmetry-corrected RMSD or centroid distance.

`f_k` is the fraction of predicted samples that fall closer to the reference than the crystal pose does. Under perfect calibration, `f_k ~ Uniform[0, 1]`.

The **Expected Coverage Probability (ECP)** curve pools fractions across all complexes and reference draws:

```
ECP(α)  =  fraction of all f values ≤ α
```

Under perfect calibration, `ECP(α) = α` (the diagonal). Deviations indicate:

| ECP relative to diagonal | Interpretation |
|---|---|
| Below diagonal | Mode-collapsed: samples cluster too tightly; crystal rarely the furthest point |
| Above diagonal | Over-dispersed: samples spread too widely; crystal often the furthest point |
| On diagonal | Well-calibrated |

The area between the ECP curve and the diagonal (ATC score = `∫ (ECP − α) dα`) provides a signed scalar summary: negative = over-confident, positive = conservative, zero = calibrated.

### MIRA

*Sharief et al. 2026 — arXiv:2605.02014*

MIRA reduces calibration to a scalar score using random balls. For each of `T` Monte Carlo runs:

1. Draw a center `c` from the prior
2. Pick a random reference sample `yᵣ` from the S posterior samples; set ball radius `r = d(c, yᵣ)`
3. Count `n⁺ = #{j ≠ r : d(c, yⱼ) < r}` (samples inside the ball, excluding `yᵣ`; `N = S − 1` total counted)
4. Set `k = 1` if the crystal pose is inside the ball (`d(c, y*) ≤ r`), else `k = 0`
5. Compute the Laplace-smoothed, normalised calibration estimate:

   ```
   p_in  = (n⁺ + 1) / (N + 2)
   p_out = (N − n⁺ + 1) / (N + 2)
   calib = (p_in · k  +  p_out · (1 − k))  /  ((N + 1) / (N + 2))
   ```

The MIRA score is `mean(calib)` over all T runs.

**Null reference.** Under perfect calibration the expected score is:

```
null(S)  =  (2/3) · (S + 1) / S
```

For S = 40 samples: `null ≈ 0.683`. The null is derived from `E[calib] = 2/3` under calibration, normalised by the maximum achievable value `(N+1)/(N+2)`.

| MIRA relative to null | Interpretation |
|---|---|
| Score > null | Over-dispersed: samples spread too wide |
| Score < null | Mode-collapsed: samples clustered too tightly |
| Score ≈ null | Well-calibrated |

---

## Installation

```bash
# From the repo root — installs molcalib plus the eval_diffdock / eval_sigmadock modules
pip install -e .
```

Requires Python ≥ 3.9. Core dependencies: `numpy`, `scipy`, `rdkit`, `spyrmsd`, `networkx`, `prody`.

For the full analysis environment including notebooks and PoseBusters:

```bash
conda env create -f envs/analysis.yml
conda run -n analysis pip install -e .
```

---

## API reference

All public symbols are importable directly from `molcalib`:

```python
from molcalib import (
    # Prior
    prepare_reference_template, generate_reference_coords,
    # Distances
    compute_rmsd_symmetry, compute_centroid_distance,
    # MIRA
    mira_null, mira_score, bootstrap_mira_groups,
    # TARP
    tarp_fractions, ecp_from_fractions, bootstrap_ecp, plot_ecp,
    # I/O
    load_ligand_sdf, load_protein_ca_coords,
)
```

---

### molcalib.prior

Generates random reference poses from the prior distribution `p(c | protein)`. Call `prepare_reference_template` once per complex and then `generate_reference_coords` for each draw.

---

#### `prepare_reference_template(crystal_mol)`

Build the reusable template for reference pose sampling.

Generates an ETKDGv2 conformer of the ligand and identifies rotatable bonds using DiffDock's graph-connectivity criterion: a bond is rotatable if removing it splits the molecular graph into two components each containing ≥ 2 atoms (non-ring, non-terminal bonds only).

**Args**

| Name | Type | Description |
|---|---|---|
| `crystal_mol` | RDKit `Mol` | Heavy-atom molecule (no hydrogens) |

**Returns** `(template_mol, rot_bonds)` — `template_mol` is an RDKit `Mol` with one ETKDG conformer; `rot_bonds` is a list of `(n0, a, b, n1)` atom-index tuples for `rdMolTransforms.SetDihedralRad`. Falls back to the crystal conformer if ETKDG embedding fails after 3 attempts.

---

#### `generate_reference_coords(template_mol, rot_bonds, ca_coords, rng)`

Draw one reference pose from the prior. Cheap to call repeatedly since the ETKDG conformer is pre-computed by `prepare_reference_template`.

Steps (mirroring DiffDock's `randomize_position` at `t = 1`):
1. Copy the ETKDG template conformer
2. Randomise all rotatable torsion angles uniformly in [−π, π]
3. Centre the molecule at the origin
4. Apply a uniformly random SO(3) rotation (sampled via unit quaternion)
5. Translate centroid to `N(Cα_COM, σ² I)` where `σ = std_Cα × 1.4602 / 1.73`

**Args**

| Name | Type | Description |
|---|---|---|
| `template_mol` | RDKit `Mol` | Template with one ETKDG conformer, from `prepare_reference_template` |
| `rot_bonds` | list of tuples | `(n0, a, b, n1)` tuples from `prepare_reference_template` |
| `ca_coords` | `(N_res, 3)` ndarray | Protein Cα coordinates |
| `rng` | `numpy.random.Generator` | Random number generator |

**Returns** `(N_atoms, 3)` numpy array — coordinates of the reference pose.

---

### molcalib.distances

Distance functions for comparing ligand poses. All functions operate on numpy arrays and RDKit `Mol` objects; no file I/O.

---

#### `compute_rmsd_symmetry(mol, ref_coords, query_coords_list, timeout=4)`

Symmetry-corrected heavy-atom RMSD between `ref_coords` and each pose in `query_coords_list`.

Uses spyrmsd's Hungarian matching over molecular graph automorphisms so that chemically equivalent atoms (e.g. in symmetric rings or identical arms) are correctly permuted before computing RMSD. A per-call timeout guards against pathological automorphism groups that can cause the Hungarian algorithm to run indefinitely.

**Args**

| Name | Type | Description |
|---|---|---|
| `mol` | RDKit `Mol` | Heavy-atom molecule defining the graph |
| `ref_coords` | `(N_atoms, 3)` ndarray | Reference pose |
| `query_coords_list` | list of `(N_atoms, 3)` ndarrays | Poses to compare |
| `timeout` | float | Seconds per spyrmsd call (default 4); returns NaN on timeout |

**Returns** `(len(query_coords_list),)` numpy array. NaN for any timed-out or failed call.

---

#### `compute_rmsd_symmetry_multi(mol, all_ref_coords, query_coords_list, timeout=4)`

Symmetry-corrected RMSD taking the element-wise **minimum** over multiple crystal conformers. Use when the crystal structure contains multiple copies or alternate conformations.

**Args**

| Name | Type | Description |
|---|---|---|
| `mol` | RDKit `Mol` | Heavy-atom molecule |
| `all_ref_coords` | list of `(N_atoms, 3)` ndarrays | One array per crystal conformer |
| `query_coords_list` | list of `(N_atoms, 3)` ndarrays | Predicted poses |
| `timeout` | float | Passed through to `compute_rmsd_symmetry` |

**Returns** `(len(query_coords_list),)` numpy array.

---

#### `compute_centroid_distance(ref_coords, query_coords_list)`

Euclidean distance between ligand centroids (mean atom positions). Faster than symRMSD and used in TARP's `mode='centroid'`.

**Args**

| Name | Type | Description |
|---|---|---|
| `ref_coords` | `(N_atoms, 3)` ndarray | Reference pose |
| `query_coords_list` | list of `(N_atoms, 3)` ndarrays | Query poses |

**Returns** `(len(query_coords_list),)` numpy array.

---

### molcalib.mira

Implements the MIRA calibration score (Sharief et al. 2026).

---

#### `mira_null(S)`

Expected MIRA score under perfect calibration for `S` posterior samples.

```
null(S) = (2/3) · (S + 1) / S
```

**Args**

| Name | Type | Description |
|---|---|---|
| `S` | int | Number of posterior samples per complex |

**Returns** Float. For S = 40: ≈ 0.6833.

---

#### `mira_score(crystal_mol, crystal_coords, sample_coords, template_mol, rot_bonds, ca_coords, num_runs=20, rng=None, timeout=4)`

MIRA score for a single complex using symmetry-corrected RMSD. See [the MIRA section](#mira) for the full algorithm.

**Args**

| Name | Type | Description |
|---|---|---|
| `crystal_mol` | RDKit `Mol` | Heavy-atom molecule (defines atom graph for symRMSD) |
| `crystal_coords` | `(N_atoms, 3)` ndarray | Crystal pose |
| `sample_coords` | list of `(N_atoms, 3)` ndarrays | S predicted poses |
| `template_mol` | RDKit `Mol` | ETKDG template from `prepare_reference_template` |
| `rot_bonds` | list of tuples | Rotatable bonds from `prepare_reference_template` |
| `ca_coords` | `(N_res, 3)` ndarray | Protein Cα coordinates |
| `num_runs` | int | Monte Carlo center draws (default 20) |
| `rng` | `numpy.random.Generator` | Created fresh if `None` |
| `timeout` | float | Per-call symRMSD timeout in seconds (default 4) |

**Returns** MIRA score (float), or `NaN` if fewer than 2 samples or all runs fail.

---

#### `bootstrap_mira(scores, n_bootstrap=500, rng=None)`

Bootstrap the MIRA mean and uncertainty by resampling complexes with replacement. Matches the aggregation approach in Sharief et al. 2026.

**Args**

| Name | Type | Description |
|---|---|---|
| `scores` | `(n_complexes,)` ndarray | Per-complex MIRA scores (NaN values are dropped) |
| `n_bootstrap` | int | Number of replicates (default 500) |
| `rng` | `numpy.random.Generator` | |

**Returns** `dict` with keys `'n'` (valid complexes), `'mean'`, `'std'` (bootstrap std of the mean), `'boot_means'` (raw bootstrap means).

---

#### `bootstrap_mira_groups(scores, group_labels, group_names, n_bootstrap=500, rng=None)`

Bootstrap per-group MIRA mean and 90% CI by resampling complexes within each group. Used for per-protein-family comparisons.

**Args**

| Name | Type | Description |
|---|---|---|
| `scores` | `(n_complexes,)` ndarray | Per-complex MIRA scores |
| `group_labels` | `(n_complexes,)` array | Group name string per complex |
| `group_names` | list of str | Ordered list of groups to include |
| `n_bootstrap` | int | Number of replicates (default 500) |
| `rng` | `numpy.random.Generator` | |

**Returns** `dict` mapping `group_name → {'n', 'mean', 'lo', 'hi', 'boot_means'}` where `lo`/`hi` are the 5th/95th bootstrap percentiles (90% CI).

---

### molcalib.tarp

Implements the TARP coverage test and ECP diagnostics (Lemos & Coogan et al. 2023).

---

#### `tarp_fractions(crystal_mol, crystal_coords, template_mol, rot_bonds, sample_coords, ca_coords, K, rng, mode='rmsd')`

Compute K TARP coverage fractions for a single complex. See [the TARP section](#tarp) for the full algorithm.

**Args**

| Name | Type | Description |
|---|---|---|
| `crystal_mol` | RDKit `Mol` | Heavy-atom molecule (defines atom graph for RMSD) |
| `crystal_coords` | `(N_atoms, 3)` ndarray | Crystal pose |
| `template_mol` | RDKit `Mol` | ETKDG template from `prepare_reference_template` |
| `rot_bonds` | list of tuples | Rotatable bonds from `prepare_reference_template` |
| `sample_coords` | list of `(N_atoms, 3)` ndarrays | S predicted poses |
| `ca_coords` | `(N_res, 3)` ndarray | Protein Cα coordinates |
| `K` | int | Number of random reference draws |
| `rng` | `numpy.random.Generator` | |
| `mode` | str | `'rmsd'` (symmetry-corrected, default) or `'centroid'` |

**Returns** `(≤K,)` numpy array with values in [0, 1]. May be shorter than K if any reference distances are non-finite.

---

#### `ecp_from_fractions(f_matrix, n_bins=50)`

Compute the Expected Coverage Probability (ECP) curve from a matrix of TARP fractions.

All fractions across complexes and reference draws are pooled into a single empirical CDF: `ECP(α) = fraction of f values ≤ α`. Under perfect calibration, `ECP(α) = α`.

**Args**

| Name | Type | Description |
|---|---|---|
| `f_matrix` | `(n_complexes, K)` ndarray | TARP fractions, or a flat 1-D array |
| `n_bins` | int | Number of α values (default 50) |

**Returns** `(ecp, alpha)` — two `(n_bins,)` numpy arrays; `alpha = linspace(0, 1, n_bins)`.

---

#### `bootstrap_ecp(f_matrix, n_bins=50, n_bootstrap=500, rng=None)`

Bootstrap confidence bands for the ECP by resampling **rows** of `f_matrix` with replacement. Resampling rows (complexes) rather than individual fractions correctly captures between-complex variability.

**Args**

| Name | Type | Description |
|---|---|---|
| `f_matrix` | `(n_complexes, K)` ndarray | TARP fractions |
| `n_bins` | int | Number of α values (default 50) |
| `n_bootstrap` | int | Number of replicates (default 500) |
| `rng` | `numpy.random.Generator` | |

**Returns** `(n_bootstrap, n_bins)` ndarray of bootstrap ECP curves. Pass to `np.percentile(..., [5, 95], axis=0)` to get a 90% confidence band.

---

#### `plot_ecp(ecp, alpha, ax=None, label=None, color=None, bootstrap_ecps=None, linestyle='solid')`

Plot an ECP curve with the perfect-calibration diagonal.

**Args**

| Name | Type | Description |
|---|---|---|
| `ecp` | `(n_bins,)` ndarray | ECP values |
| `alpha` | `(n_bins,)` ndarray | Credibility levels |
| `ax` | `matplotlib.axes.Axes` | Target axes; a new figure is created if `None` |
| `label` | str | Legend label |
| `color` | str | Line colour |
| `bootstrap_ecps` | `(n_bootstrap, n_bins)` ndarray | From `bootstrap_ecp`; draws a 90% shaded band |
| `linestyle` | str | matplotlib linestyle (default `'solid'`) |

**Returns** `matplotlib.axes.Axes`.

---

### molcalib.io

Model-agnostic helpers for loading ligand and protein structures. Model-specific loaders for DiffDock (`rank*.sdf`) and SigmaDock (`predictions.pt`) live in `eval_diffdock/loader.py` and `eval_sigmadock/` respectively.

---

#### `load_ligand_sdf(ligand_path, remove_hs=True)`

Load a heavy-atom RDKit `Mol` and all conformer coordinates from an SDF or mol2 file. When the file contains multiple records (e.g. crystallographic copies), the first record is the canonical molecule and all conformer coordinates are returned.

**Args**

| Name | Type | Description |
|---|---|---|
| `ligand_path` | str or Path | Path to `.sdf` or `.mol2` file |
| `remove_hs` | bool | Remove hydrogens (default `True`) |

**Returns** `(mol, all_coords)` — RDKit `Mol` and list of `(N_atoms, 3)` arrays, one per conformer.

**Raises** `FileNotFoundError` / `ValueError` if the file is missing or unparseable.

---

#### `load_protein_ca_coords(protein_pdb_path)`

Load Cα coordinates from a PDB file via ProDy.

**Args**

| Name | Type | Description |
|---|---|---|
| `protein_pdb_path` | str or Path | Path to `.pdb` file |

**Returns** `(N_residues, 3)` numpy array.

**Raises** `FileNotFoundError` if the file is missing.

---

## End-to-end example

### Single complex

```python
import numpy as np
import matplotlib.pyplot as plt
from molcalib import (
    load_ligand_sdf, load_protein_ca_coords,
    prepare_reference_template,
    mira_null, mira_score,
    tarp_fractions, ecp_from_fractions, bootstrap_ecp, plot_ecp,
)

# Load structures
crystal_mol, crystal_coords_list = load_ligand_sdf("crystal.sdf")
crystal_coords = crystal_coords_list[0]          # (N_atoms, 3)

_, sample_coords = load_ligand_sdf("poses.sdf")  # list of S (N_atoms, 3) arrays
ca_coords = load_protein_ca_coords("protein.pdb")

# Build prior template once per complex
template_mol, rot_bonds = prepare_reference_template(crystal_mol)
rng = np.random.default_rng(42)

# MIRA
score = mira_score(
    crystal_mol, crystal_coords, sample_coords,
    template_mol, rot_bonds, ca_coords,
    num_runs=100, rng=rng,
)
null = mira_null(S=len(sample_coords))
print(f"MIRA = {score:.4f}  (null ≈ {null:.4f})")
# score > null  →  over-dispersed
# score < null  →  mode-collapsed

# TARP
fracs = tarp_fractions(
    crystal_mol, crystal_coords,
    template_mol, rot_bonds,
    sample_coords, ca_coords,
    K=20, rng=rng, mode="rmsd",
)                                       # (≤20,)

f_matrix = fracs[np.newaxis, :]        # (1, K) for ecp_from_fractions
ecp, alpha = ecp_from_fractions(f_matrix)

fig, ax = plt.subplots(figsize=(5, 5))
plot_ecp(ecp, alpha, ax=ax, label="My model")
plt.tight_layout()
plt.savefig("ecp.png", dpi=150)
```

### Full test set with batch runners

For DiffDock or SigmaDock inference results, use the parallelised batch runners in `eval_diffdock` / `eval_sigmadock`. These handle the model-specific output format and multiprocessing.

```python
import numpy as np
from eval_diffdock.loader import build_results_index
from eval_diffdock.tarp_runner import run_tarp_eval
from eval_diffdock.mira_runner import compute_mira_scores
from molcalib.tarp import ecp_from_fractions, bootstrap_ecp, plot_ecp
from molcalib.mira import mira_null

results_index = build_results_index("results/DiffDock/pdbbind_testset/raw_chunks")
complex_names = np.load("results/DiffDock/pdbbind_testset/metrics/complex_names.npy",
                        allow_pickle=True)
DATA_DIR = "data/PDBBind_processed"

# TARP — shape (n_complexes, K)
f_matrix = run_tarp_eval(
    complex_names, results_index, DATA_DIR,
    K=10, mode="rmsd", seed=42, verbose=True, n_workers=8,
)
np.save("tarp_fractions_symrmsd_K10.npy", f_matrix)

ecp, alpha = ecp_from_fractions(f_matrix)
boot = bootstrap_ecp(f_matrix, n_bootstrap=500)   # 90% CI

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))
plot_ecp(ecp, alpha, ax=ax, bootstrap_ecps=boot, label="DiffDock")
plt.savefig("ecp_diffdock.png", dpi=150)

# MIRA — shape (n_complexes,)
names, scores = compute_mira_scores(
    complex_names, results_index, DATA_DIR,
    num_runs=100, metric="symrmsd", seed=42, verbose=True, n_workers=8,
)
np.save("mira_scores_symrmsd.npy", scores)
print(f"Mean MIRA = {scores.mean():.4f}  (null = {mira_null(40):.4f})")
```

Pre-computed `.npy` arrays are what the analysis notebooks load directly — no re-running of the batch evaluation is needed to reproduce the figures.
