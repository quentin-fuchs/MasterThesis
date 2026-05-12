# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a fork of DiffDock (diffusion-based molecular docking), used for Quentin's MPhil thesis. The goal is to understand and extend DiffDock's inference behaviour — generating, visualising, and analysing predicted ligand poses. The end goal is applying the PQMass framework to test whether DiffDock's output distribution is well-captured, potentially using energy simulation as a ground-truth distribution.

Working directory: `/home/qf226/MProject/DiffDock`. Conda environment: `diffdock`.

## HPC / SLURM

- Cluster: Cambridge HPC (login node `login-q-3`)
- Account: `fergusson-sl3-gpu`
- Partition: `ampere`
- All SLURM scripts live in `~/slurm/` (`/home/qf226/slurm/`)
- Submit with: `sbatch ~/slurm/<script>.sh`
- Logs: `logs/diffdock_<jobid>.out` / `.err` inside the project directory

### Current scripts

| Script | Purpose |
|--------|---------|
| `diffdock_6d08_100.sh` | Run inference on PDB 6d08, 100 samples, no final-step noise |
| `diffdock_inference.sh` | Generic batch inference via CSV; usage: `sbatch ~/slurm/diffdock_inference.sh <csv> [out_dir]` |

## Inference

Run single complex:
```bash
python inference.py --config default_inference_args.yaml \
    --protein_path <path>.pdb --ligand_description <path>.sdf \
    --out_dir results/<run_name> --samples_per_complex 100
```

Run from CSV:
```bash
python -m inference --config default_inference_args.yaml \
    --protein_ligand_csv data/protein_ligand_example.csv \
    --out_dir results/user_predictions_small
```

- Default config: `default_inference_args.yaml` (score model v1.1, 20 steps, confidence model epoch 75)
- Data: `data/PDBBind_processed/<pdb_id>/`
- Results go to `results/<run_name>/<complex_name>/`; each complex directory contains ranked SDF files (`rank1.sdf`, `rank1_confidence<score>.sdf`, …) and `input_metadata.json`
- Add `--save_visualisation` to also write `rank*_reverseprocess.pdb` for diffusion trajectory animation

## Architecture

DiffDock models diffusion over the **product space of ligand poses**: translation (R³), rotation (SO(3)), and torsion angles (torus T^k). The score model predicts denoising updates for all three; a separate confidence model ranks the resulting poses.

Key model: `models/aa_model.py` — `AAModel`, an E3-equivariant graph neural network built with `e3nn` tensor product layers. Inputs are heterogeneous graphs with three node types:
- `receptor` — Cα atoms with ESM2 language-model embeddings
- `atom` — all-atom receptor representation (used in `all_atoms` mode)
- `ligand` — ligand heavy atoms with RDKit features

The graph is constructed in `datasets/process_mols.py` and wrapped in `utils/inference_utils.py:InferenceDataset`.

Diffusion data flow:
1. `inference.py` — loads models, builds `InferenceDataset`, runs the loop
2. `utils/sampling.py:sampling` — iterates the reverse diffusion schedule calling the score model
3. `utils/diffusion_utils.py` — `t_to_sigma` maps timestep → noise σ for each DOF; `modify_conformer` applies updates
4. `utils/torsion.py` — torsion angle updates on ligand bonds

## Key source files

| File | Purpose |
|------|---------|
| `inference.py` | Main inference entry point |
| `train.py` | Score-model training loop |
| `evaluate.py` | Benchmark evaluation (PDBBind / DockGen / PoseBusters) |
| `default_inference_args.yaml` | Default model paths and noise-schedule hyperparameters |
| `models/aa_model.py` | All-atom E3-equivariant score / confidence model |
| `models/cg_model.py` | Coarse-grained variant |
| `models/tensor_layers.py` | `TensorProductConvLayer` building block |
| `datasets/process_mols.py` | Molecule featurization, graph construction, conformer generation |
| `datasets/pdbbind.py` / `moad.py` | Dataset loaders for PDBBind and BindingMOAD |
| `utils/inference_utils.py` | `InferenceDataset`, ESM embedding computation, `save_run_metadata` |
| `utils/diffusion_utils.py` | Noise schedules, `t_to_sigma`, `modify_conformer` |
| `utils/sampling.py` | Reverse diffusion loop (`sampling`, `randomize_position`) |
| `utils/visualise.py` | `view_inference_results`, `view_diffusion_animation`, `plot_diffusion_frames`, `load_results_dir` |
| `utils/distribution_analysis.py` | Pose-distribution analysis: RMSD matrices, clustering, torsion rose plots, saturation analysis |
| `utils/molecules_utils.py` | Ligand/PDB utilities (multi-fragment fix landed in `c0d7779`) |
| `notebooks/plot_inference_results.ipynb` | Plotting and animation of diffusion trajectories |

## Notion experiment log

- Parent page: **MPhil Project** — ID `2f68713e1a2480768e80f9640f227607`
- For every experiment, create a **subpage** under this parent page
- Each subpage should cover: what was run (parameters, script, PDB), what the results show, and what it tells us about DiffDock's pose distribution in the context of applying PQMass
- Integration name: `diffdock-claude`

## New code guidelines

All new code should be generic and reusable. Include a docstring with a clear description of the functionality, Args, and Returns. Always commit changes.

## Explanation Levels

I have basically zero biology knowledge. Always explain any processes in an easy language thoroughly.

## Gotchas

- Multi-fragment ligands previously caused a torsion assertion error; fixed in commit `c0d7779`
- SO(2)/SO(3) look-up tables are precomputed on first run and cached as `.npy` files in the project root — this takes a few minutes but is only done once
- The confidence score thresholds: `c > 0` high, `-1.5 < c < 0` moderate, `c < -1.5` low (valid for drug-like small molecules on medium-sized proteins)
