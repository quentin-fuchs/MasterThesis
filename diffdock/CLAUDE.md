# CLAUDE.md — DiffDock

Shared HPC, data, and project-wide guidance is in `../CLAUDE.md` (loaded automatically).

## Project overview

Fork of DiffDock (diffusion-based molecular docking), used for Quentin's MPhil thesis. The goal is to understand and extend DiffDock's inference behaviour — generating, visualising, and analysing predicted ligand poses. The end goal is applying the PQMass framework to test whether DiffDock's output distribution is well-captured, potentially using energy simulation as a ground-truth distribution.

- Working directory: `/home/qf226/MProject/DiffDock`
- Conda environment: `diffdock`

## SLURM scripts

Scripts live in `~/slurm/DiffDock/`. Submit with `sbatch ~/slurm/DiffDock/<script>.sh`.

| Script | Purpose |
|--------|---------|
| `diffdock_6d08_100.sh` | Run inference on PDB 6d08, 100 samples, no final-step noise |
| `diffdock_inference.sh` | Generic batch inference via CSV; usage: `sbatch ... <csv> [out_dir]` |
| `diffdock_rmsd_eval.sh` | symRMSD accuracy eval on PoseBusters benchmark → `results/posebusters_inference/metrics/` |
| `diffdock_posebusters.sh` | PB validity filtering on PDBBind test set → `results/testset_eval_merged/posebusters_results.json` |
| `diffdock_pb_eval_posebusters.sh` | PB validity filtering on PoseBusters benchmark → `results/posebusters_inference/metrics/posebusters_results_pb.json` |

## Inference

Run single complex:
```bash
python inference.py --config default_inference_args.yaml \
    --protein_path <path>.pdb --ligand_description <path>.sdf \
    --out_dir results/<run_name> --samples_per_complex 100
```

Run from CSV:
```bash
python inference.py --config default_inference_args.yaml \
    --protein_ligand_csv data/protein_ligand_example.csv \
    --out_dir results/user_predictions_small
```

Run with precomputed ESM embeddings (required for batch eval):
```bash
python inference.py ... \
    --esm_embeddings_path ~/rds/hpc-work/data/embeddings/pdbbind_esm2.pt
```

- Default config: `default_inference_args.yaml` (score model v1.1, 20 steps, confidence model epoch 75)
- Benchmark data: `~/rds/hpc-work/data/PDBBind_processed/<pdb_id>/`
- Caches: `~/rds/hpc-work/data/cache_*/`
- Results: `results/<run_name>/<complex_name>/` — contains ranked SDFs (`rank1.sdf`, `rank1_confidence<score>.sdf`, …) and `input_metadata.json`
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
| `utils/molecules_utils.py` | Ligand/PDB utilities |
| `notebooks/plot_inference_results.ipynb` | Plotting and animation of diffusion trajectories |

## Gotchas

- Multi-fragment ligands previously caused a torsion assertion error; fixed in commit `c0d7779`
- SO(2)/SO(3) look-up tables are precomputed on first run and cached as `.npy` files in the project root — takes a few minutes, done once
- Confidence score thresholds: `c > 0` high, `-1.5 < c < 0` moderate, `c < -1.5` low (valid for drug-like small molecules on medium-sized proteins)
- **`--config` yaml overrides CLI args, not the other way around.** `evaluate.py` (and `inference.py`) call `parser.parse_args()` first, then load the yaml and unconditionally overwrite `args.__dict__`. Any key present in the yaml silently wins over the CLI flag. To change a value that's in the yaml, edit the yaml (or use a separate yaml) — passing a CLI flag for it does nothing.
