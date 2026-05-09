# DiffDock — Claude Code Guide

## Project overview
This is a fork of DiffDock (diffusion-based molecular docking), used for Quentin's MPhil thesis. The goal is to understand and extend DiffDock's inference behaviour — generating, visualising, and analysing predicted ligand poses. Eventually we'll use the PQMass framework to test if the distribution is well captured in Diffdock. Perhaps utilize a energy simulation as a ground truth distribution. 

The working directory is `/home/qf226/MProject/DiffDock`. The conda environment is `diffdock`.

## HPC / SLURM
- Cluster: Cambridge HPC (login node `login-q-3`)
- Account: `fergusson-sl3-gpu`
- Partition: `ampere` (not `MPHIL-DIS-SL2-GPU` — that one is invalid on this cluster)
- All SLURM scripts live in `~/slurm/` (i.e. `/home/qf226/slurm/`)
- Submit with: `sbatch ~/slurm/<script>.sh`
- Logs go to `logs/diffdock_<jobid>.out` / `.err` inside the project directory

### Current scripts
| Script | Purpose |
|--------|---------|
| `diffdock_6d08_100.sh` | Run inference on PDB 6d08, 100 samples, no final-step noise |
| `diffdock_inference.sh` | Generic batch inference via CSV; usage: `sbatch ~/slurm/diffdock_inference.sh <csv> [out_dir]` |

## New Code
All code should be written quite generic and reusable. Always commit changes. 
Include a docstring with a clear description of the functionality, Args and Returns. 


## Inference
- Entry point: `inference.py`
- Default config: `default_inference_args.yaml` (score model v1.1, 20 steps, confidence model epoch 75)
- Data: `data/PDBBind_processed/<pdb_id>/`
- Results go to `results/<run_name>/`; every run produces a `metadata.json` alongside pose files

## Notion experiment log
- Parent page: **MPhil Project** — ID `2f68713e1a2480768e80f9640f227607`
- For every experiment, create a **subpage** under this parent page
- Each subpage should cover: what was run (parameters, script, PDB), what the results show, and what it tells us about DiffDock's pose distribution in the context of eventually applying PQMass to test whether the distribution is well-captured
- Integration name: `diffdock-claude`

## Key source files
- `utils/visualise.py` — visualisation helpers
- `utils/inference_utils.py` — inference pipeline
- `utils/molecules_utils.py` — ligand/PDB utilities (multi-fragment fix landed in c0d7779)
- `notebooks/plot_inference_results.ipynb` — plotting & animation of diffusion trajectories
- `utils/plot_inference_results.ipynb` — also present in utils/

## Gotchas
- Multi-fragment ligands previously caused a torsion assertion error; fixed in commit `c0d7779`
