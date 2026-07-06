# SLURM Scripts

These scripts were used to submit jobs on the Cambridge CSD3 HPC cluster (icelake/ampere partitions). They are kept here as exemplary references for how the evaluation pipeline was executed — they are **not portable as-is**.

All paths, account names, partition names, and conda environment paths are hardcoded for the CSD3 setup used during the MPhil project (`user: qf226`, `THESIS_DIR: /home/qf226/MProject/thesis`, data on RDS at `/home/qf226/rds/hpc-work`). Adapting them to a different cluster requires updating these variables at the top of each script.

## Scripts

| Script | Purpose |
|---|---|
| `diffdock/diffdock_eval_testset.sh` | Run DiffDock `evaluate.py` on the PDBBind test set (GPU, ampere) |
| `diffdock/diffdock_tarp_rmsd.sh` | TARP RMSD evaluation over the PDBBind test set (CPU, icelake) |
| `diffdock/diffdock_mira_symrmsd.sh` | MIRA with symmetry-corrected RMSD (CPU, icelake) |
| `diffdock/diffdock_posebusters.sh` | PoseBusters validity filtering for DiffDock predictions (CPU, icelake) |

## Usage on CSD3

```bash
sbatch slurm/diffdock/diffdock_eval_testset.sh
sbatch slurm/diffdock/diffdock_tarp_rmsd.sh
sbatch slurm/diffdock/diffdock_mira_symrmsd.sh
sbatch slurm/diffdock/diffdock_posebusters.sh
```

Logs are written to `$THESIS_DIR/logs/`.
