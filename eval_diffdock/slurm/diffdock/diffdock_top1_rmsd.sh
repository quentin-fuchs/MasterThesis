#!/bin/bash
#SBATCH --job-name=diffdock_top1_rmsd
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/qf226/MProject/DiffDock/logs/diffdock_top1_rmsd_%j.out
#SBATCH --error=/home/qf226/MProject/DiffDock/logs/diffdock_top1_rmsd_%j.err

DIFFDOCK_DIR=/home/qf226/MProject/DiffDock
export RDS=/home/qf226/rds/hpc-work/data

source ~/.bashrc
conda activate diffdock
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIFFDOCK_DIR:$PYTHONPATH"

cd "$DIFFDOCK_DIR"

echo "Running top-1 RMSD evaluation..."

python - <<'EOF'
import warnings, sys, time, os
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
import numpy as np
from rdkit import Chem
from spyrmsd import rmsd as spyrmsd_rmsd, molecule as spyrmsd_molecule

sys.path.insert(0, '.')
from utils.tarp_eval import build_results_index, load_crystal_coords, load_protein_ca_coords

DATA_DIR   = os.path.join(os.environ["RDS"], "PDBBind_processed")
RESULTS_DIR = "results/testset_eval_full"
OUT_PATH   = "results/testset_eval_merged/top1_rmsd.npy"
TIMEOUT    = 10  # seconds per spyrmsd call

complex_names = np.load("results/testset_eval_merged/complex_names.npy", allow_pickle=True)
results_index = build_results_index(RESULTS_DIR)

def symmrmsd_timeout(ref, query, atomicnums, adjacency, timeout=TIMEOUT):
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(spyrmsd_rmsd.symmrmsd, ref, query,
                           atomicnums, atomicnums, adjacency, adjacency)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            return np.nan

rmsds = []
skipped = 0
for i, pdb_id in enumerate(complex_names):
    if i % 20 == 0:
        print(f"  [{i}/{len(complex_names)}] {pdb_id} ...", flush=True)

    try:
        crystal_mol, crystal_coords = load_crystal_coords(pdb_id, DATA_DIR)
    except Exception as e:
        rmsds.append(np.nan); skipped += 1; continue

    if pdb_id not in results_index:
        rmsds.append(np.nan); skipped += 1; continue

    rank1 = results_index[pdb_id] / "rank1.sdf"
    if not rank1.exists():
        rmsds.append(np.nan); skipped += 1; continue

    pred_mol = Chem.SDMolSupplier(str(rank1), removeHs=True)[0]
    if pred_mol is None or pred_mol.GetNumConformers() == 0:
        rmsds.append(np.nan); skipped += 1; continue

    pred_coords = pred_mol.GetConformer().GetPositions()
    if pred_coords.shape != crystal_coords.shape:
        rmsds.append(np.nan); skipped += 1; continue

    spy = spyrmsd_molecule.Molecule.from_rdkit(crystal_mol)
    r = symmrmsd_timeout(crystal_coords, pred_coords, spy.atomicnums, spy.adjacency_matrix)
    rmsds.append(r)

rmsds = np.array(rmsds)
np.save(OUT_PATH, rmsds)

valid = rmsds[np.isfinite(rmsds)]
print(f"\n=== Top-1 RMSD results ===")
print(f"Complexes: {len(valid)} valid, {skipped} skipped")
print(f"Median RMSD:      {np.median(valid):.2f} Å")
print(f"Mean RMSD:        {np.mean(valid):.2f} Å")
print(f"Fraction < 2 Å:   {(valid < 2).mean():.3f}")
print(f"Fraction < 5 Å:   {(valid < 5).mean():.3f}")
print(f"Saved to {OUT_PATH}")
EOF
