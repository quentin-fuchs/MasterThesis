#!/bin/bash -l
#SBATCH --job-name=setup-mira-tarp
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/qf226/MProject/sigmadock/logs/setup_mira_tarp_%j.out
#SBATCH --error=/home/qf226/MProject/sigmadock/logs/setup_mira_tarp_%j.err

# Create the mira-tarp conda environment.
# Run once with: sbatch ~/slurm/sigmadock/setup_mira_tarp_env.sh

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Creating mira-tarp conda environment..."
conda create -n mira-tarp python=3.10 -y

conda activate mira-tarp

# Core scientific stack — use conda-forge for rdkit (avoids ABI mismatches)
conda install -y -c conda-forge rdkit networkx matplotlib numpy scipy

# Packages without good conda builds
pip install \
  mira-score==0.1.7 \
  tarp==0.1.1 \
  torch \
  spyrmsd \
  prody

echo ""
echo "mira-tarp environment ready."
conda run -n mira-tarp python -c "
import mira_score, tarp, torch, rdkit, spyrmsd, prody, networkx, matplotlib
print('mira-score:', mira_score.__version__)
print('tarp:       ', tarp.__version__)
print('torch:      ', torch.__version__)
print('OK')
"
