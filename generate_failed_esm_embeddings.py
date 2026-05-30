#!/usr/bin/env python3
"""
Generate ESM2-650M embeddings for the 24 posebusters complexes that fail with
precomputed embeddings due to tensor size mismatches from modified residues.

Uses get_sequences_from_pdbfile (ProDy-based, same as graph construction) so
the resulting embeddings are guaranteed to match the receptor feature dimension.

Usage:
    python generate_failed_esm_embeddings.py \
        --split_path data/splits/pb_failed_eval.txt \
        --data_dir   data/posebusters_benchmark_set \
        --output_pt  data/posebusters_failed_esm2_embeddings.pt
"""

import sys
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.inference_utils import get_sequences_from_pdbfile, compute_ESM_embeddings
from esm import pretrained

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--split_path', type=str, default='data/splits/pb_failed_eval.txt')
parser.add_argument('--data_dir',   type=str, default='data/posebusters_benchmark_set')
parser.add_argument('--output_pt',  type=str, default='data/posebusters_failed_esm2_embeddings.pt')
args = parser.parse_args()

with open(args.split_path) as f:
    names = [l.strip() for l in f if l.strip()]
print(f'Loaded {len(names)} complex names from {args.split_path}')

data_dir = Path(args.data_dir)

labels, sequences, valid_names = [], [], []
skipped = []
for name in names:
    pdb_path = data_dir / name / f'{name}_protein.pdb'
    if not pdb_path.exists():
        print(f'  {name}: protein PDB not found — skipping')
        skipped.append(name)
        continue
    try:
        seq_str = get_sequences_from_pdbfile(str(pdb_path))  # chains joined by ':'
        chains = seq_str.split(':')
        for j, chain_seq in enumerate(chains):
            labels.append(f'{name}_chain_{j}')
            sequences.append(chain_seq)
        valid_names.append(name)
        print(f'  {name}: {len(chains)} chain(s), lengths {[len(c) for c in chains]}')
    except Exception as e:
        print(f'  {name}: ProDy parsing failed — {e}')
        skipped.append(name)

print(f'\nSkipped {len(skipped)}: {skipped}')
print(f'Running ESM2-650M on {len(labels)} chain sequences...')

model, alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
    print('Using GPU')
else:
    print('WARNING: no GPU found, running on CPU (slow)')

embeddings = compute_ESM_embeddings(model, alphabet, labels, sequences)

print(f'\nGenerated {len(embeddings)} chain embeddings')
torch.save(embeddings, args.output_pt)
print(f'Saved to {args.output_pt}')
