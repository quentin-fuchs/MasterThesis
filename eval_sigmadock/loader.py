"""SigmaDock-specific data loading utilities.

Loads predicted poses from SigmaDock's seed_*/predictions.pt files.
"""

from pathlib import Path

import torch


def load_sigmadock_poses(model_dir):
    """Load all predicted poses from seed_*/predictions.pt files.

    Args:
        model_dir: path containing seed_* subdirectories with predictions.pt.

    Returns:
        dict mapping complex_id → (lig_ref_mol, list[np.ndarray (N,3)]).
        Crystal coordinates are in lig_ref.GetConformer().GetPositions().

    Raises:
        FileNotFoundError: if no seed_*/predictions.pt files are found.
    """
    from sigmadock.chem.statistics import get_mol_from_coords

    model_dir = Path(model_dir)
    seed_dirs = sorted(
        [p for p in model_dir.glob("seed_*") if (p / "predictions.pt").exists()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_*/predictions.pt in {model_dir}")

    poses, ref_mols = {}, {}
    for seed_dir in seed_dirs:
        pt = torch.load(seed_dir / "predictions.pt", weights_only=False)
        for complex_id, samples in pt["results"].items():
            sample = samples[0]
            lig_ref = sample["lig_ref"]
            x0_hat  = sample["x0_hat"]
            pred_mol = get_mol_from_coords(x0_hat, lig_ref)
            coords   = pred_mol.GetConformer().GetPositions()
            if complex_id not in poses:
                poses[complex_id]    = []
                ref_mols[complex_id] = lig_ref
            poses[complex_id].append(coords)

    return {cid: (ref_mols[cid], poses[cid]) for cid in sorted(poses)}
