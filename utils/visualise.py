from rdkit.Chem.rdmolfiles import MolToPDBBlock, MolToPDBFile
from rdkit.Chem import SDMolSupplier
import rdkit.Chem 
from rdkit import Geometry
from collections import defaultdict
import copy
import glob
import json
import os
import re
import numpy as np
import torch

    
class PDBFile:
    def __init__(self, mol):
        self.parts = defaultdict(dict)
        self.mol = copy.deepcopy(mol)
        [self.mol.RemoveConformer(j) for j in range(mol.GetNumConformers()) if j]        
    def add(self, coords, order, part=0, repeat=1):
        if type(coords) in [rdkit.Chem.Mol, rdkit.Chem.RWMol]:
            block = MolToPDBBlock(coords).split('\n')[:-2]
            self.parts[part][order] = {'block': block, 'repeat': repeat}
            return
        elif type(coords) is np.ndarray:
            coords = coords.astype(np.float64)
        elif type(coords) is torch.Tensor:
            coords = coords.double().numpy()
        for i in range(coords.shape[0]):
            self.mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
        block = MolToPDBBlock(self.mol).split('\n')[:-2]
        self.parts[part][order] = {'block': block, 'repeat': repeat}
        
    def write(self, path=None, limit_parts=None):
        is_first = True
        str_ = ''
        for part in sorted(self.parts.keys()):
            if limit_parts and part >= limit_parts:
                break
            part = self.parts[part]
            keys_positive = sorted(filter(lambda x: x >=0, part.keys()))
            keys_negative = sorted(filter(lambda x: x < 0, part.keys()))
            keys = list(keys_positive) + list(keys_negative)
            for key in keys:
                block = part[key]['block']
                times = part[key]['repeat']
                for _ in range(times):
                    if not is_first:
                        block = [line for line in block if 'CONECT' not in line]
                    is_first = False
                    str_ += 'MODEL\n'
                    str_ += '\n'.join(block)
                    str_ += '\nENDMDL\n'
        if not path:
            return str_
        with open(path, 'w') as f:
            f.write(str_)


def mol_to_pdb_block(mol):
    """Convert an RDKit Mol (with 3D coords) to a multi-model PDB block string."""
    pdb = PDBFile(mol)
    pdb.add(mol, order=0)
    return pdb.write()


def load_results_dir(results_dir):
    """Discover ligand SDFs and receptor PDB from a DiffDock output directory.

    Reads ``input_metadata.json`` (written by ``save_run_metadata``) to locate
    the receptor PDB.  If the protein was copied into the output dir it is used
    directly; otherwise the original path from the metadata is tried.

    Ranked SDF files are sorted by rank number.

    Parameters
    ----------
    results_dir : str
        Path to a single complex output directory
        (e.g. ``results/6w70_run/complex_0``).

    Returns
    -------
    dict
        ``{"receptor_pdb": str | None, "ligand_sdfs": [str, ...], "metadata": dict | None}``
    """
    results_dir = os.path.abspath(results_dir)
    metadata = None
    receptor_pdb = None

    # --- Read metadata -------------------------------------------------------
    meta_path = os.path.join(results_dir, "input_metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # --- Locate receptor PDB -------------------------------------------------
    # 1) Check for any .pdb file copied into the results dir
    local_pdbs = sorted(glob.glob(os.path.join(results_dir, "*.pdb")))
    # Exclude reverse-process visualisation PDBs
    local_pdbs = [p for p in local_pdbs if "reverseprocess" not in os.path.basename(p)]
    if local_pdbs:
        receptor_pdb = local_pdbs[0]
    elif metadata and metadata.get("protein_path"):
        # 2) Fall back to the original path recorded in metadata
        orig = metadata["protein_path"]
        if os.path.isfile(orig):
            receptor_pdb = orig

    # --- Collect ranked SDF files --------------------------------------------
    all_sdfs = sorted(glob.glob(os.path.join(results_dir, "rank*.sdf")))

    def _rank_key(path):
        """Extract the rank number for sorting."""
        m = re.search(r"rank(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 999

    all_sdfs.sort(key=_rank_key)

    return {
        "receptor_pdb": receptor_pdb,
        "ligand_sdfs": all_sdfs,
        "metadata": metadata,
    }


def view_inference_results(ligand_sdf=None, receptor_pdb=None, results_dir=None,
                           model_indices=None, ranks=None,
                           width=800, height=600,
                           ligand_style=None, receptor_style=None, show_surface=False,
                           surface_opacity=0.15, zoom_to_ligand=True):
    """Render 3D docking results using py3Dmol.

    Can be called in two ways:

    1. **Explicit paths** — pass ``ligand_sdf`` (and optionally ``receptor_pdb``).
    2. **Results directory** — pass ``results_dir`` pointing to a DiffDock output
       folder (e.g. ``results/6w70_run/complex_0``).  The function reads
       ``input_metadata.json`` to find the receptor PDB and discovers ranked
       SDF files automatically.

    Parameters
    ----------
    ligand_sdf : str, optional
        Path to a single SDF file (may contain multiple poses).
        Ignored when *results_dir* is provided.
    receptor_pdb : str, optional
        Path to a receptor PDB file.
        Ignored when *results_dir* is provided (read from metadata instead).
    results_dir : str, optional
        Path to a DiffDock output directory.  When given, ``ligand_sdf`` and
        ``receptor_pdb`` are discovered automatically.
    model_indices : Iterable[int], optional
        Zero-based indices into the flat list of poses to render.
        When using ``results_dir``, poses are numbered sequentially across
        all ranked SDF files (rank1 → index 0, rank2 → index 1, …).
        Ignored when a single ``ligand_sdf`` is provided (assumed to contain
        one pose).
    ranks : Iterable[int], optional
        1-based rank numbers to display when using ``results_dir``
        (e.g. ``[1, 2, 3]``).  If None, all discovered ranks are shown.
    width, height : int
        Canvas size in pixels.
    ligand_style : dict, optional
        py3Dmol style dict for ligand models.
    receptor_style : dict, optional
        py3Dmol style dict for receptor model.
    show_surface : bool
        If True, add a translucent molecular surface around the receptor.
    surface_opacity : float
        Opacity of the receptor surface (only used when show_surface=True).
    zoom_to_ligand : bool
        If True (and a receptor is present), zoom into the ligand binding site.

    Returns
    -------
    py3Dmol.view
        An interactive 3D viewer object.
    """
    try:
        import py3Dmol
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "py3Dmol is required for 3D rendering; install with `pip install py3Dmol`."
        ) from exc

    # ---- Resolve inputs from results_dir if provided ------------------------
    sdf_files = []  # list of (sdf_path, rank_label)
    if results_dir is not None:
        info = load_results_dir(results_dir)
        receptor_pdb = info["receptor_pdb"]
        rank_filter = set(ranks) if ranks is not None else None
        for sdf_path in info["ligand_sdfs"]:
            m = re.search(r"rank(\d+)", os.path.basename(sdf_path))
            rank_num = int(m.group(1)) if m else None
            if rank_filter is not None and rank_num not in rank_filter:
                continue
            sdf_files.append(sdf_path)
    elif ligand_sdf is not None:
        sdf_files.append(ligand_sdf)
    else:
        raise ValueError("Provide either 'ligand_sdf' or 'results_dir'.")

    viewer = py3Dmol.view(width=width, height=height)
    viewer.setBackgroundColor('white')

    # ---- Receptor (protein) ------------------------------------------------
    receptor_model_idx = None
    if receptor_pdb is not None:
        with open(receptor_pdb, 'r', encoding='utf-8') as handle:
            viewer.addModel(handle.read(), 'pdb')
        receptor_model_idx = 0

        default_receptor_style = {
            'cartoon': {'color': 'spectrum', 'opacity': 0.85},
        }
        viewer.setStyle(
            {'model': receptor_model_idx},
            receptor_style or default_receptor_style,
        )

        if show_surface:
            viewer.addSurface(
                py3Dmol.VDW,
                {'opacity': surface_opacity, 'color': 'white'},
                {'model': receptor_model_idx},
            )

    # ---- Ligand pose(s) -----------------------------------------------------
    # Colour palette so multiple ranked poses are visually distinguishable
    _colours = [
        '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    ]

    default_ligand_style = lambda colour: {
        'stick': {'radius': 0.2, 'color': colour},
        'sphere': {'radius': 0.4, 'color': colour, 'opacity': 0.6},
    }

    pose_filter = set(model_indices) if model_indices is not None else None
    ligand_count = 0       # total poses added to viewer
    global_pose_idx = 0    # running index across all SDF files
    for sdf_path in sdf_files:
        suppl = SDMolSupplier(sdf_path, removeHs=False)
        for idx, mol in enumerate(suppl):
            if mol is None:
                global_pose_idx += 1
                continue
            if pose_filter is not None and global_pose_idx not in pose_filter:
                global_pose_idx += 1
                continue
            colour = _colours[ligand_count % len(_colours)]
            pdb_block = mol_to_pdb_block(mol)
            viewer.addModel(pdb_block, 'pdb')
            viewer.setStyle(
                {'model': -1},
                ligand_style or default_ligand_style(colour),
            )
            ligand_count += 1
            global_pose_idx += 1

    # ---- Camera -------------------------------------------------------------
    if zoom_to_ligand and receptor_pdb is not None and ligand_count > 0:
        # Zoom to the last-added ligand so the binding site is centred
        viewer.zoomTo({'model': -1})
    else:
        viewer.zoomTo()

    return viewer