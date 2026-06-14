"""
Copy the 100-complex sensitivity whitelist from pb_evaluate_v2_merged into
sensitivity_ode_nsteps_v2/sde_20/ so that all conditions share the same
directory layout and the sde_20 baseline can be indexed by build_flat_index.

Safe to run directly on the login node — pure file I/O, no computation.
"""

import shutil
from pathlib import Path

DIFFDOCK_DIR = Path("/home/qf226/MProject/DiffDock")
RDS          = Path("/home/qf226/rds/hpc-work")

WHITELIST = DIFFDOCK_DIR / "data/splits/pb_sensitivity_rand100.txt"
SRC_DIR   = RDS / "results/DiffDock/pb_evaluate_v2_merged/poses"
DST_DIR   = RDS / "results/DiffDock/sensitivity_ode_nsteps_v2/sde_20"

DST_DIR.mkdir(parents=True, exist_ok=True)

names   = WHITELIST.read_text().split()
copied  = []
missing = []

for name in names:
    src = SRC_DIR / name
    dst = DST_DIR / name
    if not src.exists():
        missing.append(name)
        continue
    shutil.copytree(src, dst, dirs_exist_ok=True)
    copied.append(name)

print(f"Copied : {len(copied)}/{len(names)}")
print(f"Missing: {len(missing)}")
if missing:
    print("  " + "\n  ".join(missing))
