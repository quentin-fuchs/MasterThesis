## DiffDock inference
From the repository root
```bash
conda env create -f diffdock/environment.yml -n diffdock_inference
conda run -n diffdock_inference pip install "openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307"
```

---

## SigmaDock inference — `sigmadock_inference` (CSD3)

CSD3 has glibc 2.28 / GLIBCXX 3.4.25. Use Python 3.10 and cu118; newer builds need glibc ≥ 2.29.

```bash
conda create -y -n sigmadock_inference python=3.10
conda activate sigmadock_inference
cd sigmadock
bash install.sh cu118 train,test
```

`train` provides hydra-core, omegaconf, posebusters. `test` provides spyrmsd. Both are required — `sample.py` imports `statistics.py` which imports spyrmsd at module level.

### gnina for rescoring

gnina is **not on conda-forge** (the package does not exist there). The GitHub binary releases v1.3+ require glibc 2.29+ and are incompatible with CSD3.

Use the v1.1 binary (Dec 2023) — the last build compatible with glibc 2.28:

```bash
# Download from GitHub releases:
wget https://github.com/gnina/gnina/releases/download/v1.1/gnina \
     -O "${CONDA_PREFIX}/bin/gnina"
chmod +x "${CONDA_PREFIX}/bin/gnina"
```

Verify: `gnina --version` should print `gnina v1.1 master:e4cb380+ Built Dec 18 2023.`

---

## Analysis

```bash
conda env create -f envs/analysis.yml
conda run -n analysis pip install -e .
python -m ipykernel install --user --name analysis --display-name "Docking (analysis)"
```

All commands run from the thesis repo root (`/home/qf226/MProject/thesis`).
