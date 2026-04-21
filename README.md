# Automatic Laplacian-based shape optimization for patient-specific vascular grafts

This repository contains the **code and data for one example case** associated with the paper:

- Milad Habibi et al., *Automatic Laplacian-based shape optimization for patient-specific vascular grafts*, Computers in Biology and Medicine, 2024.  
  [ScienceDirect link](https://www.sciencedirect.com/science/article/pii/S0010482524013933)

## What is included

- OpenFOAM case setup (`constant/`, `system/`)
- Geometry files for **one example case** (`*.stl`)
- Optimization script: `Optimization.py`
- Post-processing script for wall shear stress: `get_WSS3D_vtk.py`

## Data availability

- The repository currently provides **geometry and scripts for one example case**.
- **Other case geometries are available upon reasonable request.**

## Requirements

- Python 3.10+ (recommended)
- OpenFOAM environment available in your shell
- Python packages listed in `requirements.txt`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

1. Activate your OpenFOAM environment.
2. Move to the repository root.
3. Run:

```bash
python Optimization.py
```

The optimization script exports deformed wall geometries and uses the OpenFOAM pipeline defined in the case directory.

## Notes

- `Optimization.py` is configured for the current example-case setup and paths.
- Make sure mesh/solver utilities (e.g., `blockMesh`, `snappyHexMesh`, `simpleFoam`, `foamToVTK`) are available.

## Citation

If you use this repository, please cite:

```text
Habibi, M., et al. Automatic Laplacian-based shape optimization for patient-specific vascular grafts.
Computers in Biology and Medicine (2024).
https://www.sciencedirect.com/science/article/pii/S0010482524013933
```
