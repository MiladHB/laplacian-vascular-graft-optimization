"""
Compute wall shear stress (WSS) from the latest foamToVTK output in ./VTK.

Writes WSSAA.txt and WSSDA.txt (same scalar today: 90th percentile of |WSS|),
matching the previous script's behavior. Run with the OpenFOAM case as CWD.

Requires: vtk, numpy (same as before).
"""

from __future__ import annotations

import argparse
import glob
import os
import warnings
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support as VN

warnings.filterwarnings("ignore")
vtk.vtkObject.GlobalWarningDisplayOff()

# Dynamic viscosity (Pa·s); keep consistent with your OpenFOAM transportProperties
MU_DEFAULT = 0.00371
PERCENTILE_DEFAULT = 90
VELOCITY_ARRAY = "U"


def _newest_vtk_file(vtk_dir: Path) -> Path:
    pattern = str(vtk_dir / "*.vtk")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No VTK files matching {pattern}. Run foamToVTK from this case first."
        )
    return Path(max(files, key=os.path.getctime))


def compute_wss_tangential(
    velocity_grad: np.ndarray,
    normals: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Return WSS vectors (N, 3): tangential part of wall traction."""
    n_pts = velocity_grad.shape[0]
    wss = np.zeros((n_pts, 3), dtype=float)
    g = np.zeros((3, 3), dtype=float)
    n_vec = np.zeros((3, 1), dtype=float)

    for i in range(n_pts):
        gv = velocity_grad[i]
        g[0, 0], g[0, 1], g[0, 2] = gv[0], gv[1], gv[2]
        g[1, 0], g[1, 1], g[1, 2] = gv[3], gv[4], gv[5]
        g[2, 0], g[2, 1], g[2, 2] = gv[6], gv[7], gv[8]
        gt = g.T
        n_vec[0, 0] = normals[i, 0]
        n_vec[1, 0] = normals[i, 1]
        n_vec[2, 0] = normals[i, 2]

        traction = -mu * (gt + g) @ n_vec
        tr = traction.ravel()
        n_hat = normals[i]
        t_n = float(np.dot(tr, n_hat))
        t_t = tr - t_n * n_hat
        wss[i, :] = t_t

    return wss


def run(
    input_vtk: Path,
    wss_vtk_out: Path,
    mu: float,
    percentile: float,
) -> tuple[float, float]:
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(input_vtk))
    reader.Update()
    data = reader.GetOutput()

    gradient_filter = vtk.vtkGradientFilter()
    gradient_filter.SetInputData(data)
    gradient_filter.SetInputArrayToProcess(0, 0, 0, 0, VELOCITY_ARRAY)
    gradient_filter.SetResultArrayName("gradtheta1")
    gradient_filter.Update()
    data_grad = gradient_filter.GetOutput()

    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputData(data_grad)
    surface.Update()
    data_grad = surface.GetOutput()

    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(data_grad)
    normals_filter.SetFeatureAngle(91)
    normals_filter.SetSplitting(0)
    normals_filter.Update()
    poly_normals = normals_filter.GetOutput()
    normals = VN.vtk_to_numpy(poly_normals.GetPointData().GetArray("Normals"))

    n_points = data_grad.GetNumberOfPoints()
    wss_grad_vector = VN.vtk_to_numpy(
        data_grad.GetPointData().GetArray("gradtheta1")
    )
    wss = compute_wss_tangential(wss_grad_vector, normals, mu)

    wss_vtk = VN.numpy_to_vtk(wss)
    wss_vtk.SetName("wss")
    data_grad.GetPointData().AddArray(wss_vtk)

    wss_vtk_out.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkDataSetWriter()
    writer.SetInputData(data_grad)
    writer.SetFileName(str(wss_vtk_out))
    writer.Write()

    wss_mag = np.sqrt(np.sum(wss**2, axis=1))
    scalar = float(np.percentile(wss_mag, percentile))
    # Historical behavior: both files store the same metric (objective uses WSSDA).
    return scalar, scalar


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vtk",
        type=Path,
        default=None,
        help="Input .vtk (default: newest VTK/*.vtk under --vtk-dir)",
    )
    p.add_argument(
        "--vtk-dir",
        type=Path,
        default=Path("VTK"),
        help="Directory to search for VTK when --vtk is omitted (default: VTK)",
    )
    p.add_argument(
        "--wss-out",
        type=Path,
        default=Path("VTK") / "wss.vtk",
        help="Where to write WSS-enriched surface (default: VTK/wss.vtk)",
    )
    p.add_argument("--mu", type=float, default=MU_DEFAULT, help="Dynamic viscosity")
    p.add_argument(
        "--percentile",
        type=float,
        default=float(PERCENTILE_DEFAULT),
        help="Percentile of |WSS| for scalar outputs",
    )
    args = p.parse_args()

    input_vtk = args.vtk if args.vtk is not None else _newest_vtk_file(args.vtk_dir)
    wssaa, wssda = run(input_vtk, args.wss_out, args.mu, args.percentile)

    with open("WSSAA.txt", "w", encoding="utf-8") as f:
        f.write(str(wssaa))
    with open("WSSDA.txt", "w", encoding="utf-8") as f:
        f.write(str(wssda))


if __name__ == "__main__":
    main()
