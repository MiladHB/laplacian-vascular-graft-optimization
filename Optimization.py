"""
Cleaned and structured version of the ONE_REGION notebook code.

This module:
- builds a Laplacian-based parametrization of a surface mesh,
- defines a deformation model using Laplacian eigenmodes,
- connects to an external OpenFOAM-based pipeline to evaluate designs,
- runs Bayesian Optimization (via BoTorch) to search for optimal parameters,
  using a Gaussian-process surrogate and **Probability of Improvement** (PI) as
  the acquisition function—the same criterion as the original notebook.
- produces simple plots of the optimization progress.

Defaults match ``ONE_REGION.ipynb`` for this case: symmetric coefficient bounds
``[-0.012, 0.012]``, ``tda = 0.00529``, 100 initial samples + 300 BO iterations,
single-stage optimization, and design exports under ``result/``.

The implementation is function-based and avoids notebook-specific boilerplate.
You will still need a working OpenFOAM case in the current directory, along
with the same STL and helper files used by the original notebook.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import trimesh
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ProbabilityOfImprovement
from botorch.acquisition.monte_carlo import qProbabilityOfImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from scipy.linalg import eigh


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GeometryConfig:
    """File names for geometry inputs and outputs."""

    interior_stl: str = "interior.stl"  # Main deformable mesh (interior surface).
    upper_boundary_stl: str = "upper_boundary.stl"  # Boundary mesh: top, used to tag fixed vertices.
    right_boundary_stl: str = "right_boundary.stl"  # Boundary mesh: right, used to tag fixed vertices.
    bottom_boundary_stl: str = "bottom_boundary.stl"  # Boundary mesh: bottom, used to tag fixed vertices.

    upper_boundary2_stl: str = "upper_boundary2.stl"  # Second boundary set for pinned vertices (Humphrey smoothing).
    right_boundary2_stl: str = "right_boundary2.stl"  # Second boundary set (right).
    bottom_boundary2_stl: str = "bottom_boundary2.stl"  # Second boundary set (bottom).

    bottom_stl: str = "bottom.stl"  # Static wall segment concatenated with the deformed interior for export.
    upper_stl: str = "upper.stl"  # Static wall segment (upper).
    right_stl: str = "right.stl"  # Static wall segment (right).

    openfoam_wall_stl: str = "constant/triSurface/wall.stl"  # Where the final combined wall STL is written for OpenFOAM.
    designs_output_dir: str = "results"  # Where each optimization design is saved as WALL_<run_id>.stl (ONE_REGION.ipynb).


@dataclass
class OptimizationConfig:
    """Hyperparameters for the Bayesian Optimization loop."""

    num_dims: int = 20  # Number of Laplacian modes / optimization parameters (coeffs dimension).
    coeff_low: float = -0.012  # Symmetric lower bound for LHS and BO (matches ONE_REGION.ipynb ``bounds``).
    coeff_high: float = 0.012  # Symmetric upper bound for LHS and BO.
    num_initial_samples: int = 100  # Number of initial random designs before sequential BO (notebook ``n_samples``).
    bo_runs: int = 300  # Sequential BO iterations after initial samples (notebook ``n_runs``).
    wss_penalty_factor: float = 0.00529  # tda: weight on WSSDA in objective (pressure − tda × WSSDA).
    optim_log_path: Optional[str] = "results/OPTIM"  # Append optimization log; None disables file.
    early_stop_patience: Optional[int] = None  # Stop if no best objective improvement for this many consecutive BO iterations; None = run full.
    early_stop_min_delta: float = 0.0  # Improvement must exceed old_best + this to reset stagnation (noise floor).
    early_stop_max_pi: Optional[float] = None  # If set, also require PI at candidate < this (e.g. 1e-3) to stop early; None = stagnation only.
    use_analytic_probability_of_improvement: bool = False  # True: analytic PI; False: qMonte Carlo PI (matches original notebook).


@dataclass
class GeometryData:
    """Precomputed geometry and modal data used by the objective function."""

    base_vertices: np.ndarray  # Interior mesh vertex positions before deformation.
    normals: np.ndarray  # Unit vertex normals for modal displacement direction.
    free_mask: np.ndarray  # True = vertex can move; False = fixed by boundary tagging.
    pinned_indices: np.ndarray  # Vertex indices pinned for Humphrey smoothing (boundary2 set).
    modes: np.ndarray  # Laplacian eigenmodes, shape (num_vertices, num_modes).
    mesh_bottom: trimesh.Trimesh  # Static bottom part of the wall for concatenation.
    mesh_upper: trimesh.Trimesh  # Static upper part of the wall for concatenation.
    mesh_right: trimesh.Trimesh  # Static right part of the wall for concatenation.


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def _find_boundary_indices(
    vertices: np.ndarray,
    boundary_vertices_list: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tol: float = 1e-5,
) -> np.ndarray:
    """Return indices of vertices that are close to any of the boundary meshes."""

    indices: list[int] = []
    bdt, bdr, bdb = boundary_vertices_list

    for i in range(vertices.shape[0]):
        v = vertices[i]
        # Distances to each boundary set of vertices
        d_t = np.min(np.linalg.norm(v - bdt, axis=1))
        d_r = np.min(np.linalg.norm(v - bdr, axis=1))
        d_b = np.min(np.linalg.norm(v - bdb, axis=1))
        if min(d_t, d_r, d_b) < tol:
            indices.append(i)

    return np.array(indices, dtype=int)


def _build_laplacian_modes(
    mesh: trimesh.Trimesh, num_modes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build graph Laplacian of the mesh and return a subset of eigenmodes.

    Returns:
        eigenvalues, modes
        where modes has shape (num_vertices, num_modes) and corresponds to
        absolute values of eigenvectors 1..num_modes (skipping the first,
        constant eigenvector) as in the original script.
    """
    n_vertices = mesh.vertices.shape[0]
    edges = mesh.edges

    g = nx.Graph()
    g.add_nodes_from(range(n_vertices))
    for i0, i1 in edges:
        g.add_edge(int(i0), int(i1), weight=1.0)

    lap = nx.laplacian_matrix(g).astype(float)
    lap_dense = lap.toarray()

    # Full eigendecomposition (can be expensive for very large meshes,
    # but this mirrors the original code behavior).
    evals, evecs = eigh(lap_dense)

    # Skip the first eigenvector (index 0) and keep num_modes of them.
    # Take absolute value, replicating the original behavior.
    evecs_abs = np.abs(np.real(evecs[:, 1 : 1 + num_modes]))
    return evals, evecs_abs


def prepare_geometry(
    geom_cfg: GeometryConfig, num_modes: int
) -> GeometryData:
    """
    Load meshes, compute boundary indices and Laplacian eigenmodes.

    This function encapsulates the geometry-related notebook code.
    """
    # Main interior mesh
    interior_mesh = trimesh.load_mesh(geom_cfg.interior_stl)
    vertices = interior_mesh.vertices.copy()
    normals = interior_mesh.vertex_normals.copy()
    num_vertices = vertices.shape[0]

    # Boundary meshes (first set)
    mesh_bdt = trimesh.load_mesh(geom_cfg.upper_boundary_stl)
    mesh_bdr = trimesh.load_mesh(geom_cfg.right_boundary_stl)
    mesh_bdb = trimesh.load_mesh(geom_cfg.bottom_boundary_stl)
    boundary_indices = _find_boundary_indices(
        vertices,
        (mesh_bdt.vertices, mesh_bdr.vertices, mesh_bdb.vertices),
    )

    # Graph Laplacian and modes
    _, modes = _build_laplacian_modes(interior_mesh, num_modes=num_modes)

    # Second set of boundary meshes (pinned vertices for smoothing)
    mesh_bdt2 = trimesh.load_mesh(geom_cfg.upper_boundary2_stl)
    mesh_bdr2 = trimesh.load_mesh(geom_cfg.bottom_boundary2_stl)
    mesh_bdb2 = trimesh.load_mesh(geom_cfg.right_boundary2_stl)
    pinned_indices = _find_boundary_indices(
        vertices,
        (mesh_bdt2.vertices, mesh_bdr2.vertices, mesh_bdb2.vertices),
    )

    # Auxiliary meshes used when exporting the final wall
    mesh_bottom = trimesh.load_mesh(geom_cfg.bottom_stl)
    mesh_upper = trimesh.load_mesh(geom_cfg.upper_stl)
    mesh_right = trimesh.load_mesh(geom_cfg.right_stl)

    free_mask = np.ones(num_vertices, dtype=bool)
    free_mask[boundary_indices] = False

    return GeometryData(
        base_vertices=vertices,
        normals=normals,
        free_mask=free_mask,
        pinned_indices=pinned_indices,
        modes=modes,
        mesh_bottom=mesh_bottom,
        mesh_upper=mesh_upper,
        mesh_right=mesh_right,
    )


def deform_vertices(
    base_vertices: np.ndarray,
    normals: np.ndarray,
    coeffs: np.ndarray,
    modes: np.ndarray,
    free_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply modal deformation to the base vertices along their normals.

    Args:
        base_vertices: (N, 3)
        normals: (N, 3)
        coeffs: (num_modes,)
        modes: (N, num_modes)
        free_mask: boolean array of length N, False for fixed vertices.
    """
    if coeffs.ndim != 1:
        coeffs = coeffs.reshape(-1)

    coef = free_mask.astype(float)[:, None]  # (N, 1)
    # (N, num_modes) * (num_modes,) -> (N,)
    modal_amplitude = (modes * coef) @ coeffs  # (N,)
    displacement = modal_amplitude[:, None] * normals  # (N, 3)
    return base_vertices + displacement


# ---------------------------------------------------------------------------
# OpenFOAM / external pipeline
# ---------------------------------------------------------------------------


def _optim_log(optim_log_path: Optional[str], message: str) -> None:
    """Print and append to OPTIM log (matches notebook stdout capture)."""
    print(message)
    if optim_log_path:
        parent = os.path.dirname(optim_log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(optim_log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def _best_objective_improved(
    new_best: float, previous_best: float, min_delta: float
) -> bool:
    """True if the global best (max) increased by more than min_delta."""
    return new_best > previous_best + min_delta


def _scalar_acquisition_value(qpi: np.ndarray) -> float:
    """Single PI value from BoTorch acquisition output (q=1)."""
    return float(np.asarray(qpi, dtype=float).ravel()[0])


def _should_early_stop(
    stagnation: int,
    patience: int,
    qpi: np.ndarray,
    max_pi: Optional[float],
) -> bool:
    """
    Stagnation-only stop if max_pi is None; otherwise require PI < max_pi as well.
    """
    if stagnation < patience:
        return False
    if max_pi is None:
        return True
    return _scalar_acquisition_value(qpi) < max_pi


def run_openfoam_pipeline(
    wss_penalty_factor: float,
    optim_log_path: Optional[str] = None,
) -> np.ndarray:
    """
    Run the external OpenFOAM pipeline and return the objective value.

    This closely follows the original sequence of shell commands used in
    the notebook. It assumes that the current working directory is an
    OpenFOAM case and that helper scripts (e.g. get_WSS3D_vtk.py) exist.
    """
    commands = [
        "mkdir 0.org",
        "cp -r 0/* 0.org/",
        "rm -r 0/*",
        "blockMesh >/dev/null 2>&1",
        "surfaceFeatureExtract >/dev/null 2>&1",
        "snappyHexMesh -overwrite >/dev/null 2>&1",
        "cp -r 0.org/* 0/",
        "rm -r 0.org",
    ]

    cleanup_before = [
        "simpleFoam >/dev/null 2>&1",
        "rm tmp.txt >/dev/null 2>&1",
        "rm WSSAA.txt >/dev/null 2>&1",
        "rm WSSDA.txt >/dev/null 2>&1",
    ]

    postprocess_commands = [
        "postProcess -fields '(p)' -func pressureDifferencePatch >/dev/null 2>&1",
        "tail -n1 ./postProcessing/pressureDifferencePatch/*/fieldValueDelta.dat | cut -c 15-  > tmp.txt",
        "foamToVTK >/dev/null 2>&1",
        "python get_WSS3D_vtk.py",
    ]

    cleanup_after = [
        "rm -r postProcessing",
        "rm -r 1* >/dev/null 2>&1",
        "rm -r 2* >/dev/null 2>&1",
        "rm -r 4* >/dev/null 2>&1",
        "rm -r 5* >/dev/null 2>&1",
        "rm -r 6* >/dev/null 2>&1",
        "rm -r 7* >/dev/null 2>&1",
        "rm -r 8* >/dev/null 2>&1",
        "rm -r 9* >/dev/null 2>&1",
        "rm -r VTK",
    ]

    for cmd in commands:
        os.system(cmd)

    start_time = time.time()
    for cmd in cleanup_before:
        os.system(cmd)
    simplefoam_time = time.time() - start_time
    _optim_log(optim_log_path, f"TIMES: {simplefoam_time}")

    for cmd in postprocess_commands:
        os.system(cmd)

    # Read pressure drop from tmp.txt
    with open("tmp.txt", encoding="utf-8") as f:
        text = f.read()
    pressure_tokens = text.split("\t")
    pressure_drop = -float(pressure_tokens[1])

    # Read wall shear stress metrics (gracefully handle missing files)
    wssaa = 0.0
    wssda = 0.0
    if os.path.exists("WSSAA.txt"):
        try:
            with open("WSSAA.txt", encoding="utf-8") as f:
                wssaa = float(f.read())
        except Exception as exc:
            print(f"Warning: failed to read WSSAA.txt ({exc}); using 0.0")
    else:
        print("Warning: WSSAA.txt not found; using 0.0")

    if os.path.exists("WSSDA.txt"):
        try:
            with open("WSSDA.txt", encoding="utf-8") as f:
                wssda = float(f.read())
        except Exception as exc:
            print(f"Warning: failed to read WSSDA.txt ({exc}); using 0.0")
    else:
        print("Warning: WSSDA.txt not found; using 0.0")

    _optim_log(optim_log_path, f"WSSAA: {wssaa}")
    _optim_log(optim_log_path, f"WSSDA: {wssda}")
    # Match ONE_REGION.ipynb: print('PRESSURE', -np.asarray(YYY)) with YYY = [-p] from tmp.txt
    _optim_log(optim_log_path, f"PRESSURE {np.asarray([-pressure_drop])}")
    objective = pressure_drop - wss_penalty_factor * wssda
    _optim_log(optim_log_path, f"Constrianed PRESSURE: {np.asarray([objective])}")

    for cmd in cleanup_after:
        os.system(cmd)

    return np.asarray([objective], dtype=float)


# ---------------------------------------------------------------------------
# Objective function wrapper
# ---------------------------------------------------------------------------


def make_objective(
    geom: GeometryData,
    geom_cfg: GeometryConfig,
    opt_cfg: OptimizationConfig,
):
    """
    Create a callable objective used by Bayesian Optimization.

    The returned function maps a tensor of shape (1, num_dims) to a
    1D torch tensor containing the (constrained) pressure drop.
    """

    def objective(coeffs_tensor: torch.Tensor, run_id: int) -> torch.Tensor:
        coeffs = coeffs_tensor.detach().cpu().numpy().reshape(-1)

        # Deform interior mesh
        deformed_vertices = deform_vertices(
            base_vertices=geom.base_vertices,
            normals=geom.normals,
            coeffs=coeffs,
            modes=geom.modes,
            free_mask=geom.free_mask,
        )

        # Build and smooth mesh
        interior_mesh = trimesh.load_mesh(geom_cfg.interior_stl)
        interior_mesh.vertices = deformed_vertices

        lap_op = trimesh.smoothing.laplacian_calculation(
            interior_mesh,
            equal_weight=False,
            pinned_vertices=geom.pinned_indices.tolist(),
        )
        smoothed_mesh = trimesh.smoothing.filter_humphrey(
            interior_mesh,
            alpha=0.0,
            beta=0.5,
            iterations=3000,
            laplacian_operator=lap_op,
        )

        # Concatenate with other wall parts
        total_mesh = trimesh.util.concatenate(
            [smoothed_mesh, geom.mesh_bottom, geom.mesh_upper, geom.mesh_right]
        )

        # Export for OpenFOAM and also keep per-design STL
        os.makedirs(os.path.dirname(geom_cfg.openfoam_wall_stl), exist_ok=True)
        total_mesh.export(geom_cfg.openfoam_wall_stl)

        os.makedirs(geom_cfg.designs_output_dir, exist_ok=True)
        design_path = os.path.join(
            geom_cfg.designs_output_dir,
            f"WALL_{run_id}.stl",
        )
        total_mesh.export(design_path)

        objective_np = run_openfoam_pipeline(
            wss_penalty_factor=opt_cfg.wss_penalty_factor,
            optim_log_path=opt_cfg.optim_log_path,
        )
        return torch.from_numpy(objective_np.astype(np.float32))

    return objective


# ---------------------------------------------------------------------------
# Bayesian Optimization helpers
# ---------------------------------------------------------------------------


def generate_initial_data(
    objective,
    opt_cfg: OptimizationConfig,
    bounds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Evaluate the objective at an initial set of random points.
    """
    num_dims = opt_cfg.num_dims
    n = opt_cfg.num_initial_samples

    _optim_log(opt_cfg.optim_log_path, "Sample_id: 0")
    train_x = torch.zeros(1, num_dims)
    exact_obj = objective(train_x, run_id=0).unsqueeze(-1)
    all_x = train_x
    all_y = exact_obj

    for idx in range(1, n):
        _optim_log(opt_cfg.optim_log_path, f"Sample_id: {idx}")
        # Uniform sampling within bounds
        low = bounds[0].numpy()
        high = bounds[1].numpy()
        sample = torch.from_numpy(
            np.random.uniform(low=low, high=high, size=(1, num_dims)).astype(
                np.float32
            )
        )
        value = objective(sample, run_id=idx).unsqueeze(-1)
        all_x = torch.cat([all_x, sample])
        all_y = torch.cat([all_y, value])

    best_value = all_y.max().item()
    return all_x, all_y, best_value


def get_next_points(
    init_x: torch.Tensor,
    init_y: torch.Tensor,
    best_init_y: float,
    bounds: torch.Tensor,
    n_points: int = 1,
    use_analytic_pi: bool = False,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Fit a GP model and optimize **Probability of Improvement** (PI) at the
    candidate point(s) relative to ``best_init_y`` (maximization).
    """
    model = SingleTaskGP(init_x, init_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    if use_analytic_pi:
        if n_points != 1:
            raise ValueError(
                "analytic ProbabilityOfImprovement requires n_points=1; "
                f"got n_points={n_points}"
            )
        acquisition = ProbabilityOfImprovement(model=model, best_f=best_init_y)
    else:
        acquisition = qProbabilityOfImprovement(model=model, best_f=best_init_y)

    candidates, _ = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=n_points,
        num_restarts=64,
        raw_samples=128,
    )

    qpi = acquisition(candidates).detach().numpy()
    return candidates, qpi


def run_optimization(
    geom: GeometryData,
    geom_cfg: GeometryConfig,
    opt_cfg: OptimizationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the Bayesian Optimization loop (same structure as ONE_REGION.ipynb).

    If ``early_stop_patience`` is set, the loop can stop after that many
    consecutive BO iterations without a better best objective (see
    ``early_stop_min_delta``). If ``early_stop_max_pi`` is also set, stopping
    requires in addition that the current PI at the candidate is below that
    threshold.

    Returns:
        best_values: array of best objective values per iteration
        acquisition_values: array of acquisition values per iteration
    """
    obj = make_objective(geom, geom_cfg, opt_cfg)
    num_dims = opt_cfg.num_dims

    low = np.full(num_dims, opt_cfg.coeff_low, dtype=np.float32)
    high = np.full(num_dims, opt_cfg.coeff_high, dtype=np.float32)
    bounds = torch.from_numpy(np.stack([low, high], axis=0))

    # Fresh OPTIM file for this run (same as redirecting notebook stdout).
    if opt_cfg.optim_log_path:
        parent = os.path.dirname(opt_cfg.optim_log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(opt_cfg.optim_log_path, "w", encoding="utf-8"):
            pass

    # Initial data
    init_x, init_y, best_y = generate_initial_data(obj, opt_cfg, bounds)
    best_values = [best_y]
    acquisition_values: list[np.ndarray] = []
    best_row = int(init_y.argmax().item())
    _optim_log(
        opt_cfg.optim_log_path,
        f"Best design run_id: {best_row}",
    )

    es_patience = opt_cfg.early_stop_patience
    es_delta = opt_cfg.early_stop_min_delta
    es_max_pi = opt_cfg.early_stop_max_pi
    use_early_stop = es_patience is not None and es_patience > 0
    stagnation = 0

    for i in range(opt_cfg.bo_runs):
        best_before_iter = init_y.max().item()
        run_id = opt_cfg.num_initial_samples + i
        _optim_log(
            opt_cfg.optim_log_path,
            f"Nr. of optimization run: {run_id}",
        )
        t0 = time.time()
        new_x, qpi = get_next_points(
            init_x,
            init_y,
            best_y,
            bounds,
            n_points=1,
            use_analytic_pi=opt_cfg.use_analytic_probability_of_improvement,
        )
        new_y = obj(new_x, run_id=run_id).unsqueeze(-1)
        dt = time.time() - t0
        _optim_log(opt_cfg.optim_log_path, f"TIME5: {dt}")
        _optim_log(opt_cfg.optim_log_path, f"New candidates are: {new_x}")
        _optim_log(opt_cfg.optim_log_path, f"QPI: {qpi}")

        init_x = torch.cat([init_x, new_x])
        init_y = torch.cat([init_y, new_y])
        best_y = init_y.max().item()
        acquisition_values.append(qpi)
        best_values.append(best_y)
        best_row = int(init_y.argmax().item())
        _optim_log(
            opt_cfg.optim_log_path,
            f"Best point performs this way: {-best_y}",
        )
        _optim_log(
            opt_cfg.optim_log_path,
            f"Best design run_id: {best_row}",
        )

        if use_early_stop:
            if _best_objective_improved(best_y, best_before_iter, es_delta):
                stagnation = 0
            else:
                stagnation += 1
            if _should_early_stop(stagnation, es_patience, qpi, es_max_pi):
                if es_max_pi is None:
                    _optim_log(
                        opt_cfg.optim_log_path,
                        f"Early stop: no improvement for {es_patience} consecutive "
                        f"iterations (min_delta={es_delta}).",
                    )
                else:
                    _optim_log(
                        opt_cfg.optim_log_path,
                        f"Early stop: no improvement for {es_patience} consecutive "
                        f"iterations and PI={_scalar_acquisition_value(qpi):.6g} "
                        f"< early_stop_max_pi={es_max_pi} (min_delta={es_delta}).",
                    )
                break

    return np.array(best_values), np.array(acquisition_values, dtype=object)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_progress(best_values: np.ndarray, acquisition_values: np.ndarray) -> None:
    """Replicate the simple progress plots from the notebook."""
    # Plot best objective value per iteration
    dims = np.arange(1, len(best_values) + 1)
    best_pressure = -1.0 * np.asarray(best_values)

    plt.figure(figsize=(11, 10))
    plt.plot(dims, best_pressure, "--")
    plt.xlabel("RUN ID")
    plt.ylabel("BEST PRESSURE DROP")
    plt.savefig("PRE_CORRECTED_PRESSURE.jpg")

    # Plot acquisition values
    if len(acquisition_values) > 0:
        dims2 = np.arange(1, len(acquisition_values) + 1)
        plt.figure(figsize=(11, 10))
        plt.plot(dims2, acquisition_values, "--")
        plt.xlabel("RUN ID")
        plt.ylabel("ACQUISITION VALUE (QPI)")
        plt.savefig("PROB.jpg")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    geom_cfg = GeometryConfig()
    opt_cfg = OptimizationConfig()

    print("Preparing geometry and Laplacian modes...")
    # Use exactly `num_dims` modes so that the modal basis and the
    # optimization parameter dimension match (avoids shape mismatch).
    geom = prepare_geometry(geom_cfg, num_modes=opt_cfg.num_dims)

    print("Starting Bayesian Optimization...")
    best_values, acquisition_values = run_optimization(geom, geom_cfg, opt_cfg)

    print("Plotting progress...")
    plot_progress(best_values, acquisition_values)

    print("Optimization completed.")


if __name__ == "__main__":
    main()

