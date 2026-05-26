""" """

import numpy as np
import vtk

import case_settings
import case_settings_pinn
import utils_data_and_folders_pinn
import utils_mechanics
import utils_mechanics_pinn
import utils_postprocess


# ---------------------------------------------------------------------------
# Helpers – parameter loaders
# ---------------------------------------------------------------------------


def _load_pinn_params():
    """Load PINN material data and trained model parameters."""
    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")
    idx_seed = 0
    params_list = utils_data_and_folders_pinn.load_params_list(idx_seed)
    nn_list = case_settings_pinn.define_nn_forwards_and_derivatives()
    return nu, G, params_list, nn_list


# ---------------------------------------------------------------------------
# Batched wrapper
# ---------------------------------------------------------------------------


def _batched_sigma_pinn(
    params_list,
    nn_list,
    points: np.ndarray,
    nu,
    G,
    batch_size: int = 512,
) -> np.ndarray:
    """Call compute_sigma_tensor_from_displ (PINN) in chunks of `batch_size` points."""
    results = []
    n = len(points)
    for start in range(0, n, batch_size):
        chunk = points[start : start + batch_size]
        sigma_chunk = utils_mechanics_pinn.compute_sigma_tensor_from_displ(
            params_list, nn_list, chunk, nu, G
        )
        results.append(np.array(sigma_chunk))
        print(
            f"  PINN stress batch {start}–{min(start + batch_size, n) - 1} / {n - 1} done"
        )
    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# add_*_errors functions
# ---------------------------------------------------------------------------


def add_stress_errors(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """Compute absolute stress errors (exact – PINN) at nodes."""

    stress_exact = np.load("./results/fem_stresses.npy")
    stress_exact *= 0.1

    nu, G, params_list, nn_list = _load_pinn_params()
    stress_pinn = _batched_sigma_pinn(params_list, nn_list, mesh.nodes, nu, G)
    stress_pinn *= 0.1

    err_pinn = np.abs(stress_exact - stress_pinn)  # (N, 3, 3)

    mesh.node_data["stress_err_pinn"] = err_pinn.reshape(err_pinn.shape[0], -1)
    mesh.node_data["stress_err_l1_pinn"] = np.sum(err_pinn, axis=(1, 2))

    return mesh


def add_traction_errors(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """Compute absolute traction errors (exact – PINN) at boundary face centres."""

    normals = mesh.boundary_outward_normals  # (F, 3)

    stress_exact_nodes = np.load("./results/fem_stresses.npy")
    stress_exact_faces = utils_postprocess.interpolate_node_values_to_face_centers(
        mesh, stress_exact_nodes, face_type="boundary"
    )
    stress_exact_faces *= 0.1
    tractions_exact = utils_mechanics.compute_sigma_n(stress_exact_faces, normals)

    nu, G, params_list, nn_list = _load_pinn_params()
    stress_pinn_faces = _batched_sigma_pinn(
        params_list, nn_list, mesh.boundary_triangle_centroids, nu, G
    )
    stress_pinn_faces *= 0.1
    tractions_pinn = utils_mechanics.compute_sigma_n(stress_pinn_faces, normals)

    diff_pinn = np.abs(tractions_exact - tractions_pinn)  # (F, 3)

    mesh.face_data["tractions_err_pinn"] = diff_pinn
    mesh.face_data["tractions_err_norm_pinn"] = np.linalg.norm(diff_pinn, axis=1)

    return mesh


def add_displacement_errors(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """Compute absolute displacement errors (exact – PINN) at nodes."""

    _, displ_exact = case_settings.compute_exact_solution(mesh.nodes)
    displ_exact *= 0.1

    nu, G, params_list, nn_list = _load_pinn_params()
    displ_pinn = utils_mechanics_pinn.compute_displacement(
        params_list, nn_list, mesh.nodes
    )
    displ_pinn *= 0.1

    err_pinn = np.abs(displ_exact - displ_pinn)  # (N, 3)

    mesh.node_data["displacement_err_pinn"] = err_pinn
    mesh.node_data["displacement_err_norm_pinn"] = np.linalg.norm(err_pinn, axis=1)

    return mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh = add_stress_errors(mesh)
    mesh = add_traction_errors(mesh)
    utils_postprocess.save_mesh(mesh, "./results/errors_stress_tractions_pinn.vtk")

    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh = add_displacement_errors(mesh)
    utils_postprocess.save_mesh(mesh, "./results/errors_displacement_pinn.vtk")

    print("\nDone!")
