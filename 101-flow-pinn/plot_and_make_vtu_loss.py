""" """

from pdb import set_trace as st

from functools import partial
import numpy as np
import pyvista as pv
import vtk
import jax


import main
import utils_data_and_folders
import utils_nn
import utils_postprocess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_hol_params():
    """ """
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")
    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")

    tm_list, _delta_tm = utils_nn.define_tm(M, bias_rotation)
    reductions = utils_nn.define_reductions()
    tm_full_list = utils_nn.make_tm_full_list(tm_list, nodes_int)

    idx_seed = 0
    params_list = utils_data_and_folders.load_params_list(idx_seed)
    nn_list = main.define_nn_forwards_and_derivatives()

    return (
        nodes_int,
        weights_int,
        tm_list,
        tm_full_list,
        reductions,
        params_list,
        nn_list,
    )


def _load_bc_data():
    point_coordinates, bc_vals, bc_type, normals, areas, tags = (
        main.generate_boundary_data()
    )
    weights_loss = main.compute_weights_loss(areas, tags)
    return bc_vals, bc_type, weights_loss


# ---------------------------------------------------------------------------
# Per-point loss via vmap
# ---------------------------------------------------------------------------


def _pointwise_loss(
    params_list,
    nn_list,
    tm_list,
    tm_full_list,
    point_coordinates: np.ndarray,
    normals: np.ndarray,
    bc_vals: np.ndarray,
    bc_type: np.ndarray,
    weights_loss: np.ndarray,
    reductions,
    nodes_int,
    weights_int,
) -> np.ndarray:
    """ """

    def loss_one_point(pt, bv, bt, n, w):
        # Add/remove the batch dimension expected by loss_l2

        val, _ = utils_nn.loss_l2(
            params_list,
            nn_list,
            tm_list,
            tm_full_list,
            None,
            reductions,
            pt[None],  # (1, 3)
            bv[None],  # (1, D)
            bt[None],  # (1, ...)
            n[None],  # (1, 3)
            w[None],  # (1,)
            nodes_int,
            weights_int,
        )
        return val  # scalar

    loss_vmap = jax.vmap(loss_one_point, in_axes=(0, 0, 0, 0, 0))

    batch_size = 512
    results = []
    n = len(point_coordinates)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = loss_vmap(
            point_coordinates[start:end],
            bc_vals[start:end],
            bc_type[start:end],
            normals[start:end],
            weights_loss[start:end],
        )
        results.append(np.array(chunk.real))
        print(f"  loss batch {start}\u2013{end - 1} / {n - 1} done")

    return np.concatenate(results, axis=0)  # (N,)


# ---------------------------------------------------------------------------
# add_* functions
# ---------------------------------------------------------------------------


def add_loss(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """ """
    (
        nodes_int,
        weights_int,
        tm_list,
        tm_full_list,
        reductions,
        params_list,
        nn_list,
    ) = _load_hol_params()

    bc_vals, bc_type, weights_loss = _load_bc_data()

    normals = mesh.boundary_outward_normals
    points = mesh.boundary_triangle_centroids

    pointwise_loss = _pointwise_loss(
        params_list,
        nn_list,
        tm_list,
        tm_full_list,
        points,
        normals,
        bc_vals,
        bc_type,
        weights_loss,
        reductions,
        nodes_int,
        weights_int,
    )

    mesh.face_data["MSE_pointwise"] = pointwise_loss
    mesh.face_data["MSE_pointwise_log10"] = np.log10(
        np.clip(pointwise_loss, a_min=1e-30, a_max=None)
    )

    return mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh = add_loss(mesh)
    utils_postprocess.save_mesh(mesh, "./results/loss_hol.vtk")

    print("\nDone!")
