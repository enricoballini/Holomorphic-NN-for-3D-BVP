from functools import partial
import numpy as np
import pyvista as pv
import vtk
import jax
import jax.numpy as jnp

import case_settings
import utils_data_and_folders
import utils_nn
import utils_postprocess


def _load_hol_params():
    """"""
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")
    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")

    tm_list, _delta_tm = utils_nn.define_tm(M, bias_rotation)
    reductions = utils_nn.define_reductions()
    tm_full_list = utils_nn.make_tm_full_list(tm_list, nodes_int)

    idx_seed = 0
    params_list = utils_data_and_folders.load_params_list(idx_seed)
    nn_list = case_settings.define_nn_forwards_and_derivatives()

    return (
        nodes_int,
        weights_int,
        tm_list,
        tm_full_list,
        reductions,
        params_list,
        nn_list,
    )


def add_V_nn(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """ """
    (
        nodes_int,
        weights_int,
        tm_grid,
        tm_full_list,
        reductions,
        params_list,
        nn_list,
    ) = _load_hol_params()

    V_nn = utils_nn.potential_guess(
        params_list,
        nn_list,
        jnp.array(tm_grid),
        tm_full_list,
        nodes_int,
        weights_int,
        reductions,
        mesh.nodes,
    )
    V_nn = np.array(np.real(V_nn)).ravel()

    mesh.node_data["V_nn"] = V_nn
    return mesh


if __name__ == "__main__":

    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh = add_V_nn(mesh)
    utils_postprocess.save_mesh(mesh, "./results/V_nn.vtk")

    print("\nDone!")
