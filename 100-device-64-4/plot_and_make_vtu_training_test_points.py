from pdb import set_trace as st

import numpy as np
import pyvista as pv
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


def add_point_sets(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """
    Copied form utils_nn
    """
    ratio = 0.9
    point_coordinates = mesh.boundary_triangle_centroids

    np.random.seed(0)
    tmp_idx = np.arange(point_coordinates.shape[0])
    np.random.shuffle(tmp_idx)

    num_data = point_coordinates.shape[0]
    num_train = int(ratio * num_data)

    tag_training = np.zeros_like(tmp_idx)
    tag_test = np.zeros_like(tmp_idx)
    tag_training[tmp_idx[:num_train]] = 1
    tag_test[tmp_idx[num_train:]] = 1

    mesh.face_data["training"] = tag_training
    mesh.face_data["test"] = tag_test

    return mesh


if __name__ == "__main__":

    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh = add_point_sets(mesh)
    utils_postprocess.save_mesh(mesh, "./results/point_sets.vtk")

    print("\nDone!")
