import os
from pdb import set_trace as st
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import vtk
from jax.tree_util import tree_map

import utils_data_and_folders
import utils_integral_geometry
import utils_mechanics
import utils_nn
import utils_postprocess
import case_settings


def plot_landscape(
    params_list,
    nn_list,
    tm_grid,
    tm_full_list,
    delta_tm,
    reductions,
    point_coordinates,
    bc_vals,
    bc_type,
    normals,
    weights_loss,
    mask_helper,
    nu,
    G,
    nodes_int,
    weights_int,
    der_displ_fcn,
):

    n = 100
    real_part_grid = jnp.linspace(-0.1, 0.1, n)
    imag_part_grid = jnp.linspace(-0.1, 0.1, n)

    loss_grid = np.zeros((n, n))

    names = params_list.keys()
    params_list_mod = {}

    def update_colum(delta_real, delta_imag):

        for name in names:
            params_list_mod[name] = tree_map(
                lambda param: param + delta_real + 1j * delta_imag,
                params_list[name],
            )

        loss_val, _ = utils_nn.loss_l2(
            params_list_mod,
            nn_list,
            tm_grid,
            tm_full_list,
            delta_tm,
            reductions,
            point_coordinates,
            bc_vals,
            bc_type,
            normals,
            weights_loss,
            mask_helper,
            nu,
            G,
            nodes_int,
            weights_int,
            der_displ_fcn,
        )
        return loss_val.real

    for ii, delta_real in enumerate(real_part_grid):
        print(ii)
        loss_grid[ii] = jax.vmap(
            lambda delta_imag: update_colum(delta_real, delta_imag)
        )(real_part_grid)

    X, Y = np.meshgrid(real_part_grid, imag_part_grid)
    points = np.column_stack((X.ravel(), Y.ravel(), loss_grid.T.ravel()))
    cloud = pv.PolyData(points)
    cloud.point_data["loss"] = loss_grid.T.ravel()
    cloud.save("./results/landscape.vtp")


if __name__ == "__main__":
    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")

    point_coordinates, bc_vals, bc_type, normals, areas, tags = (
        case_settings.generate_boundary_data()
    )
    tangents_1 = np.loadtxt("./results/face_tangents_1")
    tangents_2 = np.loadtxt("./results/face_tangents_2")

    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")
    tm_grid, delta_tm = utils_nn.define_tm(M, bias_rotation)
    reductions = utils_nn.define_reductions()

    tm_full_list = utils_nn.make_tm_full_list(tm_grid, nodes_int)

    idx_seed = 0
    params_list = utils_data_and_folders.load_params_list(idx_seed)
    nn_list = case_settings.define_nn_forwards_and_derivatives()

    weights_loss = case_settings.compute_weights_loss(areas, tags)

    plot_landscape(
        params_list,
        nn_list,
        tm_grid,
        tm_full_list,
        delta_tm,
        reductions,
        point_coordinates,
        bc_vals,
        bc_type,
        normals,
        weights_loss,
        None,
        nu,
        G,
        nodes_int,
        weights_int,
        None,
    )

    print("\nDone!")
