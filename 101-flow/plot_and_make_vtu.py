import os
from pdb import set_trace as st

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

import utils_data_and_folders
import utils_integral_geometry
import utils_lapl
import utils_nn
import utils_postprocess
import main

dprint = jax.debug.print
jax.config.update("jax_enable_x64", True)


def add_divergence(mesh):
    cloud = pv.PolyData(mesh.inner_nodes)
    seed = 0
    params_list = utils_data_and_folders.load_params_list(seed)

    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")
    tm_grid, delta_tm = utils_nn.define_tm(M, bias_rotation)

    reductions = utils_nn.define_reductions()
    nn_list = main.define_nn_forwards_and_derivatives()
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")
    tm_full_list = utils_integral_geometry.make_tm_full_list(tm_grid, nodes_int)

    def velocity(point_inner):
        vel = utils_lapl.compute_gradient(
            lambda point: utils_nn.potential_guess(
                params_list,
                nn_list,
                jnp.array(tm_grid),
                tm_full_list,
                nodes_int,
                weights_int,
                reductions,
                jnp.atleast_2d(point),
            ),
            point_inner,
        )
        return vel

    divs = jax.vmap(lambda pt: utils_lapl.divergence(velocity, pt))(mesh.inner_nodes)
    cloud.point_data["div"] = np.log10(np.abs(divs))
    return cloud


def make_vtp(
    potential_guess,
    params_list,
    nn_list,
    point_coordinates,
    tm_grid,
    tm_full_list,
    delta_tm,
    reductions,
    nodes_int,
    weights_int,
):

    # Compute guessed potential
    V_g = potential_guess(
        params_list,
        nn_list,
        jnp.array(tm_grid),
        tm_full_list,
        nodes_int,
        weights_int,
        reductions,
        point_coordinates,
    )
    V_g = np.array(np.real(V_g)).ravel()  # np.real just to change the dtype

    velocity = utils_lapl.compute_gradients(
        lambda point: potential_guess(
            params_list,
            nn_list,
            jnp.array(tm_grid),
            tm_full_list,
            nodes_int,
            weights_int,
            reductions,
            jnp.atleast_2d(point),
        ),
        point_coordinates,
    )

    # Create PolyData (point cloud)
    cloud = pv.PolyData(point_coordinates)

    # Add node values
    cloud.point_data["V_g"] = V_g
    cloud.point_data["velocity"] = jnp.real(velocity)

    # Save as .vtu (unstructured format)
    cloud.save("./results/potentials.vtk")


if __name__ == "__main__":
    seed = 0
    params_list = utils_data_and_folders.load_params_list(seed)
    node_coordinates, bc_vals, bc_type, normals, areas, tags = (
        main.generate_boundary_data()
    )
    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()

    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")
    tm_grid, delta_tm = utils_nn.define_tm(M, bias_rotation)

    reductions = utils_nn.define_reductions()
    nn_list = main.define_nn_forwards_and_derivatives()
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")
    tm_full_list = utils_integral_geometry.make_tm_full_list(tm_grid, nodes_int)

    make_vtp(
        utils_nn.potential_guess,
        params_list,
        nn_list,
        mesh.nodes,
        tm_grid,
        tm_full_list,
        delta_tm,
        reductions,
        nodes_int,
        weights_int,
    )

    # Divergence(velocity)
    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    cloud = add_divergence(mesh)
    cloud.save("./results/divergence.vtk")


print("\nDone!")
