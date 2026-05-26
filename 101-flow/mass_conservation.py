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


def compute_delta_mass():
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

    point_coordinates_boundary, areas, normals, point_coordinates_inner = (
        utils_postprocess.make_postprocess_domain_boundary_and_inner()
    )

    velocity = utils_lapl.compute_gradients(
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
        point_coordinates_boundary,
    )

    tags = {}
    tags["in"] = np.where(np.isclose(point_coordinates_boundary[:, 1], 0))[0]
    tags["out"] = np.where(np.isclose(point_coordinates_boundary[:, 0], 1))[0]

    u_n_in = np.einsum("ij,ij->i", velocity[tags["in"]], normals[tags["in"]])
    u_n_out = np.einsum("ij,ij->i", velocity[tags["out"]], normals[tags["out"]])

    mass_in = np.sum(u_n_in * areas[tags["in"]])
    mass_out = np.sum(u_n_out * areas[tags["out"]])

    return mass_in + mass_out


if __name__ == "__main__":

    print("\ndiv(u) is temporary computed and saved in plot_and_make_vtu.py")

    mass_in_minus_out = compute_delta_mass()

    vel_in = 1
    area = np.pi * 1**2
    mass_flux = vel_in * area

    print("\ndelta mass = ", mass_in_minus_out)
    print("\ndelta mass / mass_flux = ", mass_in_minus_out / mass_flux)

    np.savetxt("./results/delta_mass", np.array([mass_in_minus_out]))


print("\nDone!")
