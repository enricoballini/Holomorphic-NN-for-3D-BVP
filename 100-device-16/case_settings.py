""" """

import os
import pdb
from pdb import set_trace as st
import pickle
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np

import utils_data_and_folders
import utils_integral_geometry
import utils_nn
import utils_exact_solution

dprint = jax.debug.print


def generate_boundary_data():
    """ """
    with open("./data/tags", "rb") as fle:
        tags = pickle.load(fle)

    tags = tree_map(lambda tag: tag.astype(int), tags)

    point_coordinates = np.loadtxt("./results/centroids_coordinates")
    normals = np.loadtxt("./results/face_normals")
    areas = np.loadtxt("./results/face_areas")

    n_pt = point_coordinates.shape[0]

    bc_type = np.zeros((n_pt, 2))

    # define vectors a:
    bc_type[tags["fix"]] = np.array([1])
    bc_type[tags["supp"]] = np.array([1])
    bc_type[tags["free"]] = np.array([1])

    bc_vals = utils_exact_solution.compute_exact_solution(point_coordinates)

    return point_coordinates, bc_vals, bc_type, normals, areas, tags


def compute_weights_loss(areas, tags):
    """ """
    weights_loss = np.ones(areas.shape[0])
    weights_loss[tags["supp"]] = 1
    return weights_loss


# model ----------------------------------------------------


def define_nn_forwards_and_derivatives():
    """ """

    def common(params, zeta, t):
        zeta /= 300  # scaling

        z_hol = zeta
        z_gen = t

        for layer in params[:-1]:
            z_hol = utils_nn.activation_exp(
                z_hol @ layer["W_1"].T + z_gen @ layer["W_2"].T + layer["b_1"]
            )
            z_gen = utils_nn.activation_prelu(z_gen @ layer["W_3"].T + layer["b_2"])

        z_hol = (
            z_hol @ params[-1]["W_1"].T
            + z_gen @ params[-1]["W_2"].T
            + params[-1]["b_1"]
        )

        return z_hol

    def chi_0_forward(params, z, t):
        """ """
        return common(params, z, t)

    nn_list = utils_nn.make_list_chi_phi_and_der(
        chi_0_forward,
    )
    return nn_list


if __name__ == "__main__":

    os.system("clear")

    M = 16
    bias_rotation = -(2 * np.pi) / (M - 1) / 2
    np.savetxt("./data/M", np.array([M]))
    np.savetxt("./data/bias_rotation", np.array([bias_rotation]))

    degree = 1
    nodes_int, weights_int = np.polynomial.legendre.leggauss(degree)
    np.savetxt("./data/nodes_int", nodes_int)
    np.savetxt("./data/weights_int", weights_int)

    default_architecture = [2, 16, 16, 16, 16, 1]
    default_split = [1, 8, 8, 8, 8, 1]
    np.savetxt("./data/default_architecture", default_architecture)
    np.savetxt("./data/default_split", default_split)

    seeds = np.array([[0, 1, 2, 3, 4, 5, 6]])
    np.savetxt("./data/seeds", seeds)
    np.savetxt("./results/idx_seeds", np.arange(seeds.shape[0]))
