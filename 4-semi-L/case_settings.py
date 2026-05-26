import os
import pdb
from pdb import set_trace as st
import pickle
import time
from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_map
import optax
import numpy as np

import utils_data_and_folders
import utils_integral_geometry
import utils_mechanics
import utils_nn

dprint = jax.debug.print


# boundary data --------------------------------


def generate_boundary_data():
    r"""
    The boudary conditions are implemented only in the robi form:

        $a \circ u + b \circ \sigma n = bc_val$

    where $\circ$ is the Hadamard product.

    It is up to the user to define properly the vectors a and b.
    """

    with open("./data/tags", "rb") as fle:
        tags = pickle.load(fle)

    tags = tree_map(lambda tag: tag.astype(int), tags)

    point_coordinates = np.loadtxt("./results/centroids_coordinates")
    normals = np.loadtxt("./results/face_normals")
    areas = np.loadtxt("./results/face_areas")

    n_pt = point_coordinates.shape[0]

    bc_type = np.zeros((n_pt, 2, 3))

    # define vectors a:
    bc_type[tags["fix"], 0] = np.array([1, 1, 1])
    bc_type[tags["supp"], 0] = np.array([1, 1, 1])

    # define vectors b:
    bc_type[tags["free"], 1] = np.array([1, 1, 1])
    bc_type[tags["crown"], 1] = np.array([1, 1, 1])

    bc_vals = np.zeros_like(point_coordinates)

    bc_vals[tags["supp"], 2] = 0.2

    return point_coordinates, bc_vals, bc_type, normals, areas, tags


def compute_weights_loss(areas, tags):
    """ """
    weights_loss = np.ones(areas.shape[0])
    weights_loss[tags["supp"]] = 1
    weights_loss[tags["free"]] = 1
    weights_loss[tags["fix"]] = 1
    weights_loss[tags["crown"]] = 1
    weights_loss[tags["ring"]] = 1
    return weights_loss


# model ----------------------------------------------------
def common(params, reduction, point_coordinates, t):
    """ """
    zeta = reduction(point_coordinates, t).reshape(-1, 1)

    z_hol = zeta
    z_gen = t

    for layer in params[:-1]:
        z_hol = utils_nn.activation_exp(
            z_hol @ layer["W_1"].T + z_gen @ layer["W_2"].T + layer["b_1"]
        )
        z_gen = utils_nn.activation_prelu(z_gen @ layer["W_3"].T + layer["b_2"])

    z_hol = (
        z_hol @ params[-1]["W_1"].T + z_gen @ params[-1]["W_2"].T + params[-1]["b_1"]
    )

    return jnp.real(z_hol)


def common_return_min_max(params, reduction, point_coordinates, t):
    """ """
    zeta = reduction(point_coordinates, t).reshape(-1, 1)

    z_hol = zeta
    z_gen = t

    mins = jnp.zeros(len(params) - 1)
    maxes = jnp.zeros(len(params) - 1)

    for idx_layer, layer in enumerate(params[:-1]):
        z_hol = utils_nn.activation_exp(
            z_hol @ layer["W_1"].T + z_gen @ layer["W_2"].T + layer["b_1"]
        )
        z_gen = utils_nn.activation_prelu(z_gen @ layer["W_3"].T + layer["b_2"])

        length = jnp.sqrt(z_hol * z_hol.conj()).astype(jnp.float32)
        mins = mins.at[idx_layer].set(np.min(length))
        maxes = maxes.at[idx_layer].set(np.max(length))

    z_hol = (
        z_hol @ params[-1]["W_1"].T + z_gen @ params[-1]["W_2"].T + params[-1]["b_1"]
    )

    return jnp.real(z_hol), mins, maxes


def define_nn_forwards_and_derivatives(return_min_max=False):
    """ """

    def chi_forward(params, reduction, point_coordinates, t):
        return common(params, reduction, point_coordinates, t)

    def phi_0_forward(params, reduction, point_coordinates, t):
        return common(params, reduction, point_coordinates, t)

    def phi_1_forward(params, reduction, point_coordinates, t):
        return common(params, reduction, point_coordinates, t)

    def phi_2_forward(params, reduction, point_coordinates, t):
        return common(params, reduction, point_coordinates, t)

    if return_min_max:

        def chi_forward(params, reduction, point_coordinates, t):
            return common_return_min_max(params, reduction, point_coordinates, t)

        def phi_0_forward(params, reduction, point_coordinates, t):
            return common_return_min_max(params, reduction, point_coordinates, t)

        def phi_1_forward(params, reduction, point_coordinates, t):
            return common_return_min_max(params, reduction, point_coordinates, t)

        def phi_2_forward(params, reduction, point_coordinates, t):
            return common_return_min_max(params, reduction, point_coordinates, t)

    nn_list = utils_nn.make_list_chi_phi_and_der(
        chi_forward,
        phi_0_forward,
        phi_1_forward,
        phi_2_forward,
    )
    return nn_list


def compute_min_max_single_tm(
    params_list, nn_list, name, point_coordinates, reductions, tm
):
    reduction = reductions[0]  ###

    _, mins, maxes = getattr(nn_list.der_0, name)(
        params_list[name], reduction, point_coordinates, tm
    )
    return mins, maxes


@partial(jax.jit, static_argnames=["reductions"])
def compute_input_max_layer(params_list, tm_full_list, reductions, point_coordinates):
    """ """
    nn_list = define_nn_forwards_and_derivatives(return_min_max=True)

    nn_names = [key for key in params_list if key != "coeffs"]
    mins_all = dict.fromkeys(nn_names)
    maxes_all = dict.fromkeys(nn_names)

    for name in nn_names:
        mins_all_tm, maxes_all_tm = jax.vmap(
            lambda tm: compute_min_max_single_tm(
                params_list,
                nn_list,
                name,
                point_coordinates,
                reductions,
                jnp.atleast_1d(tm),
            )
        )(tm_full_list)
        mins_all[name] = jnp.min(mins_all_tm, axis=0)
        maxes_all[name] = jnp.max(maxes_all_tm, axis=0)

    return mins_all, maxes_all


if __name__ == "__main__":
    os.system("clear")

    # PET
    nu = 0.38
    G = 1.1  # GPa

    np.savetxt("./data/nu", np.array([nu]))
    np.savetxt("./data/G", np.array([G]))

    M = 16
    bias_rotation = 0
    np.savetxt("./data/M", np.array([M]))
    np.savetxt("./data/bias_rotation", np.array([bias_rotation]))

    degree = 2
    nodes_int, weights_int = np.polynomial.legendre.leggauss(degree)
    np.savetxt("./data/nodes_int", nodes_int)
    np.savetxt("./data/weights_int", weights_int)

    default_architecture = np.array([2, 64, 64, 64, 64, 1])
    default_split = np.array([1, 32, 32, 32, 32, 1])  # holomorphic part

    np.savetxt("./data/default_architecture", default_architecture)
    np.savetxt("./data/default_split", default_split)

    seeds = np.array([[0, 1, 2, 3, 4, 5, 6]])
    np.savetxt("./data/seeds", seeds)
    np.savetxt("./results/idx_seeds", np.arange(seeds.shape[0]))

    print("\nDone!")
