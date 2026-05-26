import os
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

import utils_nn_pinn
import utils_exact_solution

dprint = jax.debug.print


def generate_boundary_data():
    r"""
    The boudary conditions are implemented only in the robi form:

        $a \circ u + b \circ \sigma n = bc_val$

    where $\circ$ is the Hadamard product.

    It is up to the user to define properly the vectors a and b.
    """

    with open("./data/tags_3d", "rb") as fle:
        tags = pickle.load(fle)

    tags = tree_map(lambda tag: tag.astype(int), tags)

    point_coordinates = np.loadtxt("./results/centroids_coordinates_3d")
    idx_inner = np.loadtxt("./results/idx_inner").astype(int)
    normals = np.loadtxt("./results/face_normals_3d")
    areas = np.loadtxt("./results/face_areas_3d")

    n_pt = point_coordinates.shape[0]

    mask_inner_points = np.zeros(n_pt)
    mask_inner_points[idx_inner] = 1

    bc_type = np.zeros((n_pt, 2, 3))

    # define vectors a:
    bc_type[tags["-z"], 0] = np.array([1, 1, 1])

    # define vectors b:
    bc_type[tags["-x"], 1] = np.array([1, 1, 1])
    bc_type[tags["x"], 1] = np.array([1, 1, 1])
    bc_type[tags["-y"], 1] = np.array([1, 1, 1])
    bc_type[tags["y"], 1] = np.array([1, 1, 1])
    bc_type[tags["z"], 1] = np.array([1, 1, 1])

    stress_exact, displ_exact = utils_exact_solution.compute_exact_solution(
        point_coordinates
    )
    bc_vals = np.einsum("ijk,ik->ij", stress_exact, normals)
    bc_vals[tags["-z"]] = displ_exact[tags["-z"]]

    return (
        point_coordinates,
        jnp.array(mask_inner_points),
        bc_vals,
        bc_type,
        normals,
        areas,
        tags,
    )


def compute_weights_loss(areas, tags):
    """ """
    weights_loss = np.ones(areas.shape[0])
    return weights_loss


# model ----------------------------------------------------
def common(params, reduction, point_coordinates, t):
    """ """
    z = point_coordinates

    for layer in params[:-1]:
        z = utils_nn_pinn.activation_exp(z @ layer["W"].T + layer["b"])

    z = z @ params[-1]["W"].T + params[-1]["b"]

    return z


def common_return_min_max(params, reduction, point_coordinates, t):
    """ """
    z = point_coordinates

    mins = jnp.zeros(len(params) - 1)
    maxes = jnp.zeros(len(params) - 1)

    for idx_layer, layer in enumerate(params[:-1]):
        z = utils_nn_pinn.activation_exp(z @ layer["W"].T + layer["b"])

        length = jnp.sqrt(z * z).astype(jnp.float32)
        mins = mins.at[idx_layer].set(np.min(length))
        maxes = maxes.at[idx_layer].set(np.max(length))

    z = z @ params[-1]["W"].T + params[-1]["b"]

    return z, mins, maxes


def define_nn_forwards_and_derivatives(return_min_max=False):
    """ """

    def chi_forward(params, reduction, point_coordinates, t):
        return common(params, reduction, point_coordinates, t)

    if return_min_max:

        def chi_forward(params, reduction, point_coordinates, t):
            return common_return_min_max(params, reduction, point_coordinates, t)

    nn_list = utils_nn_pinn.make_list_chi_phi_and_der(
        chi_forward,
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

    nu = 0.25
    G = 1  # GPa

    np.savetxt("./data/nu", np.array([nu]))
    np.savetxt("./data/G", np.array([G]))

    M = 32
    bias_rotation = -(2 * np.pi) / (M - 1) / 2
    np.savetxt("./data/M", np.array([M]))
    np.savetxt("./data/bias_rotation", np.array([bias_rotation]))

    degree = 1
    nodes_int, weights_int = np.polynomial.legendre.leggauss(degree)
    np.savetxt("./data/nodes_int", nodes_int)
    np.savetxt("./data/weights_int", weights_int)

    default_architecture = np.array([3, 4 * 32, 3]).astype(int)
    default_split = np.array([1, 16, 1])

    np.savetxt("./data/default_architecture_pinn", default_architecture)
    np.savetxt("./data/default_split", default_split)

    seeds = np.array([[0, 1, 2, 3, 4, 5, 6]])
    np.savetxt("./data/seeds", seeds)
    np.savetxt("./results/idx_seeds", np.arange(seeds.shape[0]))

    print("\nDone!")
