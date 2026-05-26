import os
from pdb import set_trace as st
import pickle
import numpy as np
import jax
import jax.numpy as jnp

import utils_data_and_folders
import utils_integral_geometry
import utils_nn
import utils_postprocess

import case_settings
import utils_exact_solution

jax.config.update("jax_enable_x64", True)


def norm_weighted_frobenius(array, volumes):
    """ """
    assert len(array.shape) > 1
    assert len(volumes.shape) == 1

    axis_to_sum = tuple(np.arange(len(array.shape) - 1) + 1)

    norm = np.sqrt(np.sum(volumes * np.sum(array**2, axis=axis_to_sum)))
    return norm


def compute_error(
    params_list,
    nn_list,
    tm_grid,
    tm_full_list,
    reductions,
    nodes_int,
    weights_int,
):
    """ """

    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    V_exact = utils_exact_solution.compute_exact_solution(mesh.tets_centroids)

    V_nn = utils_nn.potential_guess(
        params_list,
        nn_list,
        jnp.array(tm_grid),
        tm_full_list,
        nodes_int,
        weights_int,
        reductions,
        mesh.tets_centroids,
    )
    V_nn = np.array(np.real(V_nn)).ravel()

    # displacement errors: ------------------------
    volume_total = np.sum(mesh.tets_volumes)

    V_exact = V_exact[:, None]
    V_nn = V_nn[:, None]

    delta_V = V_nn - V_exact

    err_abs_distribution = np.sqrt(np.einsum("ij,ij -> i ", delta_V, delta_V))
    err_abs = norm_weighted_frobenius(delta_V, mesh.tets_volumes) / np.sqrt(
        volume_total
    )

    ref_V = norm_weighted_frobenius(V_exact, mesh.tets_volumes)

    err_rel_distribution = np.sqrt(volume_total) * err_abs_distribution / ref_V
    err_rel = norm_weighted_frobenius(delta_V, mesh.tets_volumes) / ref_V

    # from centroids to nodes -------------------------
    err_abs_distribution = utils_postprocess.interpolate_tet_centroid_values_to_nodes(
        mesh, err_abs_distribution
    )
    err_rel_distribution = utils_postprocess.interpolate_tet_centroid_values_to_nodes(
        mesh, err_rel_distribution
    )

    return (
        err_abs,
        err_abs_distribution,
        err_rel,
        err_rel_distribution,
    )


if __name__ == "__main__":

    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")
    tm_list, delta_tm = utils_nn.define_tm(M, bias_rotation)
    reductions = utils_nn.define_reductions()

    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")

    tm_full_list = utils_integral_geometry.make_tm_full_list(tm_list, nodes_int)

    idx_seed = 0
    params_list = utils_data_and_folders.load_params_list(idx_seed)
    nn_list = case_settings.define_nn_forwards_and_derivatives()

    (
        err_abs,
        err_abs_distribution,
        err_rel,
        err_rel_distribution,
    ) = compute_error(
        params_list,
        nn_list,
        tm_list,
        tm_full_list,
        reductions,
        nodes_int,
        weights_int,
    )

    np.savetxt("./results/err_abs", np.atleast_1d(err_abs))
    np.savetxt(
        "./results/err_abs_distribution",
        np.atleast_1d(err_abs_distribution),
    )
    np.savetxt("./results/err_rel", np.atleast_1d(err_rel))
    np.savetxt(
        "./results/err_rel_distribution",
        np.atleast_1d(err_rel_distribution),
    )

    print("\nDone!")
