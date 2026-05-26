import os
from pdb import set_trace as st
import pickle
import numpy as np
import jax

import utils_data_and_folders
import utils_integral_geometry
import utils_nn
import utils_postprocess

import main
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
):
    """ """

    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    vel_exact_nodes = utils_exact_solution.compute_exact_solution()

    vel_exact = utils_postprocess.interpolate_node_values_to_tet_centroids(
        mesh, vel_exact_nodes
    )

    vel_nn = utils_postprocess._batch_vel(
        params_list,
        nn_list,
        mesh.tets_centroids,
    )

    # displacement errors: ------------------------
    sqrt_volume_total = np.sqrt(np.sum(mesh.tets_volumes))
    delta_vel = vel_nn - vel_exact

    err_abs_distribution = np.sqrt(np.einsum("ij,ij -> i ", delta_vel, delta_vel))
    err_abs = norm_weighted_frobenius(delta_vel, mesh.tets_volumes) / sqrt_volume_total

    ref_vel = norm_weighted_frobenius(vel_exact, mesh.tets_volumes)

    err_rel_distribution = sqrt_volume_total * err_abs_distribution / ref_vel
    err_rel = norm_weighted_frobenius(delta_vel, mesh.tets_volumes) / ref_vel

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

    idx_seed = 0
    params_list = utils_data_and_folders.load_params_list(idx_seed)

    nn_list = main.define_nn_forwards_and_derivatives()

    (
        err_abs,
        err_abs_distribution,
        err_rel,
        err_rel_distribution,
    ) = compute_error(
        params_list,
        nn_list,
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
