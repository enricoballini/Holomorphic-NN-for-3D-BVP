import os
from pdb import set_trace as st
import pickle
import numpy as np
import jax

import utils_data_and_folders_pinn
import utils_integral_geometry
import utils_mechanics_pinn
import utils_nn_pinn
import utils_postprocess_pinn

import case_settings_pinn
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
    tm_list,
    tm_full_list,
    reductions,
    nu,
    G,
    nodes_int,
    weights_int,
):
    """ """

    mesh = utils_postprocess_pinn.make_postprocess_domain_plot_boundary_and_inner()
    stress_exact, displ_exact = utils_exact_solution.compute_exact_solution(
        mesh.tets_centroids
    )

    stress_exact_boundary, _ = utils_exact_solution.compute_exact_solution(
        mesh.boundary_triangle_centroids
    )
    traction_exact = utils_mechanics_pinn.compute_sigma_n(
        stress_exact_boundary, mesh.boundary_outward_normals
    )

    # stress_nn = utils_nn_pinn.compute_sigma_tensor(
    #     params_list,
    #     nn_list,
    #     tm_grid,
    #     tm_full_list,
    #     delta_tm,
    #     point_coordinates,
    #     reductions,
    #     nu,
    #     G,
    #     nodes_int,
    #     weights_int,
    # )

    # displ_nn = utils_nn_pinn.compute_displacement(
    #     params_list,
    #     nn_list,
    #     tm_grid,
    #     tm_full_list,
    #     delta_tm,
    #     point_coordinates,
    #     reductions,
    #     nu,
    #     G,
    #     nodes_int,
    #     weights_int,
    # )

    jacobian_field = utils_mechanics_pinn.define_strain_from_displ()

    stress_nn = utils_postprocess_pinn._batched_sigma_from_displ(
        jacobian_field,
        tm_list,
        tm_full_list,
        params_list,
        nn_list,
        mesh.tets_centroids,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    stress_nn_boundary = utils_postprocess_pinn._batched_sigma_from_displ(
        jacobian_field,
        tm_list,
        tm_full_list,
        params_list,
        nn_list,
        mesh.boundary_triangle_centroids,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    displ_nn = utils_postprocess_pinn._batched_displacement(
        params_list,
        nn_list,
        tm_list,
        tm_full_list,
        mesh.tets_centroids,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    traction_nn = utils_mechanics_pinn.compute_sigma_n(
        stress_nn_boundary, mesh.boundary_outward_normals
    )

    # displacement errors: ------------------------
    sqrt_volume_total = np.sqrt(np.sum(mesh.tets_volumes))
    delta_displ = displ_nn - displ_exact

    err_abs_distribution_displ = np.sqrt(
        np.einsum("ij,ij -> i ", delta_displ, delta_displ)
    )
    err_abs_displ = (
        norm_weighted_frobenius(delta_displ, mesh.tets_volumes) / sqrt_volume_total
    )

    ref_displ = norm_weighted_frobenius(displ_exact, mesh.tets_volumes)

    err_rel_distribution_displ = (
        sqrt_volume_total * err_abs_distribution_displ / ref_displ
    )
    err_rel_displ = norm_weighted_frobenius(delta_displ, mesh.tets_volumes) / ref_displ

    # stress erros: ------------------------------
    delta_stress = stress_nn - stress_exact

    err_abs_distribution_stress = np.sqrt(
        np.einsum("ijk,ijk -> i ", delta_stress, delta_stress)
    )
    err_abs_stress = (
        norm_weighted_frobenius(delta_stress, mesh.tets_volumes) / sqrt_volume_total
    )

    ref_stress = norm_weighted_frobenius(stress_exact, mesh.tets_volumes)

    err_rel_distribution_stress = (
        sqrt_volume_total * err_abs_distribution_stress / ref_stress
    )
    err_rel_stress = (
        norm_weighted_frobenius(delta_stress, mesh.tets_volumes) / ref_stress
    )

    # traction errors: -----------------------------
    area_total = np.sum(mesh.boundary_triangle_areas)

    delta_traction = traction_nn - traction_exact

    err_abs_distribution_traction = np.sqrt(
        np.einsum("ij,ij -> i ", delta_traction, delta_traction)
    )
    err_abs_traction = (
        norm_weighted_frobenius(delta_traction, mesh.boundary_triangle_areas)
        / area_total
    )

    ref_traction = norm_weighted_frobenius(traction_exact, mesh.boundary_triangle_areas)

    err_rel_distribution_traction = (
        area_total * err_abs_distribution_traction / ref_traction
    )
    err_rel_traction = (
        norm_weighted_frobenius(delta_traction, mesh.boundary_triangle_areas)
        / ref_traction
    )

    # from centroids to nodes -------------------------
    err_abs_distribution_displ = (
        utils_postprocess_pinn.interpolate_tet_centroid_values_to_nodes(
            mesh, err_abs_distribution_displ
        )
    )
    err_rel_distribution_displ = (
        utils_postprocess_pinn.interpolate_tet_centroid_values_to_nodes(
            mesh, err_rel_distribution_displ
        )
    )

    err_abs_distribution_stress = (
        utils_postprocess_pinn.interpolate_tet_centroid_values_to_nodes(
            mesh, err_abs_distribution_stress
        )
    )
    err_rel_distribution_stress = (
        utils_postprocess_pinn.interpolate_tet_centroid_values_to_nodes(
            mesh, err_rel_distribution_stress
        )
    )

    return (
        err_abs_displ,
        err_abs_distribution_displ,
        err_rel_displ,
        err_rel_distribution_displ,
        #
        err_abs_stress,
        err_abs_distribution_stress,
        err_rel_stress,
        err_rel_distribution_stress,
        #
        err_abs_traction,
        err_abs_distribution_traction,
        err_rel_traction,
        err_rel_distribution_traction,
    )


if __name__ == "__main__":
    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")

    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")
    tm_list, delta_tm = utils_nn_pinn.define_tm(M, bias_rotation)
    reductions = utils_nn_pinn.define_reductions()

    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")

    tm_full_list = utils_integral_geometry.make_tm_full_list(tm_list, nodes_int)

    idx_seed = 0
    params_list = utils_data_and_folders_pinn.load_params_list(idx_seed)
    nn_list = case_settings_pinn.define_nn_forwards_and_derivatives()

    (
        err_abs_displ,
        err_abs_distribution_displ,
        err_rel_displ,
        err_rel_distribution_displ,
        err_abs_stress,
        err_abs_distribution_stress,
        err_rel_stress,
        err_rel_distribution_stress,
        err_abs_traction,
        err_abs_distribution_traction,
        err_rel_traction,
        err_rel_distribution_traction,
    ) = compute_error(
        params_list,
        nn_list,
        tm_list,
        tm_full_list,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    # displacement:
    np.savetxt("./results/err_abs_displ_pinn", np.atleast_1d(err_abs_displ))
    np.savetxt(
        "./results/err_abs_distribution_displ_pinn",
        np.atleast_1d(err_abs_distribution_displ),
    )
    np.savetxt("./results/err_rel_displ_pinn", np.atleast_1d(err_rel_displ))
    np.savetxt(
        "./results/err_rel_distribution_displ_pinn",
        np.atleast_1d(err_rel_distribution_displ),
    )

    # stress:
    np.savetxt("./results/err_abs_stress_pinn", np.atleast_1d(err_abs_stress))
    np.savetxt(
        "./results/err_abs_distribution_stress_pinn",
        np.atleast_1d(err_abs_distribution_stress),
    )
    np.savetxt("./results/err_rel_stress_pinn", np.atleast_1d(err_rel_stress))
    np.savetxt(
        "./results/err_rel_distribution_stress_pinn",
        np.atleast_1d(err_rel_distribution_stress),
    )

    # tractions
    np.savetxt("./results/err_abs_traction_pinn", np.atleast_1d(err_abs_traction))
    np.savetxt(
        "./results/err_abs_distribution_traction_pinn",
        np.atleast_1d(err_abs_distribution_traction),
    )
    np.savetxt("./results/err_rel_traction_pinn", np.atleast_1d(err_rel_traction))
    np.savetxt(
        "./results/err_rel_distribution_traction_pinn",
        np.atleast_1d(err_rel_distribution_traction),
    )

    print("\nDone!")
