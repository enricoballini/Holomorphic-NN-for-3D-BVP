""" """

from pdb import set_trace as st

import numpy as np
import vtk


import case_settings
import utils_exact_solution
import utils_data_and_folders
import utils_mechanics
import utils_nn
import utils_postprocess


def _load_hol_params():
    """ """
    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")
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
        nu,
        G,
        nodes_int,
        weights_int,
        tm_list,
        tm_full_list,
        reductions,
        params_list,
        nn_list,
    )


# ---------------------------------------------------------------------------
# add_*_errors functions
# ---------------------------------------------------------------------------


def add_stress_errors(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """ """

    err_abs = np.loadtxt("./results/err_abs_distribution_stress")
    err_rel = np.loadtxt("./results/err_rel_distribution_stress")

    mesh.node_data["err_abs_distribution_stress"] = err_abs
    mesh.node_data["err_rel_distribution_stress"] = err_rel
    return mesh


def add_displacement_errors(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """ """

    err_abs = np.loadtxt("./results/err_abs_distribution_displ")
    err_rel = np.loadtxt("./results/err_rel_distribution_displ")

    mesh.node_data["err_abs_distribution_displ"] = err_abs
    mesh.node_data["err_rel_distribution_displ"] = err_rel

    return mesh


def add_traction_errors(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """ """
    err_abs = np.loadtxt("./results/err_abs_distribution_traction")
    err_rel = np.loadtxt("./results/err_rel_distribution_traction")

    mesh.face_data["err_abs_distribution_traction"] = err_abs
    mesh.face_data["err_rel_distribution_traction"] = err_rel

    return mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh = add_stress_errors(mesh)
    mesh = add_traction_errors(mesh)
    utils_postprocess.save_mesh(mesh, "./results/errors_stress_tractions_hol.vtk")

    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh = add_displacement_errors(mesh)
    utils_postprocess.save_mesh(mesh, "./results/errors_displacement_hol.vtk")

    print("\nDone!")
