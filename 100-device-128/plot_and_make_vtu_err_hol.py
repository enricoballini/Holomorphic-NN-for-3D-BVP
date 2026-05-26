""" """

from pdb import set_trace as st

import numpy as np
import vtk


import utils_postprocess


# ---------------------------------------------------------------------------
# add_*_errors functions
# ---------------------------------------------------------------------------


def add_V_errors(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """ """

    err_abs = np.loadtxt("./results/err_abs_distribution")
    err_rel = np.loadtxt("./results/err_rel_distribution")

    mesh.node_data["err_abs_distribution"] = err_abs
    mesh.node_data["err_rel_distribution"] = err_rel
    return mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh = add_V_errors(mesh)
    utils_postprocess.save_mesh(mesh, "./results/V_err.vtk")

    print("\nDone!")
