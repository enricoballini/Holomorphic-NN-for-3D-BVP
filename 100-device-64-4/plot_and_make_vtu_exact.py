""" """

from pdb import set_trace as st
import numpy as np
import vtk


import utils_exact_solution
import utils_postprocess


def add_V_exact(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """ """
    V_exact = utils_exact_solution.compute_exact_solution(mesh.nodes)

    mesh.node_data["V_exact"] = V_exact
    return mesh


if __name__ == "__main__":

    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh = add_V_exact(mesh)
    utils_postprocess.save_mesh(mesh, "./results/V_exact.vtk")

    print("\nDone!")
