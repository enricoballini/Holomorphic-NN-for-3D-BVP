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
import utils_exact_solution
import main

dprint = jax.debug.print
jax.config.update("jax_enable_x64", True)


def make_vtp(mesh):
    """ """

    velocity = utils_exact_solution.compute_exact_solution()[mesh.inner_node_indices]

    # Create PolyData (point cloud)
    cloud = pv.PolyData(mesh.inner_nodes)

    # Add node values
    cloud.point_data["velocity FEM"] = jnp.real(velocity)

    # Save as .vtu (unstructured format)
    cloud.save("./results/potentials_fem.vtk")


if __name__ == "__main__":

    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    make_vtp(mesh)


print("\nDone!")
