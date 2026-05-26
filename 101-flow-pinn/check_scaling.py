import os
from pdb import set_trace as st
import pickle

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import vtk

import utils_data_and_folders
import utils_nn
import utils_postprocess
import main


nu = np.loadtxt("./data/nu")
G = np.loadtxt("./data/G")

point_coordinates, bc_vals, bc_type, normals, areas, tags = (
    main.generate_boundary_data()
)

M = np.loadtxt("./data/M").astype(int)
tm_list, delta_tm = utils_nn.define_tm(M)
reductions = utils_nn.define_reductions()
tm = tm_list[0]

idx_seed = 0
params_list = utils_data_and_folders.load_params_list(idx_seed)
nn_list = main.define_nn_forwards_and_derivatives()
zetas = utils_nn.compute_zetas(point_coordinates, tm, reductions)

# print(nn_list.der_0.phi_0(params_list["phi_0"], zetas[0]))
# print(nn_list.der_0.phi_1(params_list["phi_1"], zetas[1]))
# print(nn_list.der_0.phi_2(params_list["phi_2"], zetas[2]))

# print(nn_list.der_0.chi_0(params_list["chi_0"], zetas[0]))
# print(nn_list.der_0.chi_1(params_list["chi_1"], zetas[1]))
# print(nn_list.der_0.chi_2(params_list["chi_2"], zetas[2]))

tm = jnp.array([tm_list[0]])
print(
    jnp.max(jnp.abs(nn_list.der_0.phi_0(params_list["phi_0"], zetas[0], tm)))
)  # checked for only one tm
print(jnp.max(jnp.abs(nn_list.der_0.phi_1(params_list["phi_1"], zetas[1], tm))))
print(jnp.max(jnp.abs(nn_list.der_0.phi_2(params_list["phi_2"], zetas[2], tm))))
print(jnp.max(jnp.abs(nn_list.der_0.chi_0(params_list["chi_0"], zetas[0], tm))))
print(jnp.max(jnp.abs(nn_list.der_0.chi_1(params_list["chi_1"], zetas[1], tm))))
print(jnp.max(jnp.abs(nn_list.der_0.chi_2(params_list["chi_2"], zetas[2], tm))))

print("\n")

# for params in params_list.items():
#     tree_map(
#         lambda param: (
#             print(jnp.max(jnp.abs(param)))
#             if not isinstance(param, str)
#             else print(param)  # the name
#         ),
#         params,
#     )


print("\nDone!")
