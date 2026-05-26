import os
from pdb import set_trace as st
import numpy as np


def compute_exact_solution(node_coordinates):
    """ """
    L = np.loadtxt("./data/L")
    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")
    E = 2 * (1 + nu) * G

    n_pt = node_coordinates.shape[0]
    stress_exact = np.zeros((n_pt, 3, 3))
    displ_exact = np.zeros((n_pt, 3))

    tau = 0.5

    for idx_pt, node_coordinate in enumerate(node_coordinates):
        stress_exact[idx_pt, 1, 2] = tau
        stress_exact[idx_pt, 2, 1] = tau

        displ_exact[idx_pt, 1] = 2 * (1 + nu) / E * tau * node_coordinate[2]

    return stress_exact, displ_exact
