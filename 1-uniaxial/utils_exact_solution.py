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

    w = 0.1

    for idx_pt, node_coordinate in enumerate(node_coordinates):
        stress_exact[idx_pt, 0, 0] = E * nu / ((1 + nu) * (1 - 2 * nu)) * w / L
        stress_exact[idx_pt, 1, 1] = E * nu / ((1 + nu) * (1 - 2 * nu)) * w / L
        stress_exact[idx_pt, 2, 2] = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)) * w / L

        displ_exact[idx_pt, 2] = node_coordinate[2] * w / L

    return stress_exact, displ_exact
