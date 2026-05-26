import os
from pdb import set_trace as st
import numpy as np


def compute_exact_solution(node_coordinates):
    """ """
    L = np.loadtxt("./data/L")
    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")

    n_pt = node_coordinates.shape[0]
    stress_exact = np.zeros((n_pt, 3, 3))
    displ_exact = np.zeros((n_pt, 3))

    w = 0.1

    for idx_pt, node_coordinate in enumerate(node_coordinates):
        stress_exact[idx_pt, 0, 0] = 2 * 2 * G * nu / (1 - 2 * nu) * w / L
        stress_exact[idx_pt, 1, 1] = 2 * G / (1 - 2 * nu) * w / L
        stress_exact[idx_pt, 2, 2] = 2 * G / (1 - 2 * nu) * w / L
        displ_exact[idx_pt, 1] = w / L * node_coordinate[1]
        displ_exact[idx_pt, 2] = w / L * node_coordinate[2]

    return stress_exact, displ_exact
