import os
from pdb import set_trace as st
import numpy as np


def compute_exact_solution():
    """ """
    stress_exact = np.load("./results/fem_stresses.npy")
    displ_exact = np.load("./results/fem_displacements.npy")

    return stress_exact * 0.001, displ_exact
