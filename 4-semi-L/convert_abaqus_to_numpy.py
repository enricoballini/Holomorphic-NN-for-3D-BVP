import os
from pdb import set_trace as st
import numpy as np


def load_mesh_coordinates(filepath):
    """ """
    coordinates = np.loadtxt(
        filepath, delimiter=",", skiprows=2, usecols=(1, 2, 3), dtype=np.float64
    )

    return coordinates


def load_displacements(filepath):
    displacements = np.loadtxt(
        filepath, skiprows=22, usecols=(1, 2, 3), dtype=np.float64
    )
    return displacements


def load_stresses(filepath):
    sigma_vector = np.loadtxt(
        filepath, skiprows=22, usecols=(4, 5, 6, 7, 8, 9), dtype=np.float64
    )

    stresses = np.stack(
        (
            sigma_vector[:, 0],
            sigma_vector[:, 3],
            sigma_vector[:, 4],
            sigma_vector[:, 3],
            sigma_vector[:, 1],
            sigma_vector[:, 5],
            sigma_vector[:, 4],
            sigma_vector[:, 5],
            sigma_vector[:, 2],
        ),
        axis=1,
    ).reshape(-1, 3, 3)

    return stresses


nodes_coordinates = load_mesh_coordinates("./Mesh_node_coordinates.txt")
np.save("./results/fem_nodes_coordinates.npy", nodes_coordinates)

displacements = load_displacements("./Field_output.txt")
np.save("./results/fem_displacements.npy", displacements)

stresses = load_stresses("./Field_output.txt")
np.save("./results/fem_stresses.npy", stresses)


print("\nDone!")
