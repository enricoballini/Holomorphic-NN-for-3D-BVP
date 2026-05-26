import os
from pdb import set_trace as st
import numpy as np


def load_mesh_coordinates(filepath):
    """ """
    coordinates = np.loadtxt(
        filepath, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype=np.float64
    )

    return coordinates


def load_vel(filepath):
    vel = np.loadtxt(filepath, skiprows=22, usecols=(1, 2, 3), dtype=np.float64)
    return vel


nodes_coordinates = load_mesh_coordinates("./Mesh_node_coordinates.txt")
np.save("./results/fem_nodes_coordinates.npy", nodes_coordinates)

fem_vel = load_vel("./Field_output.txt")
np.save("./results/fem_vel.npy", fem_vel)


print("\nDone!")
