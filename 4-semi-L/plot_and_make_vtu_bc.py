import os
from pdb import set_trace as st
import pickle
import numpy as np
import meshio
from meshio import CellBlock

import utils_nn
import case_settings


def write_train_test_datasets():
    point_coordinates, bc_vals, bc_type, normals, areas, tags = (
        case_settings.generate_boundary_data()
    )
    weights_loss = np.ones(point_coordinates.shape[0])

    (
        point_coordinates_train,
        _,
        _,
        _,
        _,
        _,
        _,
        point_coordinates_test,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = utils_nn.split_dataset(
        point_coordinates, bc_vals, bc_type, normals, areas, weights_loss, tags
    )

    n_train = len(point_coordinates_train)
    n_test = len(point_coordinates_test)

    train_mesh = meshio.Mesh(
        points=point_coordinates_train,
        cells=[CellBlock("vertex", np.arange(n_train).reshape(-1, 1))],
    )
    test_mesh = meshio.Mesh(
        points=point_coordinates_test,
        cells=[CellBlock("vertex", np.arange(n_test).reshape(-1, 1))],
    )

    meshio.write("./results/training_points.vtk", train_mesh)
    meshio.write("./results/test_points.vtk", test_mesh)


with open("./data/tags", "rb") as fle:
    tags = pickle.load(fle)

centroids = np.loadtxt("./results/centroids_coordinates")
normals = np.loadtxt("./results/face_normals")
mesh = meshio.read("./results/geometry.vtk")

n_geometry_points = len(mesh.points)

all_points = np.vstack([mesh.points, centroids])

cells = mesh.cells.copy()
centroid_indices = np.arange(n_geometry_points, len(all_points)).reshape(-1, 1)
centroid_block = CellBlock("vertex", centroid_indices)
cells.append(centroid_block)

zero_normals = np.zeros((n_geometry_points, 3))
point_data = {"Normals": np.vstack([zero_normals, normals])}

face_tags = np.zeros(len(centroids))
point_coordinates, bc_vals, bc_type, normals, areas, tags = (
    case_settings.generate_boundary_data()
)

color_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for key, color_val in zip(tags.keys(), color_vals):
    face_tags[tags[key]] = color_val

cell_data = {"tag": []}
for block in cells:
    if block.type in ("triangle", "tri"):
        cell_data["tag"].append(face_tags.tolist())
    else:
        cell_data["tag"].append([0] * len(block.data))

merged_mesh = meshio.Mesh(
    points=all_points, cells=cells, point_data=point_data, cell_data=cell_data
)

meshio.write("./results/geometry_and_bc.vtk", merged_mesh)

write_train_test_datasets()

print("\nDone!")
