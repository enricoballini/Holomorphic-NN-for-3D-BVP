import os
from pdb import set_trace as st
import pickle
import numpy as np
import meshio
from meshio import CellBlock


with open("./data/tags", "rb") as fle:
    tags = pickle.load(fle)

centroids = np.loadtxt("./results/centroids_coordinates")
normals = np.loadtxt("./results/face_normals")
mesh = meshio.read("./results/geometry.vtk")


all_points = np.vstack([mesh.points, centroids])

cells = mesh.cells.copy()

centroid_indices = np.arange(len(mesh.points), len(all_points)).reshape(-1, 1)
centroid_block = CellBlock("vertex", centroid_indices)

cells.append(centroid_block)


zero_normals = np.zeros((len(mesh.points), 3))
point_data = {"Normals": np.vstack([zero_normals, normals])}


face_tags = np.zeros(len(centroids))

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

meshio.write("./results/geometry.vtk", merged_mesh)
print("\nDone!")
