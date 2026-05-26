"""
We assume to use simplex elements
"""

import os
import pickle
from pdb import set_trace as st
import numpy as np
import gmsh
import meshio

gmsh.initialize()
gmsh.open("geometry.step")

lc = 5  # edge size
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

gmsh.model.mesh.generate(2)

node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()

scaling = 1
np.savetxt("./results/scaling", np.array([scaling]))
print(f"\n\n ----------- Scaling: {scaling} ----------\n\n")

node_coords = node_coords.reshape((-1, 3)) / scaling

elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.get_elements(dim=2)
triangles = elem_node_tags[0].reshape((-1, 3))

areas = []
centroids = []
normals = []

for tri in triangles:
    p1, p2, p3 = (
        node_coords[tri[0] - 1],
        node_coords[tri[1] - 1],
        node_coords[tri[2] - 1],
    )

    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(normal)
    unit_normal = normal / np.linalg.norm(normal)

    assert area > 0

    centroid = (p1 + p2 + p3) / 3.0

    areas.append(area)
    centroids.append(centroid)
    normals.append(unit_normal)

areas = np.array(areas)
centroids = np.array(centroids)
normals = np.array(normals)

np.savetxt("./results/face_areas", areas)
np.savetxt("./results/centroids_coordinates", centroids)
np.savetxt("./results/face_normals", normals)

# define tags: -------------------------------------------

idx_all = np.arange(centroids.shape[0])

idx_fix = np.where(np.isclose(centroids[:, 1], 0))[0]
idx_supp = np.where(np.isclose(centroids[:, 2], 55 / scaling))[0]
idx_free = np.setdiff1d(idx_all, np.concatenate((idx_fix, idx_supp)))

tags = {"fix": idx_fix, "supp": idx_supp, "free": idx_free}

with open("./data/tags", "wb") as fle:
    pickle.dump(tags, fle)

# ---------------------------------------------------------


# Write base surface mesh
gmsh.write("./results/geometry.vtk")
gmsh.finalize()
