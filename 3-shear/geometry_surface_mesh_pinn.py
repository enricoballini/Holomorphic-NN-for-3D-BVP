"""
We assume to use simplex elements
"""

import os
import pickle
from collections import defaultdict

from pdb import set_trace as st
import numpy as np
import gmsh
import meshio

gmsh.initialize()
gmsh.open("geometry.step")

lc = 0.3  # edge size
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

gmsh.model.mesh.generate(3)

node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()

node_coords = node_coords.reshape((-1, 3))

# Get tetrahedra
elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.get_elements(dim=3)
tetrahedra = elem_node_tags[0].reshape((-1, 4))

face_count = defaultdict(int)
face_list = []

# Extract all faces from tetrahedra
for i, tet in enumerate(tetrahedra):
    # Each tetrahedron has 4 triangular faces
    faces = [
        tuple(sorted([tet[0], tet[1], tet[2]])),
        tuple(sorted([tet[0], tet[1], tet[3]])),
        tuple(sorted([tet[0], tet[2], tet[3]])),
        tuple(sorted([tet[1], tet[2], tet[3]])),
    ]
    for face in faces:
        face_count[face] += 1

# Collect unique faces with proper orientation
triangles = []
idx_inner = []

for face, count in face_count.items():
    triangles.append(list(face))
    if count > 1:  # interior face
        idx_inner.append(len(triangles) - 1)

triangles = np.array(triangles, dtype=int)
idx_inner = np.array(idx_inner, dtype=int)

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

np.savetxt("./results/face_areas_3d", areas)
np.savetxt("./results/centroids_coordinates_3d", centroids)
np.savetxt("./results/face_normals_3d", normals)
np.savetxt("./results/idx_inner", idx_inner)

# define tags: -------------------------------------------

idx_all = np.arange(centroids.shape[0])

# idx_x = np.where(np.isclose(centroids[:, 0], 0.5))[0]  # right vertical surface
# idx__x = np.where(np.isclose(centroids[:, 0], -0.5))[0]  # left vertical surface

# idx_y = np.where(np.isclose(centroids[:, 1], 0.5))[0]
# idx__y = np.where(np.isclose(centroids[:, 1], -0.5))[0]

# idx_z = np.where(np.isclose(centroids[:, 2], 0.5))[0]  # top surface
# idx__z = np.where(np.isclose(centroids[:, 2], -0.5))[0]  # bottom surface

idx_x = np.where(np.isclose(centroids[:, 0], 1))[0]  # right vertical surface
idx__x = np.where(np.isclose(centroids[:, 0], 0))[0]  # left vertical surface

idx_y = np.where(np.isclose(centroids[:, 1], 1))[0]
idx__y = np.where(np.isclose(centroids[:, 1], 0))[0]

idx_z = np.where(np.isclose(centroids[:, 2], 1))[0]  # top surface
idx__z = np.where(np.isclose(centroids[:, 2], 0))[0]  # bottom surface

tags = {"x": idx_x, "-x": idx__x, "y": idx_y, "-y": idx__y, "-z": idx__z, "z": idx_z}

with open("./data/tags_3d", "wb") as fle:
    pickle.dump(tags, fle)

# ---------------------------------------------------------


# Write base surface mesh
gmsh.write("./results/geometry.vtk")
gmsh.finalize()
