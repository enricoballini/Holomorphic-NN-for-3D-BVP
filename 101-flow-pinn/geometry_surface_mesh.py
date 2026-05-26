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

lc = 0.05  # edge size

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

gmsh.model.mesh.generate(3)

node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()

node_coords = node_coords.reshape((-1, 3))

elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.get_elements(dim=3)
tetrahedra = elem_node_tags[0].reshape((-1, 4))

# ---------------------------------------------------------------------
# Build face connectivity + mapping face -> tetra
# ---------------------------------------------------------------------

face_count = defaultdict(int)
face_to_tet = defaultdict(list)
face_list = []

for i, tet in enumerate(tetrahedra):
    faces = [
        tuple(sorted([tet[0], tet[1], tet[2]])),
        tuple(sorted([tet[0], tet[1], tet[3]])),
        tuple(sorted([tet[0], tet[2], tet[3]])),
        tuple(sorted([tet[1], tet[2], tet[3]])),
    ]

    for face in faces:
        face_count[face] += 1
        face_to_tet[face].append(i)

# Collect unique faces
triangles = []
idx_inner = []

for face, count in face_count.items():
    triangles.append(list(face))
    if count > 1:  # interior face
        idx_inner.append(len(triangles) - 1)

triangles = np.array(triangles, dtype=int)
idx_inner = np.array(idx_inner, dtype=int)

# ---------------------------------------------------------------------
# Geometry: areas, centroids, outward normals
# ---------------------------------------------------------------------

areas = []
centroids = []
normals = []

for tri in triangles:
    tri_t = tuple(tri)

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

    centroid = (p1 + p2 + p3) / 3.0

    # ---------------- OUTWARD NORMAL FIX ----------------
    adj_tet = face_to_tet[tri_t][0]
    tet = tetrahedra[adj_tet] - 1
    tet_centroid = node_coords[tet].mean(axis=0)

    if np.dot(unit_normal, centroid - tet_centroid) < 0:
        unit_normal *= -1
    # -----------------------------------------------------

    assert area > 0

    areas.append(area)
    centroids.append(centroid)
    normals.append(unit_normal)

areas = np.array(areas)
centroids = np.array(centroids)
normals = np.array(normals)

np.savetxt("./results/face_areas", areas)
np.savetxt("./results/centroids_coordinates", centroids)
np.savetxt("./results/face_normals", normals)
np.savetxt("./results/idx_inner", idx_inner)

# ---------------------------------------------------------------------
# define tags (manual as before)
# ---------------------------------------------------------------------

idx_all = np.arange(centroids.shape[0])

idx_in = np.where(np.isclose(centroids[:, 1], 0))[0]
idx_out = np.where(np.isclose(centroids[:, 0], 1))[0]
idx_wall = np.setdiff1d(idx_all, np.concatenate((idx_in, idx_out)))

tags = {"in": idx_in, "out": idx_out, "wall": idx_wall}

with open("./data/tags", "wb") as fle:
    pickle.dump(tags, fle)

# ---------------------------------------------------------------------

gmsh.write("./results/geometry.vtk")
gmsh.finalize()
