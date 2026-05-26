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

lc_global = 0.05
lc_fine = 0.01

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc_global)


# add embedded pts on circular surface:
circular_face_tag = 1
xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, circular_face_tag)
r = 0.2
num_embedded_points = 5
thetas = np.linspace(
    0, 2 * np.pi - 2 * np.pi / (num_embedded_points + 1), num=num_embedded_points
)
for theta in thetas:
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2 + r * np.sin(theta)
    cz = (zmin + zmax) / 2 + r * np.cos(theta)

    pt = gmsh.model.geo.addPoint(cx, cy, cz, lc_global)
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.embed(0, [pt], 2, circular_face_tag)

# add embedded pts on square surface:
square_face_tag = 11
r = 0.2
thetas = [1 / 4 * np.pi, 3 / 4 * np.pi, 5 / 4 * np.pi, 7 / 4 * np.pi]
for theta in thetas:
    cx = 0.25 + r * np.sin(theta)
    cy = 1
    cz = 0.25 + r * np.cos(theta)

    pt = gmsh.model.geo.addPoint(cx, cy, cz, lc_global)
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.embed(0, [pt], 2, square_face_tag)

target_curve_tags_circle = [1, 2, 3, 4, 5, 6, 7, 8]
target_curve_tags_square = [10, 20, 25, 27]
target_curve_tags = target_curve_tags_circle + target_curve_tags_square

gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", target_curve_tags)
gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_fine)
gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_global)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.01)
gmsh.model.mesh.field.setNumber(2, "DistMax", 0.011)

gmsh.model.mesh.field.setAsBackgroundMesh(2)

gmsh.model.mesh.generate(2)

node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()

node_coords = node_coords.reshape((-1, 3))

elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.get_elements(dim=2)
triangles = elem_node_tags[0].reshape((-1, 3))

areas = []
centroids = []
normals = []
ts_1 = []
ts_2 = []


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

    t1 = v1 - np.dot(v1, unit_normal) * unit_normal
    t1 /= np.linalg.norm(t1)

    t2 = np.cross(unit_normal, t1)
    t2 /= np.linalg.norm(t2)

    areas.append(area)
    centroids.append(centroid)
    normals.append(unit_normal)
    ts_1.append(t1)
    ts_2.append(t2)

areas = np.array(areas)
centroids = np.array(centroids)
normals = np.array(normals)
ts_1 = np.array(ts_1)
ts_2 = np.array(ts_2)

np.savetxt("./results/face_areas", areas)
np.savetxt("./results/centroids_coordinates", centroids)
np.savetxt("./results/face_normals", normals)
np.savetxt("./results/face_tangents_1", ts_1)
np.savetxt("./results/face_tangents_2", ts_2)

# ── Tangents on circular face (orthogonal to radial direction) ──────────────

# Re-query elements that belong only to the circular face
_, elem_tags_circ, _ = gmsh.model.mesh.get_elements(dim=2, tag=circular_face_tag)
circ_elem_tags = elem_tags_circ[0]  # shape: (N_circ_tris,)

# elem_tags[0] holds the global triangle-element tags in the same order
all_elem_tags = elem_tags[0]  # shape: (N_all_tris,)
circ_indices = np.where(np.isin(all_elem_tags, circ_elem_tags))[0]

# Circle centre: mid-point of the face bounding box
xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, circular_face_tag)
circle_center = np.array(
    [
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        (zmin + zmax) / 2,
    ]
)

tangents_radial = np.zeros((len(circ_indices), 3))

for out_i, idx in enumerate(circ_indices):
    n = normals[idx]  # unit face normal at this triangle

    # Radial vector: centre → centroid, then projected onto the face plane
    # (removes any out-of-plane drift caused by surface curvature / STEP tolerance)
    radial = centroids[idx] - circle_center
    radial = radial - np.dot(radial, n) * n
    radial /= np.linalg.norm(radial)

    # In-plane vector orthogonal to radial = n × radial
    tangent = np.cross(n, radial)
    tangent /= np.linalg.norm(tangent)

    tangents_radial[out_i] = tangent

np.savetxt("./results/pt_idx_tangents_radial", circ_indices)
np.savetxt("./results/tangents_radial", tangents_radial)

# define tags: -------------------------------------------

idx_all = np.arange(centroids.shape[0])

idx_fix = np.where(np.isclose(centroids[:, 0], 1))[0]
idx_supp = np.where(np.isclose(centroids[:, 1], 1))[0]
idx_crown = np.where((centroids[:, 0] < 1) * (centroids[:, 0] > 0.95))[0]
idx_ring = np.where(
    np.isclose(centroids[:, 0], 1)
    * np.sqrt((centroids[:, 1] - 0.25) ** 2 + (centroids[:, 2] - 0.25) ** 2)
    > 0.2
)[0]
idx_free = np.setdiff1d(
    idx_all, np.concatenate((idx_fix, idx_supp, idx_crown, idx_ring))
)
tags = {
    "fix": idx_fix,
    "supp": idx_supp,
    "free": idx_free,
    "crown": idx_crown,
    "ring": idx_ring,
}


with open("./data/tags", "wb") as fle:
    pickle.dump(tags, fle)

# ---------------------------------------------------------


# Write base surface mesh
gmsh.write("./results/geometry.vtk")
gmsh.finalize()
