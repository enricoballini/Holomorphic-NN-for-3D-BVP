import numpy as np
import pyvista as pv

# Load data
point_coordinates = np.loadtxt("./results/centroids_coordinates")
normals = np.loadtxt("./results/face_normals")
idx_inner = np.loadtxt("./results/idx_inner").astype(int)

n_pt = point_coordinates.shape[0]

mask_inner_points = np.zeros(n_pt)
mask_inner_points[idx_inner] = 1

# ── Build PyVista point cloud ─────────────────────────────────────────────────

cloud = pv.PolyData(point_coordinates)
cloud["inner_boundary_mask"] = mask_inner_points.astype(int)
cloud["normals"] = normals

# ── Save to VTK ───────────────────────────────────────────────────────────────

output_path = "./results/centroids_visualization.vtk"
cloud.save(output_path)

print(f"Saved: {output_path}")
