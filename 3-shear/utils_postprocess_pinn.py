from pdb import set_trace as st
from dataclasses import dataclass, field
import numpy as np
import gmsh
import meshio
import pickle
from scipy.spatial import Delaunay, ConvexHull
from collections import Counter

import pyvista as pv
import vtk

import utils_mechanics_pinn


@dataclass
class Mesh:
    nodes: np.ndarray | None = None
    triangles: np.ndarray | None = None
    tets: np.ndarray | None = None
    tets_volumes: np.ndarray | None = None
    tets_centroids: np.ndarray | None = None
    normals: np.ndarray | None = None
    triangle_centroids: np.ndarray | None = None
    triangle_areas: np.ndarray | None = None

    # boundary only:
    boundary_nodes: np.ndarray | None = None
    boundary_triangles: np.ndarray | None = None
    boundary_tets: np.ndarray | None = None
    boundary_normals: np.ndarray | None = None
    boundary_triangle_centroids: np.ndarray | None = None
    boundary_triangle_areas: np.ndarray | None = None

    boundary_node_indices: np.ndarray | None = None

    boundary_outward_normals: np.ndarray | None = None

    # interior only:
    inner_nodes: np.ndarray | None = None
    inner_triangles: np.ndarray | None = None
    inner_tets: np.ndarray | None = None
    inner_normals: np.ndarray | None = None
    inner_triangle_centroids: np.ndarray | None = None
    inner_triangle_areas: np.ndarray | None = None

    # Property storage
    node_data: dict = field(default_factory=dict)
    face_data: dict = field(default_factory=dict)
    cell_data: dict = field(default_factory=dict)

    def __repr__(self):
        print("nodes.shape = ", self.nodes.shape)
        print("triangles.shape = ", self.triangles.shape)
        print("tets.shape = ", self.tets.shape)
        print("boundary_triangles.shape = ", self.boundary_triangles.shape)
        return ""


def make_training_domain():
    """ """
    centroids = np.loadtxt("./results/centroids_coordinates")
    normals = np.loadtxt("./results/face_normals")

    meshio_mesh = meshio.read("./results/geometry.vtk")

    nodes = meshio_mesh.points
    tets = None
    boundary_triangles = None
    for cell_block in meshio_mesh.cells:
        if cell_block.type == "tetra":
            tets = cell_block.data
        elif cell_block.type == "triangle":
            boundary_triangles = cell_block.data

    mesh = Mesh(
        nodes=None,
        triangles=None,
        tets=None,
        tets_volumes=None,
        normals=None,
        triangle_centroids=None,
        triangle_areas=None,
        boundary_nodes=nodes,
        boundary_triangles=boundary_triangles,
        boundary_tets=tets,
        boundary_normals=normals,
        boundary_triangle_centroids=centroids,
        boundary_triangle_areas=None,
        boundary_node_indices=None,
        boundary_outward_normals=normals,
        inner_nodes=None,
        inner_triangles=None,
        inner_tets=None,
        inner_normals=None,
        inner_triangle_centroids=None,
        inner_triangle_areas=None,
        node_data={},
        face_data={},
        cell_data={},
    )

    return mesh


def make_postprocess_domain_plot_boundary_and_inner():
    step_file = "geometry.step"
    lc = 0.05

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    print("gmsh verbosity set to 0")
    gmsh.open(step_file)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    # ---------- Boundary mesh (2D) ----------
    gmsh.model.mesh.generate(2)
    node_tags_2d, node_coords_2d, _ = gmsh.model.mesh.get_nodes()
    node_coords_2d = node_coords_2d.reshape((-1, 3))

    # Create mapping from node tags to indices (0-based)
    tag_to_idx_2d = {tag: idx for idx, tag in enumerate(node_tags_2d)}

    _, _, elem_node_tags_2d = gmsh.model.mesh.get_elements(dim=2)
    boundary_triangle_tags = elem_node_tags_2d[0].reshape((-1, 3))

    # Convert from tags to 0-based indices
    boundary_triangles = np.array(
        [[tag_to_idx_2d[tag] for tag in tri] for tri in boundary_triangle_tags]
    )

    triangle_centroids = []
    triangle_normals = []
    triangle_areas = []

    for tri in boundary_triangles:
        p1, p2, p3 = node_coords_2d[tri]  # Now 0-based indexing

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(normal)
        unit_normal = normal / np.linalg.norm(normal)
        centroid = (p1 + p2 + p3) / 3.0

        triangle_areas.append(area)
        triangle_centroids.append(centroid)
        triangle_normals.append(unit_normal)

    triangle_areas = np.array(triangle_areas)
    triangle_centroids = np.array(triangle_centroids)
    triangle_normals = np.array(triangle_normals)

    # ---------- Volume mesh (3D) ----------
    gmsh.model.mesh.generate(3)
    node_tags_3d, node_coords_3d, _ = gmsh.model.mesh.get_nodes()
    node_coords_3d = node_coords_3d.reshape((-1, 3))

    # Create mapping from node tags to indices (0-based)
    tag_to_idx_3d = {tag: idx for idx, tag in enumerate(node_tags_3d)}

    _, _, elem_node_tags_3d = gmsh.model.mesh.get_elements(dim=3)
    tets_tags = elem_node_tags_3d[0].reshape((-1, 4))

    # Convert from tags to 0-based indices
    tets = np.array([[tag_to_idx_3d[tag] for tag in tet] for tet in tets_tags])

    # ---------- Compute inner faces (tet faces not on boundary) ----------
    def tet_faces(tet):
        # Returns the 4 faces of a tetrahedron
        a, b, c, d = tet
        return [
            tuple(sorted([a, b, c])),
            tuple(sorted([a, b, d])),
            tuple(sorted([a, c, d])),
            tuple(sorted([b, c, d])),
        ]

    def tet_volume(tet_nodes):
        a, b, c, d = tet_nodes
        return np.abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0

    # Set of boundary faces (now 0-based)
    boundary_faces_set = set(tuple(sorted(face)) for face in boundary_triangles)

    inner_faces = []
    for tet in tets:
        for face in tet_faces(tet):
            if face not in boundary_faces_set:
                inner_faces.append(face)

    tets_volumes = np.array([tet_volume(node_coords_3d[tet]) for tet in tets])
    tets_centroids = np.array([node_coords_3d[tet].mean(axis=0) for tet in tets])

    # Convert inner faces to centroids (now 0-based indexing works correctly)
    triangle_centroids_inner = np.array(
        [node_coords_3d[np.array(face)].mean(axis=0) for face in inner_faces]
    )

    gmsh.finalize()

    mesh = Mesh(
        nodes=node_coords_3d,
        triangles=None,
        tets=tets,
        tets_volumes=tets_volumes,
        tets_centroids=tets_centroids,
        normals=None,
        triangle_centroids=triangle_centroids_inner,
        triangle_areas=None,
        boundary_nodes=node_coords_2d,
        boundary_triangles=boundary_triangles,
        boundary_tets=tets,
        boundary_normals=None,
        boundary_triangle_centroids=triangle_centroids,
        boundary_triangle_areas=triangle_areas,
        boundary_node_indices=None,
        boundary_outward_normals=triangle_normals,
        inner_nodes=None,
        inner_triangles=None,
        inner_tets=None,
        inner_normals=None,
        inner_triangle_centroids=None,
        inner_triangle_areas=None,
        node_data={},
        face_data={},
        cell_data={},
    )

    return mesh


def interpolate_node_values_to_tet_centroids(mesh, node_values):
    """
    Linear interpolation of node values to tetrahedron centroids.
    """
    assert mesh.tets is not None, "mesh.tets is not set"
    assert node_values.shape[0] == len(
        mesh.nodes
    ), f"Expected values for all {len(mesh.nodes)} nodes, got {node_values.shape[0]}"

    # mean over the 4 corners
    return node_values[mesh.tets].mean(axis=1)


def interpolate_tet_centroid_values_to_nodes(mesh, tet_values):
    """
    Volume-weighted interpolation from tetrahedron centroids to nodes.

    This is the adjoint of interpolate_node_values_to_tet_centroids: each node
    receives the volume-weighted average of all tets that share it.

    Boundary nodes are treated identically to interior nodes — they are corners
    of tetrahedra and receive contributions from every tet they belong to.
    Nodes that belong to no tet (e.g. boundary_nodes with a separate index
    space) are left as zero and flagged via the returned mask.
    """
    assert mesh.tets is not None, "mesh.tets is not set"
    assert mesh.tets_volumes is not None, "mesh.tets_volumes is not set"
    assert tet_values.shape[0] == len(
        mesh.tets
    ), f"Expected values for all {len(mesh.tets)} tets, got {tet_values.shape[0]}"

    num_nodes = len(mesh.nodes)
    value_shape = tet_values.shape[1:]  # () for scalar, (3,) for vector, etc.

    node_values = np.zeros((num_nodes,) + value_shape)
    node_weights = np.zeros(num_nodes)  # accumulated volume per node

    # mesh.tets has shape (num_tets, 4); each row lists the 4 corner node indices
    # mesh.tets_volumes has shape (num_tets,)
    vols = mesh.tets_volumes  # (num_tets,)

    # Broadcast volumes for weighted value accumulation: (num_tets, 1, ...)
    if value_shape:
        weighted_values = vols.reshape(-1, 1) * tet_values  # keep leading dims
    else:
        weighted_values = vols * tet_values  # scalar case

    # Scatter-add over the 4 corners of every tet
    for corner in range(4):
        node_ids = mesh.tets[:, corner]  # (num_tets,) node index per tet
        np.add.at(node_values, node_ids, weighted_values)
        np.add.at(node_weights, node_ids, vols)

    # Normalise — nodes with no tet contribution keep value 0
    valid_mask = node_weights > 0.0
    node_values[valid_mask] /= node_weights[valid_mask].reshape(
        (-1,) + (1,) * len(value_shape)  # broadcast over value dims
    )

    if not np.all(valid_mask):
        n_invalid = (~valid_mask).sum()
        print(
            f"  Warning: {n_invalid} node(s) belong to no tet "
            f"(likely boundary_nodes with a disjoint index space). "
            f"Their values are left as zero."
        )

    return node_values


def interpolate_node_values_to_face_centers(mesh, node_values, face_type="boundary"):
    """
    Quadratic interpolation from 6-node triangular faces to face centers.
    """

    assert node_values.shape[0] == len(
        mesh.nodes
    ), f"Expected values for all {len(mesh.nodes)} nodes, got {node_values.shape[0]}"

    # Select which triangles to use
    if face_type == "boundary":
        triangle_6node = mesh.face_data["boundary_triangle_6node"]
    elif face_type == "inner":
        triangle_6node = mesh.face_data["inner_triangle_6node"]
    elif face_type == "all":
        triangle_6node = np.vstack(
            [
                mesh.face_data["boundary_triangle_6node"],
                mesh.face_data["inner_triangle_6node"],
            ]
        )
    else:
        raise ValueError(
            f"face_type must be 'boundary', 'inner', or 'all', got '{face_type}'"
        )

    num_faces = len(triangle_6node)

    # Get shape of data (could be scalar, vector, or matrix)
    value_shape = node_values.shape[1:]
    face_values = np.zeros((num_faces,) + value_shape)

    # Quadratic interpolation weights for center of triangle
    w_corner = -1.0 / 9.0
    w_midside = 4.0 / 9.0

    for face_idx, nodes_6 in enumerate(triangle_6node):
        n0, n1, n2, m01, m12, m02 = nodes_6

        face_values[face_idx] = w_corner * (
            node_values[n0] + node_values[n1] + node_values[n2]
        ) + w_midside * (node_values[m01] + node_values[m12] + node_values[m02])

    return face_values


def save_mesh(mesh, filename):
    # Fall back to boundary_nodes if nodes is None
    nodes = mesh.nodes if mesh.nodes is not None else mesh.boundary_nodes
    nodes = np.asarray(nodes)

    # Keys in face_data that are internal connectivity arrays, not cell data
    SKIP_FACE_DATA_KEYS = {"boundary_triangle_6node", "inner_triangle_6node"}

    # Create PyVista mesh
    if (
        mesh.boundary_triangles is not None
        and len(mesh.boundary_triangles) > 0
        and mesh.tets is not None
        and len(mesh.tets) > 0
    ):
        # Mixed mesh with both triangles and tetrahedra
        cells = []
        cell_types = []

        # Add triangles
        for tri in mesh.boundary_triangles:
            cells.extend([3, *tri])
            cell_types.append(vtk.VTK_TRIANGLE)

        # Add tetrahedra
        for tet in mesh.tets:
            cells.extend([4, *tet])
            cell_types.append(vtk.VTK_TETRA)

        # Convert to numpy arrays
        cells = np.array(cells, dtype=np.int64)
        cell_types = np.array(cell_types, dtype=np.uint8)

        pv_mesh = pv.UnstructuredGrid(cells, cell_types, nodes)

    elif mesh.boundary_triangles is not None and len(mesh.boundary_triangles) > 0:
        # Only triangles (surface mesh)
        faces = np.hstack([[3] + list(tri) for tri in mesh.boundary_triangles])
        pv_mesh = pv.PolyData(nodes, faces)
    elif mesh.tets is not None and len(mesh.tets) > 0:
        # Only tetrahedra
        cells = []
        cell_types = []
        for tet in mesh.tets:
            cells.extend([4, *tet])
            cell_types.append(vtk.VTK_TETRA)

        # Convert to numpy arrays
        cells = np.array(cells, dtype=np.int64)
        cell_types = np.array(cell_types, dtype=np.uint8)

        pv_mesh = pv.UnstructuredGrid(cells, cell_types, nodes)
    else:
        raise ValueError("Mesh must have either triangles or tetrahedra")

    # Add point data
    for key, val in mesh.node_data.items():
        pv_mesh.point_data[key] = np.asarray(val)

    # Add cell data
    cell_data_combined = {}

    # Handle face_data (for triangles)
    if mesh.face_data:
        for key, val in mesh.face_data.items():
            if key in SKIP_FACE_DATA_KEYS:
                continue

            if isinstance(val, vtk.vtkFloatArray):
                # Convert VTK array to numpy
                n_tuples = val.GetNumberOfTuples()
                n_components = val.GetNumberOfComponents()
                component_names = [val.GetComponentName(i) for i in range(n_components)]

                # Extract data from VTK array
                np_val = np.zeros((n_tuples, n_components))
                for i in range(n_tuples):
                    val.GetTuple(i, np_val[i])

                cell_data_combined[key] = (np_val, component_names)
            else:
                val = np.asarray(val)
                if val.ndim == 3:
                    val = val.reshape(val.shape[0], -1)
                cell_data_combined[key] = (val, None)

    # Handle cell_data (for tetrahedra)
    if mesh.cell_data:
        for key, val in mesh.cell_data.items():
            val = np.asarray(val)
            if val.ndim == 3:
                val = val.reshape(val.shape[0], -1)

            if key in cell_data_combined:
                # Concatenate with face data
                face_val, comp_names = cell_data_combined[key]
                cell_data_combined[key] = (
                    np.concatenate([face_val, val], axis=0),
                    comp_names,
                )
            else:
                # Pad with zeros for triangle cells if needed
                if (
                    mesh.boundary_triangles is not None
                    and len(mesh.boundary_triangles) > 0
                ):
                    if val.ndim == 1:
                        padding = np.zeros(len(mesh.boundary_triangles))
                    else:
                        padding = np.zeros((len(mesh.boundary_triangles), val.shape[1]))
                    cell_data_combined[key] = (
                        np.concatenate([padding, val], axis=0),
                        None,
                    )
                else:
                    cell_data_combined[key] = (val, None)

    # Pad face_data arrays that don't have corresponding cell_data
    if mesh.boundary_triangles is not None and mesh.tets is not None:
        for key, (val, comp_names) in list(cell_data_combined.items()):
            if len(val) == len(mesh.boundary_triangles):
                # Need to pad for tetrahedra
                if val.ndim == 1:
                    padding = np.zeros(len(mesh.tets))
                else:
                    padding = np.zeros((len(mesh.tets), val.shape[1]))
                cell_data_combined[key] = (
                    np.concatenate([val, padding], axis=0),
                    comp_names,
                )

    # Add combined cell data to PyVista mesh
    for key, (val, comp_names) in cell_data_combined.items():
        if comp_names is not None:
            # Create new VTK array with component names
            vtk_arr = vtk.vtkFloatArray()
            vtk_arr.SetNumberOfComponents(val.shape[1] if val.ndim > 1 else 1)
            vtk_arr.SetNumberOfTuples(val.shape[0])
            vtk_arr.SetName(key)

            # Set component names
            for i, name in enumerate(comp_names):
                vtk_arr.SetComponentName(i, name)

            # Populate array
            for i in range(val.shape[0]):
                vtk_arr.SetTuple(i, val[i])

            # Add to mesh
            pv_mesh.cell_data.set_array(vtk_arr, key)
        else:
            pv_mesh.cell_data[key] = val

    # Save using PyVista
    pv_mesh.save(filename)


# ---------------------------------------------------------------------------
# Batched wrapper
# ---------------------------------------------------------------------------


def _batched_sigma(
    jacobian_field,
    tm_list,
    tm_full_list,
    params_list,
    nn_list,
    points: np.ndarray,
    reductions,
    nu,
    G,
    nodes_int,
    weights_int,
    batch_size: int = 512,
) -> np.ndarray:
    """ """
    results = []
    n = len(points)
    for start in range(0, n, batch_size):
        chunk = points[start : start + batch_size]
        sigma_chunk = utils_mechanics_pinn.compute_sigma_tensor_from_displ(
            jacobian_field,
            tm_list,
            tm_full_list,
            params_list,
            nn_list,
            chunk,
            reductions,
            nu,
            G,
            nodes_int,
            weights_int,
        )
        results.append(np.array(sigma_chunk))
        print(f"  stress batch {start}–{min(start + batch_size, n) - 1} / {n - 1} done")
    return np.concatenate(results, axis=0)


def _batched_sigma_from_displ(
    jacobian_field,
    tm_list,
    tm_full_list,
    params_list,
    nn_list,
    points: np.ndarray,
    reductions,
    nu,
    G,
    nodes_int,
    weights_int,
    batch_size: int = 512,
) -> np.ndarray:
    """ """
    results = []
    n = len(points)
    for start in range(0, n, batch_size):
        chunk = points[start : start + batch_size]
        sigma_chunk = utils_mechanics_pinn.compute_sigma_tensor_from_displ(
            jacobian_field,
            tm_list,
            tm_full_list,
            params_list,
            nn_list,
            chunk,
            reductions,
            nu,
            G,
            nodes_int,
            weights_int,
        )
        results.append(np.array(sigma_chunk))
        print(f"  stress batch {start}–{min(start + batch_size, n) - 1} / {n - 1} done")
    return np.concatenate(results, axis=0)


def _batched_displacement(
    params_list,
    nn_list,
    tm_list,
    tm_full_list,
    points: np.ndarray,
    reductions,
    nu,
    G,
    nodes_int,
    weights_int,
    batch_size: int = 512,
) -> np.ndarray:
    """ """
    results = []
    n = len(points)
    for start in range(0, n, batch_size):
        chunk = points[start : start + batch_size]
        displ_chunk = utils_mechanics_pinn.compute_displacement(
            params_list,
            nn_list,
            tm_list,
            tm_full_list,
            chunk,
            reductions,
            nu,
            G,
            nodes_int,
            weights_int,
        )
        results.append(np.array(displ_chunk))
        print(
            f"  displacement batch {start}–{min(start + batch_size, n) - 1} / {n - 1} done"
        )
    return np.concatenate(results, axis=0)
