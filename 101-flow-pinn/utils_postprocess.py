from pdb import set_trace as st

from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import gmsh
import jax
import jax.numpy as jnp

import utils_lapl
import main

# # -----------------------------------------------------------------------
# # Old code kept for legacy
# # -----------------------------------------------------------------------


# def make_postprocess_domain(n_pt=20):
#     """ """
#     node_coordinates = np.loadtxt("./results/centroids_coordinates")
#     return node_coordinates


# def make_postprocess_domain_boundary_and_inner_old():
#     """ """

#     gmsh.initialize()
#     gmsh.open("geometry.step")

#     lc = 0.05  # edge size
#     gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

#     # 2D mesh: ----------------------------------------------
#     gmsh.model.mesh.generate(2)

#     node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
#     node_coords = node_coords.reshape((-1, 3))

#     elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.get_elements(dim=2)
#     triangles = elem_node_tags[0].reshape((-1, 3))

#     areas = []
#     centroids = []
#     normals = []

#     for tri in triangles:
#         p1, p2, p3 = (
#             node_coords[tri[0] - 1],
#             node_coords[tri[1] - 1],
#             node_coords[tri[2] - 1],
#         )

#         v1 = p2 - p1
#         v2 = p3 - p1

#         normal = np.cross(v1, v2)
#         area = 0.5 * np.linalg.norm(normal)
#         unit_normal = normal / np.linalg.norm(normal)

#         assert area > 0

#         centroid = (p1 + p2 + p3) / 3.0

#         areas.append(area)
#         centroids.append(centroid)
#         normals.append(unit_normal)

#     areas = np.array(areas)
#     point_coordinates_boundary = np.array(centroids)
#     normals = np.array(normals)

#     # 3D mesh ---------------------------------------------------
#     gmsh.model.mesh.generate(3)

#     node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
#     node_coords = node_coords.reshape((-1, 3))

#     elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.get_elements(dim=3)
#     tets = elem_node_tags[0].reshape((-1, 4))

#     surface_node_tags = set(triangles.flatten())
#     all_node_tags = set(node_tags)
#     inner_node_tags = np.array(list(all_node_tags - surface_node_tags))

#     point_coordinates_inner = node_coords[inner_node_tags - 1]

#     gmsh.finalize()

#     return point_coordinates_boundary, areas, normals, point_coordinates_inner


# -----------------------------------------------------------------------
# Mesh class
# -----------------------------------------------------------------------

from pdb import set_trace as st
from dataclasses import dataclass, field
import numpy as np
import gmsh
import meshio

import pyvista as pv
import vtk


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
    inner_node_indices: np.ndarray | None = None

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


def make_postprocess_domain_plot_boundary_and_inner():
    """Load mesh from C3D10 connectivity file"""

    point_coordinates = np.load("./results/fem_nodes_coordinates.npy")
    connectivity_file = "./Connectivity.txt"

    c3d10_elements = []
    with open(connectivity_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("*") or not line:
                continue

            parts = [x.strip() for x in line.split(",")]

            # C3D10 elements should have exactly 11 values: 1 element ID + 10 nodes
            # Skip lines that don't match this format (node sets, element sets, etc.)
            if len(parts) != 11:
                continue

            try:
                node_ids = [
                    int(parts[i]) for i in range(1, 11)
                ]  # Skip element ID, get 10 node IDs
                c3d10_elements.append(node_ids)
            except ValueError:
                continue

    c3d10_elements = np.array(c3d10_elements) - 1  # Convert to 0-based

    # Extract corner nodes for tet topology
    tets = c3d10_elements[:, :4]
    node_coords_3d = point_coordinates

    # C3D10 node ordering:
    # Corners: 0,1,2,3
    # Mid-edges: 4(0-1), 5(1-2), 6(0-2), 7(0-3), 8(1-3), 9(2-3)

    def tet_faces_with_midnodes(elem):
        """
        Returns 4 faces, each with corner nodes + mid-edge nodes
        Each face: (corner1, corner2, corner3, mid01, mid12, mid02)
        """
        # Corner nodes
        n0, n1, n2, n3 = elem[:4]
        # Mid-edge nodes
        m01, m12, m02, m03, m13, m23 = elem[4:10]

        return [
            # Face 0-1-2
            {
                "corners": tuple(sorted([n0, n1, n2])),
                "nodes": [n0, n1, n2, m01, m12, m02],
            },
            # Face 0-1-3
            {
                "corners": tuple(sorted([n0, n1, n3])),
                "nodes": [n0, n1, n3, m01, m13, m03],
            },
            # Face 0-2-3
            {
                "corners": tuple(sorted([n0, n2, n3])),
                "nodes": [n0, n2, n3, m02, m23, m03],
            },
            # Face 1-2-3
            {
                "corners": tuple(sorted([n1, n2, n3])),
                "nodes": [n1, n2, n3, m12, m23, m13],
            },
        ]

    # Count face occurrences to find boundary
    face_count = Counter()
    face_to_nodes = {}  # Map corner tuple to 6-node list

    for elem in c3d10_elements:
        for face_info in tet_faces_with_midnodes(elem):
            corners = face_info["corners"]
            face_count[corners] += 1
            face_to_nodes[corners] = face_info["nodes"]

    # Boundary faces (appear once)
    boundary_faces = [face for face, count in face_count.items() if count == 1]
    inner_faces = [face for face, count in face_count.items() if count > 1]

    # Store both triangle connectivity (3 corners) and full 6-node connectivity
    boundary_triangles = np.array([list(face) for face in boundary_faces])
    boundary_triangle_6node = np.array([face_to_nodes[face] for face in boundary_faces])

    inner_triangles = np.array([list(face) for face in inner_faces])
    inner_triangle_6node = np.array([face_to_nodes[face] for face in inner_faces])

    # Compute triangle properties for BOUNDARY triangles
    boundary_triangle_centroids = []
    boundary_normals = []
    boundary_triangle_areas = []

    for tri in boundary_triangles:
        p1, p2, p3 = node_coords_3d[tri]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(normal)
        unit_normal = normal / (np.linalg.norm(normal) + 1e-12)
        centroid = (p1 + p2 + p3) / 3.0

        boundary_triangle_areas.append(area)
        boundary_triangle_centroids.append(centroid)
        boundary_normals.append(unit_normal)

    boundary_triangle_areas = np.array(boundary_triangle_areas)
    boundary_triangle_centroids = np.array(boundary_triangle_centroids)
    boundary_normals = np.array(boundary_normals)

    # Compute triangle properties for INNER triangles
    inner_triangle_centroids = []
    inner_normals = []
    inner_triangle_areas = []

    for tri in inner_triangles:
        p1, p2, p3 = node_coords_3d[tri]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(normal)
        unit_normal = normal / (np.linalg.norm(normal) + 1e-12)
        centroid = (p1 + p2 + p3) / 3.0

        inner_triangle_areas.append(area)
        inner_triangle_centroids.append(centroid)
        inner_normals.append(unit_normal)

    inner_triangle_areas = np.array(inner_triangle_areas)
    inner_triangle_centroids = np.array(inner_triangle_centroids)
    inner_normals = np.array(inner_normals)

    # Combine all triangles and properties
    triangles = np.vstack([boundary_triangles, inner_triangles])
    triangle_centroids = np.vstack(
        [boundary_triangle_centroids, inner_triangle_centroids]
    )
    normals = np.vstack([boundary_normals, inner_normals])
    triangle_areas = np.concatenate([boundary_triangle_areas, inner_triangle_areas])

    # # Boundary nodes
    # boundary_node_indices = np.unique(boundary_triangles.flatten())
    # boundary_nodes = node_coords_3d[boundary_node_indices]

    # # Inner nodes — complement of boundary nodes in the full node set
    # all_node_indices = np.arange(len(node_coords_3d))
    # inner_node_indices = np.setdiff1d(all_node_indices, boundary_node_indices)
    # inner_nodes = node_coords_3d[inner_node_indices]

    boundary_node_indices = np.unique(boundary_triangle_6node.flatten())
    boundary_nodes = node_coords_3d[boundary_node_indices]

    all_node_indices = np.arange(len(node_coords_3d))
    inner_node_indices = np.setdiff1d(all_node_indices, boundary_node_indices)
    inner_nodes = node_coords_3d[inner_node_indices]

    # Outward-pointing boundary normals. Use the mesh centroid as an interior reference point.
    mesh_centroid = node_coords_3d.mean(axis=0)

    boundary_outward_normals = boundary_normals.copy()
    for i, (n, c) in enumerate(zip(boundary_normals, boundary_triangle_centroids)):
        outward_ref = c - mesh_centroid
        if np.dot(outward_ref, n) < 0:
            boundary_outward_normals[i] = -n

    def tet_volume(tet_nodes):
        a, b, c, d = tet_nodes
        return np.abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0

    tets_volumes = np.array([tet_volume(node_coords_3d[tet]) for tet in tets])
    tets_centroids = np.array([node_coords_3d[tet].mean(axis=0) for tet in tets])

    mesh = Mesh(
        nodes=node_coords_3d,
        triangles=triangles,
        tets=tets,
        tets_volumes=tets_volumes,
        tets_centroids=tets_centroids,
        normals=normals,
        triangle_centroids=triangle_centroids,
        triangle_areas=triangle_areas,
        boundary_nodes=boundary_nodes,
        boundary_triangles=boundary_triangles,
        boundary_tets=None,  # TODO
        boundary_normals=boundary_normals,
        boundary_triangle_centroids=boundary_triangle_centroids,
        boundary_triangle_areas=boundary_triangle_areas,
        boundary_node_indices=boundary_node_indices,
        inner_node_indices=inner_node_indices,
        boundary_outward_normals=boundary_outward_normals,
        inner_nodes=inner_nodes,
        inner_triangles=inner_triangles,
        inner_tets=None,  # TODO
        inner_normals=inner_normals,
        inner_triangle_centroids=inner_triangle_centroids,
        inner_triangle_areas=inner_triangle_areas,
        node_data={},
        face_data={},
        cell_data={},
    )

    # Store the 6-node connectivity for quadratic interpolation
    mesh.face_data["boundary_triangle_6node"] = boundary_triangle_6node
    mesh.face_data["inner_triangle_6node"] = inner_triangle_6node

    print(f"\nMesh info:")
    mesh.__repr__()

    return mesh


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


def _batched_lapl(
    params_list,
    nn_list,
    tm_grid,
    tm_full_list,
    points: np.ndarray,
    reductions,
    nodes_int,
    weights_int,
    batch_size: int = 512,
) -> np.ndarray:
    """ """
    results = []
    n = len(points)
    for start in range(0, n, batch_size):
        chunk = points[start : start + batch_size]
        lapl_chunk = utils_lapl.potential_guess(
            params_list,
            nn_list,
            tm_grid,
            tm_full_list,
            nodes_int,
            weights_int,
            reductions,
            chunk,
        )
        results.append(np.array(lapl_chunk))
        print(f"  lapl batch {start}–{min(start + batch_size, n) - 1} / {n - 1} done")
    return np.concatenate(results, axis=0)


def _batch_vel(
    params_list,
    nn_list,
    points: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """ """

    def scalar_potential(point):
        return jnp.real(
            utils_lapl.potential_guess(
                params_list,
                nn_list,
                point[None],
            )[0, 0]
        )

    grad_fn = jax.grad(scalar_potential)
    batched_grad = jax.vmap(grad_fn)

    results = []
    n = len(points)
    for start in range(0, n, batch_size):
        chunk = jnp.array(points[start : start + batch_size])
        grad_chunk = batched_grad(chunk)
        results.append(np.array(grad_chunk))
        print(f"  vel batch {start}–{min(start + batch_size, n) - 1} / {n - 1} done")

    return np.concatenate(results, axis=0)


def directional_derivative(f, point, direction):
    """ """
    _, val = jax.jvp(f, primals=(point,), tangents=(direction,))
    return val


def divergence(f, point):
    """ """
    e_x = jnp.array([1.0, 0.0, 0.0])
    e_y = jnp.array([0.0, 1.0, 0.0])
    e_z = jnp.array([0.0, 0.0, 1.0])

    def component(f, i):
        return lambda x: f(x)[:, i]

    val = (
        directional_derivative(component(f, 0), point, e_x)
        + directional_derivative(component(f, 1), point, e_y)
        + directional_derivative(component(f, 2), point, e_z)
    )
    return val
