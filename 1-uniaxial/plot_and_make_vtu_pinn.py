"""
plot_and_make_vtu_pinn.py
-------------------------
Exports the PINN stress, traction and displacement fields.

Output files
------------
./results/stresses_tractions_pinn.vtk
./results/displacement_pinn.vtk
./results/displacement_mesh_deformed_pinn.vtk
"""

import numpy as np
import vtk

import case_settings_pinn
import utils_data_and_folders_pinn
import utils_mechanics
import utils_mechanics_pinn
import utils_postprocess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_pinn_params():
    """Load PINN material data and trained model parameters."""
    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")
    idx_seed = 0
    params_list = utils_data_and_folders_pinn.load_params_list(idx_seed)
    nn_list = case_settings_pinn.define_nn_forwards_and_derivatives()
    return nu, G, params_list, nn_list


def _make_stress_vtk_array(stress: np.ndarray, name: str) -> vtk.vtkFloatArray:
    """Convert an (N, 3, 3) stress tensor array into a 7-component VTK float array
    (6 Voigt components + element-wise L1 norm)."""
    component_names = ["XX", "YY", "ZZ", "XY", "XZ", "YZ", "element-wise l1 norm"]
    voigt = np.column_stack(
        [
            stress[:, 0, 0],
            stress[:, 1, 1],
            stress[:, 2, 2],
            stress[:, 0, 1],
            stress[:, 0, 2],
            stress[:, 1, 2],
            np.sum(np.abs(stress), axis=(1, 2)),
        ]
    )
    vtk_arr = vtk.vtkFloatArray()
    vtk_arr.SetNumberOfComponents(7)
    vtk_arr.SetNumberOfTuples(voigt.shape[0])
    vtk_arr.SetName(name)
    for idx, cname in enumerate(component_names):
        vtk_arr.SetComponentName(idx, cname)
    for i in range(voigt.shape[0]):
        vtk_arr.SetTuple(i, voigt[i])
    return vtk_arr


# ---------------------------------------------------------------------------
# Batched wrapper – avoids GPU OOM on large meshes
# ---------------------------------------------------------------------------


def _batched_sigma_pinn(
    params_list,
    nn_list,
    points: np.ndarray,
    nu,
    G,
    batch_size: int = 512,
) -> np.ndarray:
    """Call utils_mechanics_pinn.compute_sigma_tensor_from_displ in chunks of
    `batch_size` points to avoid GPU OOM on large meshes."""
    results = []
    n = len(points)
    for start in range(0, n, batch_size):
        chunk = points[start : start + batch_size]
        sigma_chunk = utils_mechanics_pinn.compute_sigma_tensor_from_displ(
            params_list, nn_list, chunk, nu, G
        )
        results.append(np.array(sigma_chunk))
        print(f"  stress batch {start}–{min(start + batch_size, n) - 1} / {n - 1} done")
    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# add_* functions
# ---------------------------------------------------------------------------


def add_stresses(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """Compute PINN stress at every node and store in mesh.node_data."""
    nu, G, params_list, nn_list = _load_pinn_params()

    stress_pinn = _batched_sigma_pinn(params_list, nn_list, mesh.nodes, nu, G)

    stress_pinn *= 0.1

    mesh.node_data["stress_pinn"] = _make_stress_vtk_array(stress_pinn, "stress_pinn")
    return mesh


def add_tractions(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """Compute PINN tractions at boundary face centres and store in mesh.face_data."""
    nu, G, params_list, nn_list = _load_pinn_params()

    stress_pinn_faces = _batched_sigma_pinn(
        params_list, nn_list, mesh.boundary_triangle_centroids, nu, G
    )

    normals = mesh.boundary_outward_normals  # (F, 3) unit outward normals at face centres
    tractions_pinn = utils_mechanics.compute_sigma_n(stress_pinn_faces, normals)

    tractions_pinn *= 0.1

    mesh.face_data["tractions_pinn"] = tractions_pinn
    return mesh


def add_displacement(mesh: utils_postprocess.Mesh):
    """Compute PINN displacement at nodes, store it, and return the deformed mesh.

    Returns
    -------
    mesh               : original mesh with displacement_pinn added to node_data
    mesh_deformed_pinn : mesh with nodes moved by the PINN displacement
    """
    nu, G, params_list, nn_list = _load_pinn_params()

    displ_pinn = utils_mechanics_pinn.compute_displacement(params_list, nn_list, mesh.nodes)

    def deform_mesh(nodes_displacement: np.ndarray) -> utils_postprocess.Mesh:
        nodes_def = mesh.nodes + nodes_displacement

        triangles_def = mesh.boundary_triangles.copy()
        tets_def = mesh.tets.copy()

        boundary_triangle_centroids_def = []
        boundary_normals_def = []
        boundary_triangle_areas_def = []

        for tri in triangles_def:
            p1, p2, p3 = nodes_def[tri]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(normal)
            unit_normal = normal / np.linalg.norm(normal)
            centroid = (p1 + p2 + p3) / 3.0
            boundary_triangle_centroids_def.append(centroid)
            boundary_normals_def.append(unit_normal)
            boundary_triangle_areas_def.append(area)

        boundary_triangle_centroids_def = np.array(boundary_triangle_centroids_def)
        boundary_normals_def = np.array(boundary_normals_def)
        boundary_triangle_areas_def = np.array(boundary_triangle_areas_def)

        # Recompute inner face centroids from deformed nodes (if present)
        inner_triangle_centroids_def = None
        if mesh.inner_triangle_centroids is not None:

            def tet_faces(tet):
                a, b, c, d = tet
                return [
                    tuple(sorted([a, b, c])),
                    tuple(sorted([a, b, d])),
                    tuple(sorted([a, c, d])),
                    tuple(sorted([b, c, d])),
                ]

            boundary_faces_set = set(tuple(sorted(f)) for f in triangles_def)
            inner_faces = []
            for tet in tets_def:
                for face in tet_faces(tet):
                    if face not in boundary_faces_set:
                        inner_faces.append(face)

            inner_triangle_centroids_def = np.array(
                [nodes_def[np.array(face)].mean(axis=0) for face in inner_faces]
            )

        return utils_postprocess.Mesh(
            nodes=nodes_def,
            boundary_triangles=triangles_def,
            tets=tets_def,
            boundary_triangle_centroids=boundary_triangle_centroids_def,
            boundary_normals=boundary_normals_def,
            boundary_triangle_areas=boundary_triangle_areas_def,
            inner_triangle_centroids=inner_triangle_centroids_def,
            node_data=mesh.node_data.copy(),
            face_data=mesh.face_data.copy(),
            cell_data=mesh.cell_data.copy(),
        )

    displ_pinn *= 0.1

    mesh.node_data["displacement_pinn"] = displ_pinn
    mesh_deformed_pinn = deform_mesh(displ_pinn)
    return mesh, mesh_deformed_pinn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Stresses and tractions – PINN
    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh = add_stresses(mesh)
    mesh = add_tractions(mesh)
    utils_postprocess.save_mesh(mesh, "./results/stresses_tractions_pinn.vtk")

    # Displacement – PINN + deformed mesh
    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh, mesh_deformed_pinn = add_displacement(mesh)
    utils_postprocess.save_mesh(mesh, "./results/displacement_pinn.vtk")
    utils_postprocess.save_mesh(
        mesh_deformed_pinn, "./results/displacement_mesh_deformed_pinn.vtk"
    )

    print("\nDone!")
