"""
plot_and_make_vtu_exact.py
--------------------------
Exports the exact (FEM/analytical) stress, traction and displacement fields.

Output files
------------
./results/stresses_tractions_exact.vtk
./results/displacement_exact.vtk
./results/displacement_mesh_deformed_exact.vtk
"""

from pdb import set_trace as st
import numpy as np
import vtk


import utils_exact_solution
import utils_mechanics
import utils_postprocess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# add_* functions
# ---------------------------------------------------------------------------


def add_stresses(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """ """
    stress_exact, displ_exact = utils_exact_solution.compute_exact_solution(mesh.nodes)

    stress_exact *= 1

    mesh.node_data["stress_exact"] = _make_stress_vtk_array(
        stress_exact, "stress_exact"
    )
    return mesh


def add_tractions(mesh: utils_postprocess.Mesh) -> utils_postprocess.Mesh:
    """ """
    stress_exact_faces, displ_exact = utils_exact_solution.compute_exact_solution(
        mesh.boundary_triangle_centroids
    )
    # # Quadratic interpolation from nodes to boundary face centres
    # stress_exact_faces = utils_postprocess.interpolate_node_values_to_face_centers(
    #     mesh, stress_exact_nodes, face_type="boundary"
    # )
    normals = (
        mesh.boundary_outward_normals
    )  # (F, 3) unit outward normals at face centres
    tractions_exact = utils_mechanics.compute_sigma_n(stress_exact_faces, normals)

    tractions_exact *= 1

    mesh.face_data["tractions_exact"] = tractions_exact
    return mesh


def add_displacement(mesh: utils_postprocess.Mesh):
    """ """
    stress_exact, displ_exact = utils_exact_solution.compute_exact_solution(mesh.nodes)

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

    displ_exact *= 1

    mesh.node_data["displacement_exact"] = displ_exact
    mesh_deformed_exact = deform_mesh(displ_exact)
    return mesh, mesh_deformed_exact


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Stresses and tractions
    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh = add_stresses(mesh)
    mesh = add_tractions(mesh)
    utils_postprocess.save_mesh(mesh, "./results/stresses_tractions_exact.vtk")

    # Displacement
    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh, mesh_deformed_exact = add_displacement(mesh)
    utils_postprocess.save_mesh(mesh, "./results/displacement_exact.vtk")
    utils_postprocess.save_mesh(
        mesh_deformed_exact, "./results/displacement_mesh_deformed_exact.vtk"
    )

    print("\nDone!")
