import os
from pdb import set_trace as st
import pickle
import copy

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import vtk
import meshio
from meshio import CellBlock


import utils_data_and_folders

# import utils_data_and_folders_pinn
import case_settings

# import case_settings_pinn
import utils_mechanics

# import utils_mechanics_pinn
import utils_mechanics

import utils_nn

# import utils_nn_pinn
import utils_postprocess


def add_dataset_labels():
    point_coordinates, bc_vals, bc_type, normals, areas, tags = (
        case_settings.generate_boundary_data()
    )

    (
        point_coordinates_train,
        bc_vals_train,
        bc_type_train,
        normals_train,
        areas_train,
        weights_loss_train,
        tags_list_train,
        point_coordinates_test,
        bc_vals_test,
        bc_type_test,
        normals_test,
        areas_test,
        weights_loss_test,
        tags_list_test,
    ) = utils_nn.split_dataset(
        point_coordinates, bc_vals, bc_type, normals, areas, np.zeros_like(areas), tags
    )

    train_mask = (
        np.isclose(
            point_coordinates[:, None, :],
            point_coordinates_train[None, :, :],
        )
        .all(axis=2)
        .any(axis=1)
    )

    test_mask = (
        np.isclose(
            point_coordinates[:, None, :],
            point_coordinates_test[None, :, :],
        )
        .all(axis=2)
        .any(axis=1)
    )

    cloud = pv.PolyData(point_coordinates)

    labels = np.zeros(point_coordinates.shape[0], dtype=int)
    labels[train_mask] = 1
    labels[test_mask] = 2
    cloud.point_data["set_type"] = labels

    return cloud


def add_loss(mesh: utils_postprocess.Mesh):
    """
    TODO: currently only for training mesh
    """
    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")
    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")

    tm_grid, delta_tm = utils_nn.define_tm(M, bias_rotation)
    reductions = utils_nn.define_reductions()
    tm_full_list = utils_nn.make_tm_full_list(tm_grid, nodes_int)

    idx_seed = 0
    params_list = utils_data_and_folders.load_params_list(idx_seed)
    nn_list = case_settings.define_nn_forwards_and_derivatives()

    point_coordinates, bc_vals, bc_type, normals, areas, tags = (
        case_settings.generate_boundary_data()
    )

    weights_loss = case_settings.compute_weights_loss(areas, tags)

    bc_vals = np.zeros((point_coordinates.shape[0], 3))  # TODO

    def loss_single_pt(pt):
        val = utils_nn.loss_l2(
            params_list,
            nn_list,
            tm_grid,
            tm_full_list,
            reductions,
            jnp.atleast_2d(pt),
            bc_vals,
            bc_type,
            normals,
            weights_loss,
            nu,
            G,
            nodes_int,
            weights_int,
        )
        return jnp.real(val)

    loss = np.array(jax.vmap(loss_single_pt)(point_coordinates))

    mesh.face_data["loss"] = loss

    return mesh


def add_stresses(mesh: utils_postprocess.Mesh):
    """ """
    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")
    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")

    tm_list, delta_tm = utils_nn.define_tm(M, bias_rotation)
    reductions = utils_nn.define_reductions()
    tm_full_list = utils_nn.make_tm_full_list(tm_list, nodes_int)

    idx_seed = 0
    params_list = utils_data_and_folders.load_params_list(idx_seed)
    nn_list = case_settings.define_nn_forwards_and_derivatives()

    stress_exact, displ_exact = case_settings.compute_exact_solution(mesh.nodes)

    # stress_nn = utils_mechanics.compute_sigma_tensor(
    #     params_list,
    #     nn_list,
    #     tm_list,
    #     tm_full_list,
    #     delta_tm,
    #     mesh.nodes,
    #     reductions,
    #     nu,
    #     G,
    #     nodes_int,
    #     weights_int,
    # )

    jacobian_field = utils_mechanics.define_strain_from_displ()

    stress_nn = utils_mechanics.compute_sigma_tensor_from_displ(
        jacobian_field,
        tm_list,
        tm_full_list,
        params_list,
        nn_list,
        mesh.nodes,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    stress_exact *= 1
    stress_nn *= 1

    component_names = ["XX", "YY", "ZZ", "XY", "XZ", "YZ", "element-wise l1 norm"]

    stress_exact_voigt = np.column_stack(
        [
            stress_exact[:, 0, 0],
            stress_exact[:, 1, 1],
            stress_exact[:, 2, 2],
            stress_exact[:, 0, 1],
            stress_exact[:, 0, 2],
            stress_exact[:, 1, 2],
            np.sum(np.abs(stress_exact), axis=(1, 2)),
        ]
    )

    stress_exact_vtk = vtk.vtkFloatArray()
    stress_exact_vtk.SetNumberOfComponents(7)
    stress_exact_vtk.SetNumberOfTuples(stress_exact_voigt.shape[0])
    stress_exact_vtk.SetName("stress_exact")

    for idx, name in enumerate(component_names):
        stress_exact_vtk.SetComponentName(idx, name)

    for i in range(stress_exact_voigt.shape[0]):
        stress_exact_vtk.SetTuple(i, stress_exact_voigt[i])

    stress_nn_voigt = np.column_stack(
        [
            stress_nn[:, 0, 0],
            stress_nn[:, 1, 1],
            stress_nn[:, 2, 2],
            stress_nn[:, 0, 1],
            stress_nn[:, 0, 2],
            stress_nn[:, 1, 2],
            np.sum(np.abs(stress_nn), axis=(1, 2)),
        ]
    )

    stress_nn_vtk = vtk.vtkFloatArray()
    stress_nn_vtk.SetNumberOfComponents(7)
    stress_nn_vtk.SetNumberOfTuples(stress_nn_voigt.shape[0])
    stress_nn_vtk.SetName("stress_nn")

    for idx, name in enumerate(component_names):
        stress_nn_vtk.SetComponentName(idx, name)

    for i in range(stress_nn_voigt.shape[0]):
        stress_nn_vtk.SetTuple(i, stress_nn_voigt[i])

    mesh.node_data["stress_exact"] = stress_exact_vtk
    mesh.node_data["stress_nn"] = stress_nn_vtk

    # mesh.node_data["err_abs_distribution_stress"] = np.loadtxt(
    #     "./results/err_abs_distribution_stress"
    # )
    # mesh.node_data["err_rel_distribution_stress"] = np.loadtxt(
    #     "./results/err_rel_distribution_stress"
    # )

    # # PINN: ------------------------------------
    # idx_seed = 0
    # params_list = utils_data_and_folders_pinn.load_params_list(idx_seed)
    # nn_list = case_settings_pinn.define_nn_forwards_and_derivatives()

    # stress_pinn = utils_mechanics_pinn.compute_sigma_tensor_from_displ(
    #     params_list, nn_list, mesh.nodes, nu, G
    # )

    # stress_pinn *= 1

    # stress_pinn_voigt = np.column_stack(
    #     [
    #         stress_pinn[:, 0, 0],
    #         stress_pinn[:, 1, 1],
    #         stress_pinn[:, 2, 2],
    #         stress_pinn[:, 0, 1],
    #         stress_pinn[:, 0, 2],
    #         stress_pinn[:, 1, 2],
    #         np.sum(np.abs(stress_pinn), axis=(1, 2)),
    #     ]
    # )

    # stress_pinn_vtk = vtk.vtkFloatArray()
    # stress_pinn_vtk.SetNumberOfComponents(7)
    # stress_pinn_vtk.SetNumberOfTuples(stress_pinn_voigt.shape[0])
    # stress_pinn_vtk.SetName("stress_pinn")

    # for idx, name in enumerate(component_names):
    #     stress_pinn_vtk.SetComponentName(idx, name)

    # for i in range(stress_pinn_voigt.shape[0]):
    #     stress_pinn_vtk.SetTuple(i, stress_pinn_voigt[i])

    # mesh.node_data["stress_pinn"] = stress_pinn_vtk

    return mesh


def add_tractions(mesh: utils_postprocess.Mesh):
    """"""
    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")
    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")

    tm_list, delta_tm = utils_nn.define_tm(M, bias_rotation)
    reductions = utils_nn.define_reductions()
    tm_full_list = utils_nn.make_tm_full_list(tm_list, nodes_int)

    idx_seed = 0
    params_list = utils_data_and_folders.load_params_list(idx_seed)
    nn_list = case_settings.define_nn_forwards_and_derivatives()

    stress_exact, displ_exact = case_settings.compute_exact_solution(
        mesh.points_coordinates_boundary
    )

    jacobian_field = utils_mechanics.define_strain_from_displ()

    stress_nn = utils_mechanics.compute_sigma_tensor_from_displ(
        jacobian_field,
        tm_list,
        tm_full_list,
        params_list,
        nn_list,
        mesh.points_coordinates_boundary,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    stress_exact *= 1
    stress_nn *= 1

    # Normals
    normals = mesh.triangle_normals

    tractions_exact = utils_mechanics.compute_sigma_n(stress_exact, normals)
    tractions_nn = utils_mechanics.compute_sigma_n(stress_nn, normals)

    mesh.face_data["tractions_exact"] = tractions_exact
    mesh.face_data["tractions_nn"] = tractions_nn

    # # PINN: --------------------
    # idx_seed = 0
    # params_list = utils_data_and_folders_pinn.load_params_list(idx_seed)
    # nn_list = case_settings_pinn.define_nn_forwards_and_derivatives()

    # stress_pinn = utils_mechanics_pinn.compute_sigma_tensor_from_displ(
    #     params_list, nn_list, mesh.points_coordinates_boundary, nu, G
    # )
    # tractions_pinn = utils_mechanics.compute_sigma_n(stress_pinn, normals)
    # mesh.face_data["tractions_pinn"] = tractions_pinn

    return mesh


def add_displacement(mesh: utils_postprocess.Mesh):
    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")
    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")

    tm_list, delta_tm = utils_nn.define_tm(M, bias_rotation)
    reductions = utils_nn.define_reductions()
    tm_full_list = utils_nn.make_tm_full_list(tm_list, nodes_int)

    idx_seed = 0
    params_list = utils_data_and_folders.load_params_list(idx_seed)
    nn_list = case_settings.define_nn_forwards_and_derivatives()

    stress_exact, displ_exact = case_settings.compute_exact_solution(mesh.nodes)

    displ_nn = utils_nn.compute_displacement(
        params_list,
        nn_list,
        tm_list,
        tm_full_list,
        mesh.nodes,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    displ_exact *= 1
    displ_nn *= 1

    mesh.node_data["displacement_exact"] = displ_exact
    mesh.node_data["displacement_nn"] = displ_nn

    # mesh.node_data["err_abs_distribution_displ"] = np.loadtxt(
    #     "./results/err_abs_distribution_displ"
    # )
    # mesh.node_data["err_rel_distribution_displ"] = np.loadtxt(
    #     "./results/err_rel_distribution_displ"
    # )

    def deform_mesh(nodes_displacement):
        nodes_def = mesh.nodes + nodes_displacement

        triangles_def = mesh.boundary_triangles.copy()
        tets_def = mesh.tets.copy()

        triangle_centroids_def = []
        triangle_normals_def = []
        triangle_areas_def = []

        for tri in triangles_def:
            p1, p2, p3 = nodes_def[tri]  # Changed from tri - 1
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(normal)
            unit_normal = normal / np.linalg.norm(normal)
            centroid = (p1 + p2 + p3) / 3.0
            triangle_centroids_def.append(centroid)
            triangle_normals_def.append(unit_normal)
            triangle_areas_def.append(area)

        triangle_centroids_def = np.array(triangle_centroids_def)
        triangle_normals_def = np.array(triangle_normals_def)
        triangle_areas_def = np.array(triangle_areas_def)

        points_coordinates_boundary_def = triangle_centroids_def

        # Recompute inner face centroids from deformed nodes
        points_coordinates_inner_def = None
        if mesh.points_coordinates_inner is not None:

            def tet_faces(tet):
                a, b, c, d = tet
                return [
                    tuple(sorted([a, b, c])),
                    tuple(sorted([a, b, d])),
                    tuple(sorted([a, c, d])),
                    tuple(sorted([b, c, d])),
                ]

            boundary_faces_set = set(tuple(sorted(face)) for face in triangles_def)

            inner_faces = []
            for tet in tets_def:
                for face in tet_faces(tet):
                    if face not in boundary_faces_set:
                        inner_faces.append(face)

            # Compute centroids from deformed nodes (0-based indexing)
            points_coordinates_inner_def = np.array(
                [nodes_def[np.array(face)].mean(axis=0) for face in inner_faces]
            )

        return utils_postprocess.Mesh(
            nodes=nodes_def,
            boundary_triangles=triangles_def,
            tets=tets_def,
            points_coordinates_boundary=points_coordinates_boundary_def,
            points_coordinates_inner=points_coordinates_inner_def,
            triangle_centroids=triangle_centroids_def,
            triangle_normals=triangle_normals_def,
            triangle_areas=triangle_areas_def,
            node_data=mesh.node_data.copy(),
            face_data=mesh.face_data.copy(),
            cell_data=mesh.cell_data.copy(),
        )

    mesh_deformed_nn = deform_mesh(displ_nn)
    mesh_deformed_exact = deform_mesh(displ_exact)

    # # PINN: -----------------------
    # idx_seed = 0
    # params_list = utils_data_and_folders_pinn.load_params_list(idx_seed)
    # nn_list = case_settings_pinn.define_nn_forwards_and_derivatives()

    # displ_pinn = utils_mechanics_pinn.compute_displacement(
    #     params_list,
    #     nn_list,
    #     mesh.nodes,
    # )
    # mesh.node_data["displacement_pinn"] = displ_pinn
    # mesh_deformed_pinn = deform_mesh(displ_pinn)

    return mesh, mesh_deformed_exact, mesh_deformed_nn, None


def save_mesh(mesh, filename):
    # Ensure nodes is a numpy array
    nodes = np.asarray(mesh.nodes)

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
                cell_data_combined[key] = (np.vstack([face_val, val]), comp_names)
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
                    cell_data_combined[key] = (np.vstack([padding, val]), None)
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
                cell_data_combined[key] = (np.vstack([val, padding]), comp_names)

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


if __name__ == "__main__":

    # # make vtp for training and test points
    # cloud = add_dataset_labels()
    # cloud.save("./results/dataset_labels.vtp")

    # make vtk with loss
    # mesh = utils_postprocess.make_postprocess_domain()
    # mesh = add_loss(mesh)
    # save_mesh(mesh, "./results/loss.vtk")

    # make vtk with stresses, stresses' errors, and tractions
    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh = add_stresses(mesh)
    mesh = add_tractions(mesh)
    save_mesh(mesh, "./results/stresses_tractions.vtk")

    # make vtk with displacements and errors
    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()
    mesh, mesh_deformed_exact, mesh_deformed_nn, mesh_deformed_pinn = add_displacement(
        mesh
    )
    save_mesh(mesh, "./results/displacement.vtk")
    save_mesh(mesh_deformed_exact, "./results/displacement_mesh_deformed_exact.vtk")
    save_mesh(mesh_deformed_nn, "./results/displacement_mesh_deformed_nn.vtk")
    # save_mesh(mesh_deformed_pinn, "./results/displacement_mesh_deformed_pinn.vtk")

    print("\nDone!")
