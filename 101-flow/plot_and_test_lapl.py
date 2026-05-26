from pdb import set_trace as st

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from functools import partial
import pyvista as pv

import utils_data_and_folders
import utils_integral_geometry
import utils_lapl
import utils_postprocess
import main
import utils_nn

dprint = jax.debug.print

jax.config.update("jax_enable_x64", True)


def integrate_fast_single_point(
    f,
    t_grid,
    tm_full_list,
    nodes_int,
    weights_int,
    nn_list,
    params_list,
    point_coordinate,
    reductions,
):
    """ """
    a, b = t_grid[:-1, None], t_grid[1:, None]  # shape (intervals, 1)
    t_nodes = (b - a) / 2 * nodes_int + (a + b) / 2  # shape (intervals, n)
    w_nodes = (b - a) / 2 * weights_int  # shape (intervals, n)

    f_vals = jax.vmap(
        lambda t_row: jax.vmap(
            lambda t: f(
                nn_list,
                params_list,
                jnp.atleast_2d(point_coordinate),
                reductions,
                jnp.atleast_1d(t),
                tm_full_list,
            )
        )(t_row)
    )(t_nodes)

    return jnp.sum(f_vals * w_nodes[..., None], axis=(0, 1))


def integrate_slow_single_point(
    f,
    t_grid,
    tm_full_list,
    nodes_int,
    weights_int,
    nn_list,
    params_list,
    point_coordinate,
    reductions,
):
    a, b = t_grid[:-1, None], t_grid[1:, None]  # shape (intervals, 1)
    t_nodes = (b - a) / 2 * nodes_int + (a + b) / 2  # shape (intervals, n)
    w_nodes = (b - a) / 2 * weights_int  # shape (intervals, n)

    n_intervals, n_nodes = t_nodes.shape
    f_vals = []

    for i in range(n_intervals):
        f_vals_row = []
        for j in range(n_nodes):
            t = t_nodes[i, j]
            val = f(
                nn_list,
                params_list,
                jnp.atleast_2d(point_coordinate),
                reductions,
                jnp.atleast_1d(t),
                tm_full_list,
            )
            f_vals_row.append(val)
        f_vals.append(f_vals_row)

    f_vals = jnp.array(f_vals)

    return jnp.sum(f_vals * w_nodes[..., None], axis=(0, 1))


def directional_derivative(f, point, direction):
    """ """
    _, val = jax.jvp(f, primals=(point,), tangents=(direction,))
    return val


def laplacian(f, point):
    """"""
    e_x = jnp.array([1.0, 0.0, 0.0])
    e_y = jnp.array([0.0, 1.0, 0.0])
    e_z = jnp.array([0.0, 0.0, 1.0])

    def second_derivative(f, p, e):
        return directional_derivative(lambda pt: directional_derivative(f, pt, e), p, e)

    return (
        second_derivative(f, point, e_x)
        + second_derivative(f, point, e_y)
        + second_derivative(f, point, e_z)
    )


def test_laplacian():

    point_coordinates_boundary, areas, normal, point_coordinates_inner = (
        utils_postprocess.make_postprocess_domain_boundary_and_inner_old()
    )

    seed = 0
    params_list = utils_data_and_folders.load_params_list(seed)

    params_chi_0 = jax.tree_util.tree_map(
        lambda param: param.astype(jnp.complex128), params_list["chi_0"]
    )
    params_list["chi_0"] = params_chi_0

    M = np.loadtxt("./data/M").astype(int)
    # M = 3  ###
    bias_rotation = np.loadtxt("./data/bias_rotation")
    tm_grid, delta_tm = utils_nn.define_tm(M, bias_rotation)

    reductions = utils_nn.define_reductions()
    nn_list = main.define_nn_forwards_and_derivatives()
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")
    # degree = 1  ###
    # nodes_int, weights_int = np.polynomial.legendre.leggauss(degree)  ###

    tm_full_list = utils_integral_geometry.make_tm_full_list(tm_grid, nodes_int)

    def V(point):
        return integrate_fast_single_point(
            utils_lapl.V_t,
            jnp.array(tm_grid),
            tm_full_list,
            nodes_int,
            weights_int,
            nn_list,
            params_list,
            point,
            reductions,
        )

    def V_slow(point):
        return integrate_slow_single_point(
            utils_lapl.V_t,
            jnp.array(tm_grid),
            tm_full_list,
            nodes_int,
            weights_int,
            nn_list,
            params_list,
            point,
            reductions,
        )

    laplacian_vmap = jax.vmap(lambda pt: laplacian(V, pt))
    laplacian_vmap_slow = jax.vmap(lambda pt: laplacian(V_slow, pt))

    residual = laplacian_vmap(point_coordinates_inner) - 0
    residual_slow = laplacian_vmap_slow(point_coordinates_inner) - 0

    potential = jax.vmap(lambda pt: V(pt))(point_coordinates_inner)
    potential_slow = jax.vmap(lambda pt: V_slow(pt))(point_coordinates_inner)

    print(potential[:6])
    print(potential_slow[:6])
    print(residual[:6])
    print(residual_slow[:6])

    assert np.allclose(residual, 0, rtol=0, atol=1e-1)
    print("\nTest passed: all residuals are small")


def plot_result_inner_points():
    """ """
    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()

    seed = 0
    params_list = utils_data_and_folders.load_params_list(seed)

    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")
    tm_grid, delta_tm = utils_nn.define_tm(M, bias_rotation)

    reductions = utils_nn.define_reductions()
    nn_list = main.define_nn_forwards_and_derivatives()
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")

    tm_full_list = utils_integral_geometry.make_tm_full_list(tm_grid, nodes_int)

    # Compute guessed potential
    V_g = utils_nn.potential_guess(
        params_list,
        nn_list,
        jnp.array(tm_grid),
        tm_full_list,
        nodes_int,
        weights_int,
        reductions,
        mesh.inner_nodes,
    )
    V_g = np.array(np.real(V_g)).ravel()  # np.real just to change the dtype

    velocity = utils_lapl.compute_gradients(
        lambda point: utils_nn.potential_guess(
            params_list,
            nn_list,
            jnp.array(tm_grid),
            tm_full_list,
            nodes_int,
            weights_int,
            reductions,
            jnp.atleast_2d(point),
        ),
        mesh.inner_nodes,
    )

    # Create PolyData (point cloud)
    cloud = pv.PolyData(
        mesh.inner_nodes,
    )

    # Add node values
    cloud.point_data["V_g"] = V_g
    cloud.point_data["velocity"] = jnp.real(velocity)

    # Save as .vtu (unstructured format)
    cloud.save("./results/potentials_inner.vtk")


if __name__ == "__main__":
    # test_laplacian()
    plot_result_inner_points()
