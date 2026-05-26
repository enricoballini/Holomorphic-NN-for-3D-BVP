import os
from pdb import set_trace as st
import pickle
import numpy as np
from numpy.polynomial.legendre import leggauss
import jax
import jax.numpy as jnp

import utils_data_and_folders
import utils_integral_geometry
import utils_mechanics
import utils_nn
import utils_postprocess
import case_settings


jax.config.update("jax_enable_x64", True)


def initialize_setup(
    default_architecture=np.array([2, 2, 1]),
    default_split=np.array([1, 1, 1]),
    scale_w=1,
    scale_b=0,
):
    nu = 0.25
    G = 1

    point_coordinates, bc_vals, bc_type, normals, areas, tags = (
        case_settings.generate_boundary_data()
    )

    M = 16
    bias_rotation = 0

    degree = 13
    nodes_int, weights_int = np.polynomial.legendre.leggauss(degree)

    seed = np.array([0, 1, 2, 3, 4, 5, 6])

    tm_grid, delta_tm = utils_nn.define_tm(M, bias_rotation)

    reductions = utils_nn.define_reductions()

    nn_list = case_settings.define_nn_forwards_and_derivatives()

    architectures = {
        "chi": (default_architecture, default_split),
        "phi_0": (default_architecture, default_split),
        "phi_1": (default_architecture, default_split),
        "phi_2": (default_architecture, default_split),
    }

    np.random.seed(seed[0])

    key = [
        jax.random.key(seed[1]),
        jax.random.key(seed[2]),
        jax.random.key(seed[3]),
        jax.random.key(seed[4]),
        jax.random.key(seed[5]),
        jax.random.key(seed[6]),
    ]

    tm_full_list = utils_integral_geometry.make_tm_full_list(tm_grid, nodes_int)

    params_list = utils_nn.generate_params_list(architectures, key, tm_full_list)
    nn_names = architectures.keys()
    params_list = {name: None for name in nn_names}

    for nn_name, key in zip(nn_names, key):
        params_list[nn_name] = utils_nn.generate_params(
            architectures[nn_name],
            key,
            beta=0.5,
            scale_w=scale_w,
            scale_b=scale_b,
        )

    params_list["coeffs"] = jax.tree.map(
        lambda p: jnp.zeros_like(p), params_list["chi"]
    )
    params_list["coeffs"][0]["W_1"] = jnp.ones((len(tm_full_list), 3))

    return (
        params_list,
        nn_list,
        tm_grid,
        tm_full_list,
        point_coordinates,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )


def test_displacement_shape():
    """ """
    (
        params_list,
        nn_list,
        tm_grid,
        tm_full_list,
        point_coordinates,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    ) = initialize_setup()

    displ = utils_mechanics.compute_displacement(
        params_list,
        nn_list,
        tm_grid,
        tm_full_list,
        point_coordinates,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    assert displ.shape == point_coordinates.shape
    print("test displacement shape passed!")


def test_derivatives_nn():

    (
        params_list,
        nn_list,
        tm_grid,
        tm_full_list,
        point_coordinates,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    ) = initialize_setup(
        default_architecture=np.array([2, 1]),
        default_split=np.array([1, 1]),
        scale_w=1,
        scale_b=0,
    )

    point_coordinates = point_coordinates[0:10]  # for simplicity

    reduction = reductions[0]
    tm = jnp.array([0.2])

    grad_chi = nn_list.der_1.chi(params_list["chi"], reduction, point_coordinates, tm)

    n_pt = point_coordinates.shape[0]

    dz_dx = 1j * np.ones(n_pt)[:, None]
    dz_dy = np.sin(tm) * np.ones(n_pt)[:, None]
    dz_dz = np.cos(tm) * np.ones(n_pt)[:, None]

    derivatives_x_exact = np.real(dz_dx @ params_list["chi"][0]["W_1"].T).squeeze()
    derivatives_y_exact = np.real(dz_dy @ params_list["chi"][0]["W_1"].T).squeeze()
    derivatives_z_exact = np.real(dz_dz @ params_list["chi"][0]["W_1"].T).squeeze()

    derivatives_x_nn = grad_chi[:, 0]
    derivatives_y_nn = grad_chi[:, 1]
    derivatives_z_nn = grad_chi[:, 2]

    assert grad_chi.shape == point_coordinates.shape
    assert np.allclose(derivatives_x_exact, derivatives_x_nn, rtol=0, atol=1e-15)
    assert np.allclose(derivatives_y_exact, derivatives_y_nn, rtol=0, atol=1e-15)
    assert np.allclose(derivatives_z_exact, derivatives_z_nn, rtol=0, atol=1e-15)
    print("test derivatives nn passed!")


def test_strain_tensor():
    (
        params_list,
        _,
        tm_grid,
        tm_full_list,
        point_coordinates,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    ) = initialize_setup(
        default_architecture=np.array([2, 1]),
        default_split=np.array([1, 1]),
        scale_w=1,
        scale_b=0,
    )

    def define_nn_forwards_and_derivatives():
        """ """

        def chi_forward(params, reduction, point_coordinates, t):
            return point_coordinates[:, 0] ** 2 + t

        def phi_0_forward(params, reduction, point_coordinates, t):
            return point_coordinates[:, 1] + t

        def phi_1_forward(params, reduction, point_coordinates, t):
            return t

        def phi_2_forward(params, reduction, point_coordinates, t):
            return 0 * t

        nn_list = utils_nn.make_list_chi_phi_and_der(
            chi_forward,
            phi_0_forward,
            phi_1_forward,
            phi_2_forward,
        )
        return nn_list

    def strain_exact(point_coordinates):
        """there is a 2 in extra diagonals because of a bad notation"""
        nu = 0.25
        G = 1
        upsilon = 1 / (4 * (1 - nu))

        eps = np.array(
            [-2 * np.pi / G, 0, 0, 2 * np.pi * (1 - 2 * upsilon) / (2 * G), 0, 0],
        )
        return np.tile(eps, (point_coordinates.shape[0], 1))

    nn_list = define_nn_forwards_and_derivatives()

    point_coordinates = point_coordinates[0:10]  # for simplicity

    reduction = reductions[0]
    tm = jnp.array([0.2])

    n_pt = point_coordinates.shape[0]

    eps_exact = strain_exact(point_coordinates)

    jacobian_field = utils_mechanics.define_strain_from_displ()
    eps_nn = utils_mechanics.compute_strain_from_displ(
        jacobian_field,
        tm_grid,
        tm_full_list,
        params_list,
        nn_list,
        point_coordinates,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    assert np.allclose(eps_exact, eps_nn, rtol=0, atol=1e-15)
    print("Test strain tensor passed!")


def make_uniform_grid(a, b, n):
    return jnp.linspace(a, b, n + 1)


def test_midpoint_linear_exact():
    """Midpoint rule integrates linear functions exactly."""
    f = lambda t: jnp.array([3.0 * t - 1.0])  # exact = 0.5
    a = 0.0
    b = 1.0

    degree = 1
    nodes_int, weights_int = np.polynomial.legendre.leggauss(degree)

    result = utils_integral_geometry.integrate_fast(
        f, make_uniform_grid(a, b, 1), nodes_int, weights_int
    )
    assert np.allclose(result[0, 0], 0.5, rtol=0, atol=1e-12)
    print("test midpoint linear exact passed!")


def test_midpoint_convergence_order2():
    """Midpoint rule should converge at order 2 on a smooth integrand."""
    f = lambda t: jnp.array([jnp.exp(-t), jnp.exp(-t), 2 * jnp.exp(-t)])
    a = 0.0
    b = 1.0

    exacts = np.array([1 - np.exp(-1), 1 - np.exp(-1), 2 * (1 - np.exp(-1))])

    degree = 1
    nodes_int, weights_int = np.polynomial.legendre.leggauss(degree)

    n_list = [4, 8, 16, 32]
    values = [
        utils_integral_geometry.integrate_fast(
            f, make_uniform_grid(a, b, n), nodes_int, weights_int
        )
        for n in n_list
    ]

    errors = [np.linalg.norm(value.ravel() - exacts) for value in values]

    log_h = np.log(1.0 / np.array(n_list, dtype=float))
    order, _ = np.polyfit(log_h, np.log(errors), 1)

    assert 1.9 <= order <= 2.1, f"Expected convergence order >= 2, got {order:.3f}"
    print(f"test midpoint convergence order2 passed! (order = {order:.3f})")


def test_midpoint_matches_gauss_legendre():
    """Midpoint and 4-pt Gauss-Legendre should agree to high precision on a fine grid."""
    f = lambda t: jnp.array([jnp.exp(-t) * jnp.cos(t)])
    a, b = 0.0, 2 * np.pi
    n = 2048

    degree = 1
    nodes_mp, weights_mp = np.polynomial.legendre.leggauss(degree)

    degree = 2
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(degree)

    result_mp = utils_integral_geometry.integrate_fast(
        f, make_uniform_grid(a, b, n), nodes_mp, weights_mp
    )
    result_gl = utils_integral_geometry.integrate_fast(
        f, make_uniform_grid(a, b, n), nodes_gl, weights_gl
    )

    assert np.allclose(result_mp, result_gl, rtol=0, atol=1e-6)
    print("test midpoint matches gauss legendre passed!")


if __name__ == "__main__":

    test_displacement_shape()
    test_derivatives_nn()
    test_strain_tensor()

    test_midpoint_linear_exact()
    test_midpoint_convergence_order2()
    test_midpoint_matches_gauss_legendre()


print("\nDone!")
