from pdb import set_trace as st

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from functools import partial

import utils_data_and_folders
import utils_integral_geometry
import utils_mechanics
import utils_nn
import utils_postprocess
import case_settings

dprint = jax.debug.print
jax.config.update("jax_enable_x64", True)


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
        return directional_derivative(lambda x: directional_derivative(f, x, e), p, e)

    return (
        second_derivative(f, point, e_x)
        + second_derivative(f, point, e_y)
        + second_derivative(f, point, e_z)
    )


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


def bilaplacian(points, U):

    def laplacian_of_U_at(p):
        return laplacian(U, p)

    bilap_vmap = jax.vmap(lambda p: laplacian(laplacian_of_U_at, p))
    return bilap_vmap(points)


def potential_U(phi, chi, points):
    import utils_integral_geometry

    tm = jnp.array([[4.189683]])  # random number
    reductions = utils_integral_geometry.define_reductions()

    points = jnp.atleast_2d(points)
    zetas = utils_integral_geometry.compute_zetas(points, tm, reductions)

    zetas_0 = zetas[0]

    return jnp.real(jnp.conj(zetas_0) * phi(zetas_0, tm) + chi(zetas_0, tm))


def make_phi_chi():
    import utils_integral_geometry
    import case_settings
    import utils_nn

    nn_list = case_settings.define_nn_forwards_and_derivatives()

    default_architecture = [2, 8, 1]
    default_split = [1, 4, 1]
    architectures = {
        "chi_0": (default_architecture, default_split),
        "phi_0": (default_architecture, default_split),
        "chi_1": (default_architecture, default_split),
        "phi_1": (default_architecture, default_split),
        "chi_2": (default_architecture, default_split),
        "phi_2": (default_architecture, default_split),
    }

    seed = [0, 1, 2, 3, 4, 5, 6]

    key = [
        jax.random.key(seed[1]),
        jax.random.key(seed[2]),
        jax.random.key(seed[3]),
        jax.random.key(seed[4]),
        jax.random.key(seed[5]),
        jax.random.key(seed[6]),
    ]

    M = 1
    bias_rotation = 0
    tm_list, delta_tm = utils_nn.define_tm(M, bias_rotation)

    degree = 1
    nodes_int, weights_int = np.polynomial.legendre.leggauss(degree)
    tm_full_list = utils_integral_geometry.make_tm_full_list(tm_list, nodes_int)

    params_list = utils_nn.generate_params_list(architectures, key, tm_full_list)

    def phi(zeta, t):
        return nn_list.der_0.phi_0(params_list["phi_0"], zeta, t)

    def chi(zeta, t):
        return nn_list.der_0.chi_0(params_list["chi_0"], zeta, t)

    return phi, chi


def test_bilaplacian():

    n_pt = 20
    x = np.linspace(0, 1, n_pt)
    y = np.linspace(0, 1, n_pt)
    z = np.linspace(0, 1, n_pt)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    points = jnp.array(points)

    phi, chi = make_phi_chi()

    def U(zetas):
        return potential_U(phi, chi, zetas)

    residual = bilaplacian(points, U) - 0

    print(residual)

    assert np.allclose(residual, 0, rtol=0, atol=1e-9)
    print("\nTest passed: all residuals are small")


def test_equilibrium_weak():
    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()

    areas = mesh.boundary_triangle_areas
    normals = mesh.boundary_outward_normals

    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")
    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")
    tm_grid, delta_tm = utils_nn.define_tm(M, bias_rotation)
    tm_full_list = utils_integral_geometry.make_tm_full_list(tm_grid, nodes_int)
    reductions = utils_nn.define_reductions()

    idx_seed = 0
    params_list = utils_data_and_folders.load_params_list(idx_seed)
    nn_list = case_settings.define_nn_forwards_and_derivatives()

    # stress_nn = utils_nn.compute_sigma_tensor(
    #     params_list,
    #     nn_list,
    #     tm_grid,
    #     tm_full_list,
    #     delta_tm,
    #     points_boundary,
    #     reductions,
    #     nu,
    #     G,
    #     nodes_int,
    #     weights_int,
    # )

    jacobian_field = utils_mechanics.define_strain_from_displ()

    stress_nn = utils_postprocess._batched_sigma_from_displ(
        jacobian_field,
        tm_grid,
        tm_full_list,
        params_list,
        nn_list,
        mesh.boundary_triangle_centroids,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    residual_equilibrium = np.sum(
        np.einsum("ijk,ik->ij", stress_nn, normals) * areas[:, None], axis=0
    )

    err_equilibrium = 999

    return residual_equilibrium, err_equilibrium


def test_equilibrium_strong():
    mesh = utils_postprocess.make_postprocess_domain_plot_boundary_and_inner()

    points_inner = mesh.nodes  # contains both inner and boundary, used for simplicity

    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")
    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")
    tm_grid, delta_tm = utils_nn.define_tm(M, bias_rotation)
    reductions = utils_nn.define_reductions()

    tm_full_list = utils_integral_geometry.make_tm_full_list(tm_grid, nodes_int)

    idx_seed = 0
    params_list = utils_data_and_folders.load_params_list(idx_seed)

    for name in params_list.keys():
        params_list[name] = jax.tree_util.tree_map(
            lambda param: param.astype(jnp.complex128), params_list[name]
        )

    nn_list = case_settings.define_nn_forwards_and_derivatives()

    jacobian_field = utils_mechanics.define_strain_from_displ()

    def stress_tensor(point):
        return utils_mechanics.compute_sigma_tensor_from_displ(
            jacobian_field,
            tm_grid,
            tm_full_list,
            params_list,
            nn_list,
            jnp.atleast_2d(point),
            reductions,
            nu,
            G,
            nodes_int,
            weights_int,
        )

    div_sigma = jax.vmap(lambda pt: divergence(stress_tensor, pt))(points_inner)
    print("div sigma = ", div_sigma)

    assert np.max(np.abs(div_sigma) < 1e-9)

    return div_sigma


if __name__ == "__main__":
    # test_bilaplacian()

    residual_equilibrium, _ = test_equilibrium_weak()
    np.savetxt("./results/equilibrium_weak", residual_equilibrium)

    div_sigma = test_equilibrium_strong()
    np.savetxt("./results/equilibrium_strong", np.array(div_sigma.reshape(-1, 1)))

    print("\nDone!")
