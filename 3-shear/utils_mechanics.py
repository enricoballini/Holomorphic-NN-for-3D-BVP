import time
import pickle
from collections import namedtuple
from pdb import set_trace as st

from functools import partial
import scipy
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
from jax.tree_util import tree_leaves
from jax.tree_util import tree_reduce
import numpy as np

from utils_integral_geometry import *


dprint = jax.debug.print


def compute_sigma_xx_tm():
    """ """


def compute_sigma_yy_tm():
    """ """


def compute_sigma_zz_tm():
    """ """


def compute_sigma_xy_tm():
    """ """


def compute_sigma_zx_tm():
    """ """


def compute_sigma_yz_tm():
    """ """


def compute_u_single_tm(
    point_coordinates,
    phi_0_evaluated,
    der_x_chi_evaluated,
    der_x_phi_0_evaluated,
    der_x_phi_1_evaluated,
    der_x_phi_2_evaluated,
    tm_full_list,  # for h
    nu,
    G,
):
    """ """
    upsilon = 1 / (4 * (1 - nu))
    u_tm = (
        (1 - upsilon) * phi_0_evaluated
        - der_x_chi_evaluated
        - upsilon
        * (
            point_coordinates[:, 0] * der_x_phi_0_evaluated
            + point_coordinates[:, 1] * der_x_phi_1_evaluated
            + point_coordinates[:, 2] * der_x_phi_2_evaluated
        )
    )

    # TODO: add function h
    return u_tm / (2 * G)


def compute_v_single_tm(
    point_coordinates,
    phi_1_evaluated,
    der_y_chi_evaluated,
    der_y_phi_0_evaluated,
    der_y_phi_1_evaluated,
    der_y_phi_2_evaluated,
    tm_full_list,  # for h
    nu,
    G,
):
    """ """
    upsilon = 1 / (4 * (1 - nu))
    v_tm = (
        (1 - upsilon) * phi_1_evaluated
        - der_y_chi_evaluated
        - upsilon
        * (
            point_coordinates[:, 0] * der_y_phi_0_evaluated
            + point_coordinates[:, 1] * der_y_phi_1_evaluated
            + point_coordinates[:, 2] * der_y_phi_2_evaluated
        )
    )

    # TODO: add function h
    return v_tm / (2 * G)


def compute_w_single_tm(
    point_coordinates,
    phi_2_evaluated,
    der_z_chi_evaluated,
    der_z_phi_0_evaluated,
    der_z_phi_1_evaluated,
    der_z_phi_2_evaluated,
    tm_full_list,  # for h
    nu,
    G,
):
    """ """
    upsilon = 1 / (4 * (1 - nu))
    w_tm = (
        (1 - upsilon) * phi_2_evaluated
        - der_z_chi_evaluated
        - upsilon
        * (
            point_coordinates[:, 0] * der_z_phi_0_evaluated
            + point_coordinates[:, 1] * der_z_phi_1_evaluated
            + point_coordinates[:, 2] * der_z_phi_2_evaluated
        )
    )

    # TODO: add function h
    return w_tm / (2 * G)


def compute_sigma_tensor_from_claw(strain_vector, nu, G):
    """ """
    C_11 = (
        2
        * G
        / (1 - 2 * nu)
        * jnp.array([[1 - nu, nu, nu], [nu, 1 - nu, nu], [nu, nu, 1 - nu]])
    )
    C_12 = jnp.zeros((3, 3))
    C_21 = jnp.zeros((3, 3))
    C_22 = jnp.array([[G, 0, 0], [0, G, 0], [0, 0, G]])
    C = jnp.block([[C_11, C_12], [C_21, C_22]])

    sigma_vector = jnp.einsum("ij, kj -> ki", C, strain_vector)

    sigma = jnp.stack(
        (
            sigma_vector[:, 0],
            sigma_vector[:, 3],
            sigma_vector[:, 5],
            sigma_vector[:, 3],
            sigma_vector[:, 1],
            sigma_vector[:, 4],
            sigma_vector[:, 5],
            sigma_vector[:, 4],
            sigma_vector[:, 2],
        ),
        axis=1,
    ).reshape(-1, 3, 3)

    return sigma


def _displ_integrand(
    params_list,
    nn_list,
    point_coordinates,
    tm,
    tm_full_list,
    reductions,
    nu,
    G,
):
    reduction = reductions[0]  # TODO: how should we choose the rotation axis?

    phi_0_eval = nn_list.der_0.phi_0(
        params_list["phi_0"], reduction, point_coordinates, tm
    )
    phi_1_eval = nn_list.der_0.phi_1(
        params_list["phi_1"], reduction, point_coordinates, tm
    )
    phi_2_eval = nn_list.der_0.phi_2(
        params_list["phi_2"], reduction, point_coordinates, tm
    )

    grad_chi = nn_list.der_1.chi(params_list["chi"], reduction, point_coordinates, tm)
    grad_phi_0 = nn_list.der_1.phi_0(
        params_list["phi_0"], reduction, point_coordinates, tm
    )
    grad_phi_1 = nn_list.der_1.phi_1(
        params_list["phi_1"], reduction, point_coordinates, tm
    )
    grad_phi_2 = nn_list.der_1.phi_2(
        params_list["phi_2"], reduction, point_coordinates, tm
    )
    u_ = compute_u_single_tm(
        point_coordinates,
        phi_0_eval.squeeze(),
        grad_chi[:, 0],
        grad_phi_0[:, 0],
        grad_phi_1[:, 0],
        grad_phi_2[:, 0],
        tm_full_list,
        nu,
        G,
    )
    v_ = compute_v_single_tm(
        point_coordinates,
        phi_1_eval.squeeze(),
        grad_chi[:, 1],
        grad_phi_0[:, 1],
        grad_phi_1[:, 1],
        grad_phi_2[:, 1],
        tm_full_list,
        nu,
        G,
    )
    w_ = compute_w_single_tm(
        point_coordinates,
        phi_2_eval.squeeze(),
        grad_chi[:, 2],
        grad_phi_0[:, 2],
        grad_phi_1[:, 2],
        grad_phi_2[:, 2],
        tm_full_list,
        nu,
        G,
    )
    return jnp.stack((u_, v_, w_), axis=1)


def compute_displacement(
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
):
    """ """

    tm_grid = jnp.array(tm_grid)  # TODO...

    def displ_integrand(t):
        return _displ_integrand(
            params_list,
            nn_list,
            point_coordinates,
            t,
            tm_full_list,
            reductions,
            nu,
            G,
        )

    displ = integrate_fast(
        displ_integrand,
        tm_grid,
        nodes_int,
        weights_int,
    )

    return displ


def define_strain_from_displ():

    def displ_at_point(
        tm_grid,
        tm_full_list,
        params_list,
        nn_list,
        point,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    ):
        def displ_integrand(t):
            return _displ_integrand(
                params_list,
                nn_list,
                point[None],
                t,
                tm_full_list,
                reductions,
                nu,
                G,
            )

        return integrate_fast(
            displ_integrand,
            tm_grid,
            nodes_int,
            weights_int,
        ).squeeze()

    def jacobian_field(
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
    ):
        return jax.vmap(
            lambda point: jax.jacobian(
                lambda pt: displ_at_point(
                    tm_grid,
                    tm_full_list,
                    params_list,
                    nn_list,
                    pt,
                    reductions,
                    nu,
                    G,
                    nodes_int,
                    weights_int,
                )
            )(point)
        )(point_coordinates)

    return jacobian_field


def compute_strain_from_displ(
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
):
    """ """
    tm_grid = jnp.array(tm_grid)

    J = jacobian_field(
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
    eps_xx = J[:, 0, 0:1]
    eps_yy = J[:, 1, 1:2]
    eps_zz = J[:, 2, 2:3]
    eps_xy = J[:, 0, 1:2] + J[:, 1, 0:1]
    eps_yz = J[:, 1, 2:3] + J[:, 2, 1:2]
    eps_zx = J[:, 2, 0:1] + J[:, 0, 2:3]
    return jnp.hstack((eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_zx))


def compute_sigma_tensor_from_displ(
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
):
    strain_vector = compute_strain_from_displ(
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
    return compute_sigma_tensor_from_claw(strain_vector, nu, G)


def compute_sigma_n(sigma_tensor, normals):
    return jnp.einsum("nij,nj->ni", sigma_tensor, normals)


def compute_sigma_t(sigma_tensor, tangents):
    return jnp.einsum("nij,nj->ni", sigma_tensor, tangents)
