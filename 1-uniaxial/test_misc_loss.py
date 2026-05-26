""" """

import os
import pdb
from pdb import set_trace as st
import pickle
import time
from collections import namedtuple
from pdb import set_trace as st

from functools import partial
import jax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
from jax.tree_util import tree_leaves
from jax.tree_util import tree_reduce

import utils_data_and_folders

dprint = jax.debug.print

# ------------------------------------------------------------------------


def integrate(f, t_grid, n=2):
    """ """
    t_grid = jnp.array(t_grid)
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes = jnp.array(nodes)
    weights = jnp.array(weights)

    def integrate_interval(a, b):
        t_nodes = (b - a) / 2 * nodes + (a + b) / 2
        w_nodes = (b - a) / 2 * weights

        f_vals = jax.vmap(lambda t: f(jnp.atleast_1d(t)))(t_nodes)
        return jnp.sum(f_vals * w_nodes[:, None], axis=0)

    interval_results = jax.vmap(integrate_interval)(t_grid[:-1], t_grid[1:])
    return jnp.sum(interval_results, axis=0)


# @partial(jax.jit, static_argnames=("f", "nn_list", "reductions", "coeffs_abc"))
def integrate_fast(
    f,
    t_grid,
    tm_full_list,
    params_list,
    nn_list,
    point_coordinates,
    reductions,
    coeffs_abc,
    nu,
    nodes_int,
    weights_int,
):
    """ """
    a, b = t_grid[:-1, None], t_grid[1:, None]  # shape (intervals, 1)
    t_nodes = (b - a) / 2 * nodes_int + (a + b) / 2  # shape (intervals, n)
    w_nodes = (b - a) / 2 * weights_int  # shape (intervals, n)
    f_vals = jax.vmap(
        lambda t_row: jax.vmap(
            lambda t: f(
                params_list,
                nn_list,
                point_coordinates,
                jnp.atleast_1d(t),
                tm_full_list,
                reductions,
                coeffs_abc,
                nu,
            )
        )(t_row)
    )(t_nodes)
    return jnp.sum(f_vals * w_nodes[..., None], axis=(0, 1))


# geometry and pde related utils: -------------------------------------------------------------------


def coeffs_abc():
    """ """

    # zeta_1: ------------
    def a_1(tm):
        return 1j

    def b_1(tm):
        return jnp.sin(tm)

    def c_1(tm):
        return jnp.cos(tm)

    # zeta_2: ------------
    def a_2(tm):
        return jnp.cos(tm)

    def b_2(tm):
        return 1j

    def c_2(tm):
        return jnp.sin(tm)

    # zeta_3: ------------
    def a_3(tm):
        return jnp.sin(tm)

    def b_3(tm):
        return jnp.cos(tm)

    def c_3(tm):
        return 1j

    return ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3))


def define_reductions():
    """ """

    def r_1(coordinates, tm):
        ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3)) = coeffs_abc()
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = coordinates[:, 2]
        return a_1(tm) * x + b_1(tm) * y + c_1(tm) * z

    def r_2(coordinates, tm):
        ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3)) = coeffs_abc()
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = coordinates[:, 2]
        return a_2(tm) * x + b_2(tm) * y + c_2(tm) * z

    def r_3(coordinates, tm):
        ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3)) = coeffs_abc()
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = coordinates[:, 2]
        return a_3(tm) * x + b_3(tm) * y + c_3(tm) * z

    reductions = (r_1, r_2, r_3)
    return reductions


def define_tm(M, bias_rotation):
    """ """
    if M == 1:
        tm_list = np.array([0.0])
        delta_tm = 1.0
    else:
        tm_list = np.linspace(0, 2 * np.pi, M) + bias_rotation
        delta_tm = tm_list[1] - tm_list[0]
    return tuple(tm_list), delta_tm


def compute_zetas(point_coordinates, tm, reductions):
    return [reduction(point_coordinates, tm).reshape(-1, 1) for reduction in reductions]


def make_tm_full_list(tm_list, nodes_int):
    tm_list = jnp.array(tm_list)  ###
    tm_full_list = []
    for a, b in zip(tm_list[:-1], tm_list[1:]):
        tm_full_list.append((b - a) / 2 * nodes_int + (a + b) / 2)

    return jnp.array(tm_full_list).flatten()


# @jax.jit
def h_fcn(t, tm_full_list, params):
    idx = jnp.searchsorted(tm_full_list, t, side="left")[0]
    return params["coeffs"][idx]


# @partial(jax.jit, static_argnames=("nn_list", "reductions", "coeffs_abc"))
def compute_sigma_xx_tm(
    params_list,
    nn_list,
    point_coordinates,
    tm,
    tm_full_list,
    reductions,
    coeffs_abc,
    nu,
):
    """ """
    zetas = compute_zetas(point_coordinates, tm, reductions)

    abc = tree_map(lambda coeff: coeff(tm), coeffs_abc())
    ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3)) = abc

    term_1 = jnp.real(
        2 * nn_list.der_1.phi_0(params_list["phi_0"], zetas[0], tm)
        + jnp.conj(zetas[0]) * nn_list.der_2.phi_0(params_list["phi_0"], zetas[0], tm)
        + nn_list.der_2.chi_0(params_list["chi_0"], zetas[0], tm)
    )
    term_2 = jnp.real(
        2
        * (1 - c_2**2 * (1 - 2 * nu))
        * nn_list.der_1.phi_1(params_list["phi_1"], zetas[1], tm)
        - (1 - c_2**2)
        * (
            jnp.conj(zetas[1]) * nn_list.der_2.phi_1(params_list["phi_1"], zetas[1], tm)
            + nn_list.der_2.chi_1(params_list["chi_1"], zetas[1], tm)
        )
    )
    term_3 = jnp.real(
        2
        * (1 - b_3**2 * (1 - 2 * nu))
        * nn_list.der_1.phi_2(params_list["phi_2"], zetas[2], tm)
        - (1 - b_3**2)
        * (
            jnp.conj(zetas[2]) * nn_list.der_2.phi_2(params_list["phi_2"], zetas[2], tm)
            + nn_list.der_2.chi_2(params_list["chi_2"], zetas[2], tm)
        )
    )
    coeffs = h_fcn(tm, tm_full_list, params_list)

    return (
        term_1.flatten() * coeffs[0]
        + term_2.flatten() * coeffs[1]
        + term_3.flatten() * coeffs[2]
    )


# @partial(jax.jit, static_argnames=("nn_list", "reductions", "coeffs_abc"))
def compute_sigma_yy_tm(
    params_list,
    nn_list,
    point_coordinates,
    tm,
    tm_full_list,
    reductions,
    coeffs_abc,
    nu,
):
    """ """
    zetas = compute_zetas(point_coordinates, tm, reductions)

    abc = tree_map(lambda coeff: coeff(tm), coeffs_abc())
    ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3)) = abc

    term_1 = jnp.real(
        2
        * (1 - c_1**2 * (1 - 2 * nu))
        * nn_list.der_1.phi_0(params_list["phi_0"], zetas[0], tm)
        - (1 - c_1**2)
        * (
            jnp.conj(zetas[0]) * nn_list.der_2.phi_0(params_list["phi_0"], zetas[0], tm)
            + nn_list.der_2.chi_0(params_list["chi_0"], zetas[0], tm)
        )
    )

    term_2 = jnp.real(
        2 * nn_list.der_1.phi_1(params_list["phi_1"], zetas[1], tm)
        + jnp.conj(zetas[1]) * nn_list.der_2.phi_1(params_list["phi_1"], zetas[1], tm)
        + nn_list.der_2.chi_1(params_list["chi_1"], zetas[1], tm)
    )

    term_3 = jnp.real(
        2
        * (1 - a_3**2 * (1 - 2 * nu))
        * nn_list.der_1.phi_2(params_list["phi_2"], zetas[2], tm)
        - (1 - a_3**2)
        * (
            jnp.conj(zetas[2]) * nn_list.der_2.phi_2(params_list["phi_2"], zetas[2], tm)
            + nn_list.der_2.chi_2(params_list["chi_2"], zetas[2], tm)
        )
    )

    coeffs = h_fcn(tm, tm_full_list, params_list)

    return (
        term_1.flatten() * coeffs[0]
        + term_2.flatten() * coeffs[1]
        + term_3.flatten() * coeffs[2]
    )


# @partial(jax.jit, static_argnames=("nn_list", "reductions", "coeffs_abc"))
def compute_sigma_zz_tm(
    params_list,
    nn_list,
    point_coordinates,
    tm,
    tm_full_list,
    reductions,
    coeffs_abc,
    nu,
):
    """ """
    zetas = compute_zetas(point_coordinates, tm, reductions)

    abc = tree_map(lambda coeff: coeff(tm), coeffs_abc())
    ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3)) = abc

    term_1 = jnp.real(
        2
        * (1 - b_1**2 * (1 - 2 * nu))
        * nn_list.der_1.phi_0(params_list["phi_0"], zetas[0], tm)
        - (1 - b_1**2)
        * (
            jnp.conj(zetas[0]) * nn_list.der_2.phi_0(params_list["phi_0"], zetas[0], tm)
            + nn_list.der_2.chi_0(params_list["chi_0"], zetas[0], tm)
        )
    )

    term_2 = jnp.real(
        2
        * (1 - a_2**2 * (1 - 2 * nu))
        * nn_list.der_1.phi_1(params_list["phi_1"], zetas[1], tm)
        - (1 - a_2**2)
        * (
            jnp.conj(zetas[1]) * nn_list.der_2.phi_1(params_list["phi_1"], zetas[1], tm)
            + nn_list.der_2.chi_1(params_list["chi_1"], zetas[1], tm)
        )
    )

    term_3 = jnp.real(
        2 * nn_list.der_1.phi_2(params_list["phi_2"], zetas[2], tm)
        + jnp.conj(zetas[2]) * nn_list.der_2.phi_2(params_list["phi_2"], zetas[2], tm)
        + nn_list.der_2.chi_2(params_list["chi_2"], zetas[2], tm)
    )

    coeffs = h_fcn(tm, tm_full_list, params_list)

    return (
        term_1.flatten() * coeffs[0]
        + term_2.flatten() * coeffs[1]
        + term_3.flatten() * coeffs[2]
    )


# @partial(jax.jit, static_argnames=("nn_list", "reductions", "coeffs_abc"))
def compute_sigma_xy_tm(
    params_list,
    nn_list,
    point_coordinates,
    tm,
    tm_full_list,
    reductions,
    coeffs_abc,
    nu,
):
    """ """
    zetas = compute_zetas(point_coordinates, tm, reductions)

    abc = tree_map(lambda coeff: coeff(tm), coeffs_abc())
    ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3)) = abc

    term_1 = b_1 * jnp.imag(
        jnp.conj(zetas[0]) * nn_list.der_2.phi_0(params_list["phi_0"], zetas[0], tm)
        + nn_list.der_2.chi_0(params_list["chi_0"], zetas[0], tm)
    )

    term_2 = a_2 * jnp.imag(
        jnp.conj(zetas[1]) * nn_list.der_2.phi_1(params_list["phi_1"], zetas[1], tm)
        + nn_list.der_2.chi_1(params_list["chi_1"], zetas[1], tm)
    )

    term_3 = (
        a_3
        * b_3
        * jnp.real(
            (2 - 4 * nu) * nn_list.der_1.phi_2(params_list["phi_2"], zetas[2], tm)
            - jnp.conj(zetas[2])
            * nn_list.der_2.phi_2(params_list["phi_2"], zetas[2], tm)
            - nn_list.der_2.chi_2(params_list["chi_2"], zetas[2], tm)
        )
    )

    coeffs = h_fcn(tm, tm_full_list, params_list)

    return (
        term_1.flatten() * coeffs[0]
        + term_2.flatten() * coeffs[1]
        + term_3.flatten() * coeffs[2]
    )


# @partial(jax.jit, static_argnames=("nn_list", "reductions", "coeffs_abc"))
def compute_sigma_zx_tm(
    params_list,
    nn_list,
    point_coordinates,
    tm,
    tm_full_list,
    reductions,
    coeffs_abc,
    nu,
):
    """ """
    zetas = compute_zetas(point_coordinates, tm, reductions)

    abc = tree_map(lambda coeff: coeff(tm), coeffs_abc())
    ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3)) = abc

    term_1 = c_1 * jnp.imag(
        jnp.conj(zetas[0]) * nn_list.der_2.phi_0(params_list["phi_0"], zetas[0], tm)
        + nn_list.der_2.chi_0(params_list["chi_0"], zetas[0], tm)
    )

    term_2 = (
        a_2
        * c_2
        * jnp.real(
            (2 - 4 * nu) * nn_list.der_1.phi_1(params_list["phi_1"], zetas[1], tm)
            - jnp.conj(zetas[1])
            * nn_list.der_2.phi_1(params_list["phi_1"], zetas[1], tm)
            - nn_list.der_2.chi_1(params_list["chi_1"], zetas[1], tm)
        )
    )

    term_3 = a_3 * jnp.imag(
        jnp.conj(zetas[2]) * nn_list.der_2.phi_2(params_list["phi_2"], zetas[2], tm)
        + nn_list.der_2.chi_2(params_list["chi_2"], zetas[2], tm)
    )

    coeffs = h_fcn(tm, tm_full_list, params_list)

    return (
        term_1.flatten() * coeffs[0]
        + term_2.flatten() * coeffs[1]
        + term_3.flatten() * coeffs[2]
    )


# @partial(jax.jit, static_argnames=("nn_list", "reductions", "coeffs_abc"))
def compute_sigma_yz_tm(
    params_list,
    nn_list,
    point_coordinates,
    tm,
    tm_full_list,
    reductions,
    coeffs_abc,
    nu,
):
    """ """
    zetas = compute_zetas(point_coordinates, tm, reductions)

    abc = tree_map(lambda coeff: coeff(tm), coeffs_abc())
    ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3)) = abc

    term_1 = (
        b_1
        * c_1
        * jnp.real(
            (2 - 4 * nu) * nn_list.der_1.phi_0(params_list["phi_0"], zetas[0], tm)
            - jnp.conj(zetas[0])
            * nn_list.der_2.phi_0(params_list["phi_0"], zetas[0], tm)
            - nn_list.der_2.chi_0(params_list["chi_0"], zetas[0], tm)
        )
    )

    term_2 = c_2 * jnp.imag(
        jnp.conj(zetas[1]) * nn_list.der_2.phi_1(params_list["phi_1"], zetas[1], tm)
        + nn_list.der_2.chi_1(params_list["chi_1"], zetas[1], tm)
    )

    term_3 = b_3 * jnp.imag(
        jnp.conj(zetas[2]) * nn_list.der_2.phi_2(params_list["phi_2"], zetas[2], tm)
        + nn_list.der_2.chi_2(params_list["chi_2"], zetas[2], tm)
    )

    coeffs = h_fcn(tm, tm_full_list, params_list)

    return (
        term_1.flatten() * coeffs[0]
        + term_2.flatten() * coeffs[1]
        + term_3.flatten() * coeffs[2]
    )


# @partial(jax.jit, static_argnames=("nn_list", "reductions", "coeffs_abc"))
def compute_u_tm(
    params_list,
    nn_list,
    point_coordinates,
    tm,
    tm_full_list,
    reductions,
    coeffs_abc,
    nu,
):
    """ """
    zetas = compute_zetas(point_coordinates, tm, reductions)

    abc = tree_map(lambda coeff: coeff(tm), coeffs_abc())
    ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3)) = abc

    term_1 = jnp.imag(
        (3 - 4 * nu) * nn_list.der_0.phi_0(params_list["phi_0"], zetas[0], tm)
        + jnp.conj(zetas[0]) * nn_list.der_1.phi_0(params_list["phi_0"], zetas[0], tm)
        + nn_list.der_1.chi_0(params_list["chi_0"], zetas[0], tm)
    )
    term_2 = a_2 * jnp.real(
        (3 - 4 * nu) * nn_list.der_0.phi_1(params_list["phi_1"], zetas[1], tm)
        - jnp.conj(zetas[1]) * nn_list.der_1.phi_1(params_list["phi_1"], zetas[1], tm)
        - nn_list.der_1.chi_1(params_list["chi_1"], zetas[1], tm)
    )
    term_3 = a_3 * jnp.real(
        (3 - 4 * nu) * nn_list.der_0.phi_2(params_list["phi_2"], zetas[2], tm)
        - jnp.conj(zetas[2]) * nn_list.der_1.phi_2(params_list["phi_2"], zetas[2], tm)
        - nn_list.der_1.chi_2(params_list["chi_2"], zetas[2], tm)
    )

    # return jnp.stack((term_1.flatten(), term_2.flatten(), term_3.flatten()), axis=0) # keep this line. It returns the three terms in the correct format
    coeffs = h_fcn(tm, tm_full_list, params_list)

    return (
        term_1.flatten() * coeffs[0]
        + term_2.flatten() * coeffs[1]
        + term_3.flatten() * coeffs[2]
    )


# @partial(jax.jit, static_argnames=("nn_list", "reductions", "coeffs_abc"))
def compute_v_tm(
    params_list,
    nn_list,
    point_coordinates,
    tm,
    tm_full_list,
    reductions,
    coeffs_abc,
    nu,
):
    """ """
    zetas = compute_zetas(point_coordinates, tm, reductions)

    abc = tree_map(lambda coeff: coeff(tm), coeffs_abc())
    ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3)) = abc

    term_1 = b_1 * jnp.real(
        (3 - 4 * nu) * nn_list.der_0.phi_0(params_list["phi_0"], zetas[0], tm)
        - jnp.conj(zetas[0]) * nn_list.der_1.phi_0(params_list["phi_0"], zetas[0], tm)
        - nn_list.der_1.chi_0(params_list["chi_0"], zetas[0], tm)
    )

    term_2 = jnp.imag(
        (3 - 4 * nu) * nn_list.der_0.phi_1(params_list["phi_1"], zetas[1], tm)
        + jnp.conj(zetas[1]) * nn_list.der_1.phi_1(params_list["phi_1"], zetas[1], tm)
        + nn_list.der_1.chi_1(params_list["chi_1"], zetas[1], tm)
    )

    term_3 = b_3 * jnp.real(
        (3 - 4 * nu) * nn_list.der_0.phi_2(params_list["phi_2"], zetas[2], tm)
        - jnp.conj(zetas[2]) * nn_list.der_1.phi_2(params_list["phi_2"], zetas[2], tm)
        - nn_list.der_1.chi_2(params_list["chi_2"], zetas[2], tm)
    )

    coeffs = h_fcn(tm, tm_full_list, params_list)

    return (
        term_1.flatten() * coeffs[0]
        + term_2.flatten() * coeffs[1]
        + term_3.flatten() * coeffs[2]
    )


# @partial(jax.jit, static_argnames=("nn_list", "reductions", "coeffs_abc"))
def compute_w_tm(
    params_list,
    nn_list,
    point_coordinates,
    tm,
    tm_full_list,
    reductions,
    coeffs_abc,
    nu,
):
    """ """
    zetas = compute_zetas(point_coordinates, tm, reductions)

    abc = tree_map(lambda coeff: coeff(tm), coeffs_abc())
    ((a_1, b_1, c_1), (a_2, b_2, c_2), (a_3, b_3, c_3)) = abc

    term_1 = c_1 * jnp.real(
        (3 - 4 * nu) * nn_list.der_0.phi_0(params_list["phi_0"], zetas[0], tm)
        - jnp.conj(zetas[0]) * nn_list.der_1.phi_0(params_list["phi_0"], zetas[0], tm)
        - nn_list.der_1.chi_0(params_list["chi_0"], zetas[0], tm)
    )

    term_2 = c_2 * jnp.real(
        (3 - 4 * nu) * nn_list.der_0.phi_1(params_list["phi_1"], zetas[1], tm)
        - jnp.conj(zetas[1]) * nn_list.der_1.phi_1(params_list["phi_1"], zetas[1], tm)
        - nn_list.der_1.chi_1(params_list["chi_1"], zetas[1], tm)
    )

    term_3 = jnp.imag(
        (3 - 4 * nu) * nn_list.der_0.phi_2(params_list["phi_2"], zetas[2], tm)
        + jnp.conj(zetas[2]) * nn_list.der_1.phi_2(params_list["phi_2"], zetas[2], tm)
        + nn_list.der_1.chi_2(params_list["chi_2"], zetas[2], tm)
    )

    coeffs = h_fcn(tm, tm_full_list, params_list)

    return (
        term_1.flatten() * coeffs[0]
        + term_2.flatten() * coeffs[1]
        + term_3.flatten() * coeffs[2]
    )


def compute_sigma_tensor(
    params_list,
    nn_list,
    tm_grid,
    tm_full_list,
    delta_tm,
    point_coordinates,
    reductions,
    nu,
    G,
    nodes_int,
    weights_int,
):
    """ """
    scaling = np.loadtxt("./results/scaling")
    top_pressure = -1000 / (np.pi * (30 / scaling) ** 2)

    with open("./data/tags", "rb") as fle:
        tags = pickle.load(fle)

    n_pt = point_coordinates.shape[0]

    sigma_xx = np.zeros((n_pt, 1))
    sigma_yy = np.zeros((n_pt, 1))
    sigma_zz = np.zeros((n_pt, 1))
    sigma_xy = np.zeros((n_pt, 1))
    sigma_yz = np.zeros((n_pt, 1))
    sigma_zx = np.zeros((n_pt, 1))

    sigma_zz[tags["supp"]] = top_pressure

    sigma_xx[tags["fix"]] = 999
    sigma_yy[tags["fix"]] = 999
    sigma_zz[tags["fix"]] = 999
    sigma_xy[tags["fix"]] = 999
    sigma_yz[tags["fix"]] = 999
    sigma_zx[tags["fix"]] = 999

    sigma_xx = jnp.array(sigma_xx)
    sigma_yy = jnp.array(sigma_yy)
    sigma_zz = jnp.array(sigma_zz)
    sigma_xy = jnp.array(sigma_xy)
    sigma_yz = jnp.array(sigma_yz)
    sigma_zx = jnp.array(sigma_zx)

    sigma = jnp.stack(
        (
            sigma_xx,
            sigma_xy,
            sigma_zx,
            sigma_xy,
            sigma_yy,
            sigma_yz,
            sigma_zx,
            sigma_yz,
            sigma_zz,
        ),
        axis=1,
    ).reshape(-1, 3, 3)

    return sigma


def compute_displacement(
    params_list,
    nn_list,
    tm_grid,
    tm_full_list,
    delta_tm,
    point_coordinates,
    reductions,
    nu,
    G,
    nodes_int,
    weights_int,
):
    """ """
    n_pt = point_coordinates.shape[0]

    u = jnp.zeros(n_pt)
    v = jnp.zeros(n_pt)
    w = jnp.zeros(n_pt)

    return jnp.stack((u, v, w), axis=1)


# neural netwrork's arhitecture utils: ------------------------------------------------------


class XY:
    """here it make sense to have a class to mange the data"""

    def __init__():
        pass


def split_dataset(
    point_coordinates, bc_vals, bc_type, normals, areas, weights_loss, ratio=0.9
):
    """ """
    np.random.seed(0)

    tmp_idx = np.arange(point_coordinates.shape[0])
    # np.random.shuffle(tmp_idx)
    print("NO SHUFFLING")

    point_coordinates = point_coordinates[tmp_idx]
    bc_vals = bc_vals[tmp_idx]
    bc_type = bc_type[tmp_idx]
    normals = normals[tmp_idx]
    areas = areas[tmp_idx]
    weights_loss = weights_loss[tmp_idx]

    num_data = point_coordinates.shape[0]
    num_train = int(ratio * num_data)

    point_coordinates_train = point_coordinates[:num_train]
    bc_vals_train = bc_vals[:num_train]
    bc_type_train = bc_type[:num_train]
    normals_train = normals[:num_train]
    areas_train = areas[:num_train]
    weights_loss_train = weights_loss[:num_train]

    point_coordinates_test = point_coordinates[num_train:]
    bc_vals_test = bc_vals[num_train:]
    bc_type_test = bc_type[num_train:]
    normals_test = normals[num_train:]
    areas_test = areas[num_train:]
    weights_loss_test = weights_loss[num_train:]

    return (
        point_coordinates_train,
        bc_vals_train,
        bc_type_train,
        normals_train,
        areas_train,
        weights_loss_train,
        point_coordinates_test,
        bc_vals_test,
        bc_type_test,
        normals_test,
        areas_test,
        weights_loss_test,
    )


def generate_params(
    sizes, key, beta=0.5, scale_w=1, scale_b=0.0001, dtype=jnp.complex64
):
    """ """
    keys = jax.random.split(key, 2 * (len(sizes[0]) - 1))
    keys = keys.reshape(len(sizes[0]) - 1, -1)
    params = []
    for k, (size_in, size_out, idx_split_in, idx_split_out) in zip(
        keys, zip(sizes[0][:-1], sizes[0][1:], sizes[1][:-1], sizes[1][1:])
    ):
        k_real_W, k_imag_W = jax.random.split(k[0])
        k_real_b, k_imag_b = jax.random.split(k[1])

        std = beta / (2 * size_in * jnp.exp(beta))

        W_real = scale_w * std * jax.random.normal(k_real_W, (size_out, size_in))
        W_imag = scale_w * std * jax.random.normal(k_imag_W, (size_out, size_in))

        b_real = jax.random.uniform(
            k_real_b, (size_out,), minval=-scale_b, maxval=scale_b
        )
        b_imag = jax.random.uniform(
            k_imag_b, (size_out,), minval=-scale_b, maxval=scale_b
        )

        W = (W_real + 1j * W_imag).astype(dtype)
        b = (b_real + 1j * b_imag).astype(dtype)

        W_1 = W[:idx_split_out, :idx_split_in]
        W_2 = W[:idx_split_out, idx_split_in:]
        W_3 = W[idx_split_out:, idx_split_in:]
        b_1 = b[:idx_split_out]
        b_2 = b[idx_split_out:]

        params.append({"W_1": W_1, "W_2": W_2, "W_3": W_3, "b_1": b_1, "b_2": b_2})
    return params


def generate_params_list(architectures, keys, tm_full_list):
    """ """
    nn_names = architectures.keys()
    params_list = {name: None for name in nn_names}

    for nn_name, key in zip(nn_names, keys):
        params_list[nn_name] = generate_params(architectures[nn_name], key)

    params_list["coeffs"] = jnp.ones((len(tm_full_list), 3))

    return params_list


def activation_exp(z):
    return jnp.exp(z)


def activation_cos_sqrt(z):
    return jnp.cos(jnp.sqrt(z))


def activation_prelu(x, a=0.1):
    return jnp.maximum(0, x) + a * jnp.minimum(0, x)


def make_list_chi_phi_and_der(
    chi_0,
    phi_0,
    chi_1,
    phi_1,
    chi_2,
    phi_2,
):

    def scalarize(f):
        # @jax.jit
        def wrapper(params, z, t):
            return f(params, z, t).squeeze()

        return wrapper

    def vmap_z(f):
        # @jax.jit
        def wrapped(params, z, t):
            return jax.vmap(lambda zi: f(params, zi, t))(z)

        return wrapped

    def dz(f):
        # @jax.jit
        def df(params, z, t):
            _, val = jax.jvp(
                lambda z_: f(params, z_, t),
                (z,),
                (jnp.ones_like(z),),
            )
            return val[..., None]

        return df

    forwards = [
        chi_0,
        phi_0,
        chi_1,
        phi_1,
        chi_2,
        phi_2,
    ]

    der_0, der_1, der_2 = [], [], []

    for f in forwards:
        f_scalar = scalarize(f)
        der_0.append(f)
        der_1.append(vmap_z(dz(f_scalar)))
        der_2.append(vmap_z(dz(scalarize(dz(f_scalar)))))

    Derivatives = namedtuple(
        "Derivatives",
        ["chi_0", "phi_0", "chi_1", "phi_1", "chi_2", "phi_2"],
    )
    NNList = namedtuple("NNList", ["der_0", "der_1", "der_2"])

    return NNList(
        der_0=Derivatives(*der_0),
        der_1=Derivatives(*der_1),
        der_2=Derivatives(*der_2),
    )


# optimizers utils: --------------------------------------------------------------------


# @jax.jit
def loss_reg(params_list):
    return 0.001 * tree_reduce(
        lambda a, b: a + b, tree_map(lambda p: jnp.sum(p * p.conj()), params_list)
    )


# #@jax.jit
def compute_loss_val(bc_type, displ_nn, sigma_n_nn, bc_vals, weights_loss):
    diff = bc_type[:, 0] * displ_nn + bc_type[:, 1] * sigma_n_nn - bc_vals

    return (
        jnp.mean(weights_loss * jnp.einsum("ij,ij->i", jnp.conj(diff), diff))
    ).astype(jnp.complex64)


# @jax.jit
def compute_sigma_n(sigma_tensor, normals):
    return jnp.einsum("nij,nj->ni", sigma_tensor, normals)


def loss_l2(
    params_list,
    nn_list,
    tm_grid,
    tm_full_list,
    delta_tm,
    reductions,
    point_coordinates,
    bc_vals,
    bc_type,
    normals,
    weights_loss,
    nu,
    G,
    nodes_int,
    weights_int,
):
    """ """
    sigma_tensor = compute_sigma_tensor(
        params_list,
        nn_list,
        tm_grid,
        tm_full_list,
        delta_tm,
        point_coordinates,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    sigma_n_nn = compute_sigma_n(sigma_tensor, normals)

    displ_nn = compute_displacement(
        params_list,
        nn_list,
        tm_grid,
        tm_full_list,
        delta_tm,
        point_coordinates,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    loss_val = compute_loss_val(bc_type, displ_nn, sigma_n_nn, bc_vals, weights_loss)
    # reg = loss_reg(params_list)

    print("\nloss_val = ", loss_val, "\n")
    assert np.isclose(loss_val, 0, rtol=0, atol=1e-15)
    print("\nTest passed\n")
    st()
    return loss_val  # + reg


# @partial(
#     jax.jit,
#     static_argnames=[
#         "loss_l2",
#         "nn_list",
#         "tm_list",
#         "delta_tm",
#         "reductions",
#     ],
# )
def compute_grads(
    loss_l2,
    params_list,
    nn_list,
    tm_list,
    tm_full_list,
    delta_tm,
    reductions,
    point_coordinates,
    bc_vals,
    bc_type,
    normals,
    weights_loss,
    nu,
    G,
    nodes_int,
    weights_int,
):
    """
    Note: These are not the derivatives used in gradient-based optimization.
    There is an extra minus sign here.

    See https://docs.jax.dev/en/latest/advanced-autodiff.html
    Note the different behavious between vjp and jvp.

    However, for consistency of notation with other works, I do not include the complex conjugate at this stage; it will be introduced in the subsequent functions.
    """

    def loss_fn(params_list):
        return loss_l2(
            params_list,
            nn_list,
            tm_list,
            tm_full_list,
            delta_tm,
            reductions,
            point_coordinates,
            bc_vals,
            bc_type,
            normals,
            weights_loss,
            nu,
            G,
            nodes_int,
            weights_int,
        )

    _, vjp_fun = jax.vjp(loss_fn, params_list)
    grads_all = vjp_fun(jnp.array(1.0 + 0j))[0]
    return grads_all


class AdaptiveLR:
    def __init__(
        self,
        init_lr,
        window_size=7,
        osc_threshold=0.03,
        low_threshold=0.003,
        beta=0.98,
        gamma=1.01,
        min_lr=1e-6,
        max_lr=1.0,
    ):
        """ """
        self.lr = init_lr
        self.window_size = window_size
        self.loss_history = []
        self.osc_threshold = osc_threshold
        self.low_threshold = low_threshold
        self.beta = beta
        self.gamma = gamma
        self.min_lr = min_lr
        self.max_lr = max_lr

    def update(self, loss):
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)

        if len(self.loss_history) < self.window_size:
            return self.lr  # not enough data yet

        # compute trend using simple linear regression
        y = np.array(self.loss_history)
        x = np.arange(len(y))
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

        trend = slope * x + intercept
        oscillation = np.mean(np.abs(y - trend)) / np.mean(np.abs(y))

        if oscillation > self.osc_threshold:
            self.lr = max(self.lr * self.beta, self.min_lr)
        elif oscillation < self.low_threshold and slope < 0:
            self.lr = min(self.lr * self.gamma, self.max_lr)

        if slope > 0:
            self.lr = max(self.lr * self.beta, self.min_lr)

        return self.lr


# Adam: -----------------------------------------------------


def debug_print_global_max(grads, lr):
    leaves = tree_leaves(grads)
    max_per_leaf = [jnp.max(jnp.abs(g)) for g in leaves]
    global_max = jnp.max(jnp.stack(max_per_leaf))

    def true_fun(_):
        jax.debug.print("max grad: {}", global_max)
        return 0

    def false_fun(_):
        return 0

    lax.cond(global_max > 10 * lr, true_fun, false_fun, operand=None)
    return global_max


def debug_print_min_coeffs(coeffs):
    """ """
    min_min = jnp.min(coeffs)

    def true_fun(_):
        jax.debug.print("min min: {}", min_min)
        return 0

    def false_fun(_):
        return 0

    lax.cond(min_min < 1e-5, true_fun, false_fun, operand=None)


def adam_init(params_list):
    """ """
    momentums_list = {}

    for name, params in zip(params_list.keys(), params_list.values()):
        m = tree_map(lambda param: jnp.zeros_like(param), params)
        v = tree_map(lambda param: jnp.zeros_like(param), params)
        t = 0
        momentums_list[name] = {"m": m, "v": v, "t": t}
    return momentums_list


# @jax.jit
def update_adam(
    params_list,
    grads_list,
    momentums_list,
    lr,
    lr_coeffs_scaling,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
):

    names = params_list.keys()

    # debug_print_min_coeffs(params_list["coeffs"])

    for name, (params, grads, momentums) in zip(
        names, zip(params_list.values(), grads_list.values(), momentums_list.values())
    ):

        # debug_print_global_max(grads, lr)

        m = momentums["m"]
        v = momentums["v"]
        t = momentums["t"] + 1

        m = tree_map(
            lambda m_i, g_i: beta1 * m_i + (1.0 - beta1) * jnp.conj(g_i), m, grads
        )
        v = tree_map(
            lambda v_i, g_i: beta2 * v_i + (1.0 - beta2) * (g_i * jnp.conj(g_i)),
            v,
            grads,
        )

        m_hat = tree_map(lambda m_i: m_i / (1.0 - beta1**t), m)
        v_hat = tree_map(lambda v_i: v_i / (1.0 - beta2**t), v)

        if name == "coeffs":
            params_list[name] = tree_map(
                lambda param, m_hat_i, v_hat_i: jnp.maximum(
                    param
                    - lr_coeffs_scaling * lr * m_hat_i / (jnp.sqrt(v_hat_i) + eps),
                    1e-8,
                ),
                params,
                m_hat,
                v_hat,
            )
        else:
            params_list[name] = tree_map(
                lambda param, m_hat_i, v_hat_i: param
                - lr * m_hat_i / (jnp.sqrt(v_hat_i) + eps),
                params,
                m_hat,
                v_hat,
            )

        momentums_list[name] = {"m": m, "v": v, "t": t}

    return params_list, momentums_list


# boundary data --------------------------------


def generate_boundary_data():
    r"""
    The boudary conditions are implemented only in the robi form:

        $a \circ u + b \circ \sigma n = bc_val$

    where $\circ$ is the Hadamard product.

    It is up to the user to define properly the vectors a and b.
    """

    scaling = np.loadtxt("./results/scaling")

    with open("./data/tags", "rb") as fle:
        tags = pickle.load(fle)

    tags = tree_map(lambda tag: tag.astype(int), tags)

    point_coordinates = np.loadtxt("./results/centroids_coordinates")
    normals = np.loadtxt("./results/face_normals")
    areas = np.loadtxt("./results/face_areas")

    n_pt = point_coordinates.shape[0]

    bc_type = np.zeros((n_pt, 2, 3))

    # define vectors a:
    bc_type[tags["fix"], 0] = np.array([1, 1, 1])
    # bc_type[tags["supp"], 0] = np.array([1, 1, 1])

    # define vectors b:
    bc_type[tags["supp"], 1] = np.array([1, 1, 1])
    bc_type[tags["free"], 1] = np.array([1, 1, 1])

    bc_vals = np.zeros_like(point_coordinates)

    top_pressure = -1000 / (
        np.pi * (30 / scaling) ** 2
    )  # XXX N applied to top surface np.pi * 30**2 ### THERE IS A SCALING FACTOR OF 300

    # top_pressure = -1
    print("top_pressure = ", top_pressure)

    bc_vals[tags["supp"], 2] = top_pressure

    return point_coordinates, bc_vals, bc_type, normals, areas, tags


def compute_weights_loss(areas, tags):
    """ """
    weights_loss = np.ones(areas.shape[0])
    weights_loss[tags["supp"]] = 1
    weights_loss[tags["fix"]] = 1
    return weights_loss


# model ----------------------------------------------------


def define_nn_forwards_and_derivatives():
    """ """

    # @jax.jit
    def common(params, zeta, t):
        scaling_nn = 300
        zeta = zeta / scaling_nn

        z_hol = zeta
        z_gen = t

        for layer in params[:-1]:
            z_hol = activation_exp(
                z_hol @ layer["W_1"].T + z_gen @ layer["W_2"].T + layer["b_1"]
            )
            z_gen = activation_prelu(z_gen @ layer["W_3"].T + layer["b_2"])

        z_hol = (
            z_hol @ params[-1]["W_1"].T
            + z_gen @ params[-1]["W_2"].T
            + params[-1]["b_1"]
        )
        # z_gen = z_gen @ params[-1]["W_3"].T + params[-1]["b_2"] # the output is scalar, so no z_gen

        return z_hol

    def chi_0_forward(params, z, t):
        """ """
        return common(params, z, t)

    def phi_0_forward(params, z, t):
        """ """
        return common(params, z, t)

    def chi_1_forward(params, z, t):
        """ """
        return common(params, z, t)

    def phi_1_forward(params, z, t):
        """ """
        return common(params, z, t)

    def chi_2_forward(params, z, t):
        """ """
        return common(params, z, t)

    def phi_2_forward(params, z, t):
        """ """
        return common(params, z, t)

    nn_list = make_list_chi_phi_and_der(
        chi_0_forward,
        phi_0_forward,
        chi_1_forward,
        phi_1_forward,
        chi_2_forward,
        phi_2_forward,
    )
    return nn_list


# training loop -------------------------------------------


def train_adam(
    epochs,
    lr_1,
    lr_2,
    lr_3,
    lr_coeffs_scaling,
    batch_size,
    params_list,
    nn_list,
    tm_list,
    tm_full_list,
    delta_tm,
    reductions,
    point_coordinates_train,
    bc_vals_train,
    bc_type_train,
    normals_train,
    weights_loss_train,
    point_coordinates_test,
    bc_vals_test,
    bc_type_test,
    normals_test,
    weights_loss_test,
    test_every,
    nu,
    G,
    nodes_int,
    weights_int,
):
    """ """
    num_data = point_coordinates_train.shape[0]
    num_batches = np.maximum(int(np.floor(num_data / batch_size)), 1)

    momentums_list = adam_init(params_list)

    # adaptive_lr = utils_nn.AdaptiveLR(init_lr=lr, min_lr=min_lr, max_lr=max_lr)

    losses_train = []
    losses_test = []

    for epoch in range(epochs):
        idx_data = np.arange(num_data)
        np.random.shuffle(idx_data)
        indices_batches = np.split(
            idx_data[: num_batches * batch_size], num_batches
        ) + [idx_data[num_batches * batch_size :]]

        # _ = utils_nn.loss_l2(
        #     params_list,
        #     nn_list,
        #     tm_list,
        #     tm_full_list,
        #     delta_tm,
        #     reductions,
        #     # point_coordinates_train,
        #     # bc_vals_train,
        #     # bc_type_train,
        #     # normals_train,
        #     # weights_loss_train,
        #     np.vstack((point_coordinates_train, point_coordinates_test)),
        #     np.vstack((bc_vals_train, bc_vals_test)),
        #     np.vstack((bc_type_train, bc_type_test)),
        #     np.vstack((normals_train, normals_test)),
        #     np.concatenate((weights_loss_train, weights_loss_test)),
        #     nu,
        #     G,
        #     nodes_int,
        #     weights_int,
        # )
        # print("---")

        for indices_batch in indices_batches:
            # t1 = time.time()
            grads_list = compute_grads(
                loss_l2,
                params_list,
                nn_list,
                tm_list,
                tm_full_list,
                delta_tm,
                reductions,
                # point_coordinates_train[indices_batch],
                # bc_vals_train[indices_batch],
                # bc_type_train[indices_batch],
                # normals_train[indices_batch],
                # weights_loss_train[indices_batch],
                np.vstack((point_coordinates_train, point_coordinates_test)),
                np.vstack((bc_vals_train, bc_vals_test)),
                np.vstack((bc_type_train, bc_type_test)),
                np.vstack((normals_train, normals_test)),
                np.concatenate((weights_loss_train, weights_loss_test)),
                nu,
                G,
                nodes_int,
                weights_int,
            )
            t2 = time.time()

            if epoch <= 1000:
                lr = lr_1
            elif 1000 <= epoch < 10000:
                lr = lr_2
            else:
                lr = lr_3

            params_list, momentums_list = update_adam(
                params_list, grads_list, momentums_list, lr, lr_coeffs_scaling
            )

        if epoch % test_every == 0:
            loss_train = loss_l2(
                params_list,
                nn_list,
                tm_list,
                tm_full_list,
                delta_tm,
                reductions,
                point_coordinates_train,
                bc_vals_train,
                bc_type_train,
                normals_train,
                weights_loss_train,
                nu,
                G,
                nodes_int,
                weights_int,
            )
            loss_test = loss_l2(
                params_list,
                nn_list,
                tm_list,
                tm_full_list,
                delta_tm,
                reductions,
                point_coordinates_test,
                bc_vals_test,
                bc_type_test,
                normals_test,
                weights_loss_test,
                nu,
                G,
                nodes_int,
                weights_int,
            )
            # lr = adaptive_lr.update(loss_l2_train)
            losses_train.append(loss_train)
            losses_test.append(loss_test)

            print(
                f"{epoch}, train: {np.real(loss_train):.4e}, test: {np.real(loss_test):.4e}, lr={lr}"
            )
    return params_list, losses_train, losses_test


if __name__ == "__main__":
    os.system("clear")

    # Alluminium
    nu = 0.34
    G = 24  # GPa

    # PET
    nu = 0.38
    G = 1.1  # GPa

    np.savetxt("./data/nu", np.array([nu]))
    np.savetxt("./data/G", np.array([G]))

    point_coordinates, bc_vals, bc_type, normals, areas, tags = generate_boundary_data()
    print(
        f"\nnumber of data = number of boundary points = {point_coordinates.shape[0]}"
    )

    M = 5
    bias_rotation = -(2 * np.pi) / (M - 1) / 2
    np.savetxt("./data/M", np.array([M]))
    np.savetxt("./data/bias_rotation", np.array([bias_rotation]))
    tm_list, delta_tm = define_tm(M, bias_rotation)

    reductions = define_reductions()

    nn_list = define_nn_forwards_and_derivatives()

    default_architecture = [2, 16, 1]
    default_split = [1, 8, 1]  # holomorphic part
    architectures = {
        "chi_0": (default_architecture, default_split),
        "phi_0": (default_architecture, default_split),
        "chi_1": (default_architecture, default_split),
        "phi_1": (default_architecture, default_split),
        "chi_2": (default_architecture, default_split),
        "phi_2": (default_architecture, default_split),
    }

    seeds = [[0, 1, 2, 3, 4, 5, 6]]  # , [7, 8, 9, 10, 11, 12, 13]]
    np.savetxt("./results/idx_seeds", np.arange(len(seeds)))

    weights_loss = compute_weights_loss(areas, tags)

    losses = {"train": [], "test": []}
    epochs = 51

    lr_1 = 1e-2
    lr_2 = 1e-3
    lr_3 = 1e-4
    lr_coeffs_scaling = 0.1

    batch_size = 999999

    np.savetxt("./results/num_epochs", np.array([epochs]))

    test_every = 10
    np.savetxt("./results/test_every", np.array([test_every]))

    degree = 2
    nodes_int, weights_int = np.polynomial.legendre.leggauss(degree)
    np.savetxt("./data/nodes_int", nodes_int)
    np.savetxt("./data/weights_int", weights_int)

    tm_full_list = make_tm_full_list(tm_list, nodes_int)

    for idx_seed, seed in enumerate(seeds):
        np.random.seed(seed[0])

        key = [
            jax.random.key(seed[1]),
            jax.random.key(seed[2]),
            jax.random.key(seed[3]),
            jax.random.key(seed[4]),
            jax.random.key(seed[5]),
            jax.random.key(seed[6]),
        ]

        params_list = generate_params_list(architectures, key, tm_full_list)

        (
            point_coordinates_train,
            bc_vals_train,
            bc_type_train,
            normals_train,
            areas_train,
            weights_loss_train,
            point_coordinates_test,
            bc_vals_test,
            bc_type_test,
            normals_test,
            areas_test,
            weights_loss_test,
        ) = split_dataset(
            point_coordinates, bc_vals, bc_type, normals, areas, weights_loss
        )

        params_list, losses_train, losses_test = train_adam(
            epochs,
            lr_1,
            lr_2,
            lr_3,
            lr_coeffs_scaling,
            batch_size,
            params_list,
            nn_list,
            tm_list,
            tm_full_list,
            delta_tm,
            reductions,
            point_coordinates_train,
            bc_vals_train,
            bc_type_train,
            normals_train,
            weights_loss_train,
            point_coordinates_test,
            bc_vals_test,
            bc_type_test,
            normals_test,
            weights_loss_test,
            test_every,
            nu,
            G,
            nodes_int,
            weights_int,
        )

        utils_data_and_folders.save_params_list(params_list, idx_seed)
        losses["train"].append(np.real(losses_train))
        losses["test"].append(np.real(losses_test))

    with open("./results/losses.pkl", "wb") as fle:
        pickle.dump(losses, fle)

    np.set_printoptions(threshold=99999)
    print("\n      h(t1),         h(t2),         h(t3),         tm")
    print(
        np.hstack(
            (
                params_list["coeffs"],
                tm_full_list.reshape(tm_full_list.shape[0], -1) * 180 / np.pi,
            )
        )
    )
    print("\nDone!")
