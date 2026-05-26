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


dprint = jax.debug.print


def integrate_fast(
    f,
    t_grid,
    nodes_int,
    weights_int,
):
    """ """
    a, b = t_grid[:-1, None], t_grid[1:, None]
    t_all_quadrature_points = (b - a) / 2 * nodes_int + (a + b) / 2
    w_nodes = (b - a) / 2 * weights_int

    f_vals = jax.vmap(
        lambda t_single_interval: jax.vmap(
            lambda t: f(
                jnp.atleast_1d(t),
            )
        )(t_single_interval)
    )(t_all_quadrature_points)
    return jnp.sum(f_vals * w_nodes[..., None, None], axis=(0, 1))


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


def h_fcn(t, tm_full_list, params):
    idx = jnp.searchsorted(tm_full_list, t, side="left")[0]
    return params["coeffs"][0]["W_1"][idx]
