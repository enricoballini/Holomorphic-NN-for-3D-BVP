import time
from collections import namedtuple
from pdb import set_trace as st

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
from jax.tree_util import tree_leaves
from jax.tree_util import tree_reduce
import numpy as np

from utils_integral_geometry import *

dprint = jax.debug.print


@partial(jax.jit, static_argnames=["nn_list", "reductions"])
def V_t(nn_list, params_list, point_coordinates, reductions, tm, tm_full_list):
    zetas = compute_zetas(point_coordinates, tm, reductions)
    zeta_2 = zetas[1]  # only rotation around one axis

    # return (
    #     h_fcn(tm, tm_full_list, params_list)
    #     * jnp.real(nn_list.der_0.chi_0(params_list["chi_0"], zeta_2, tm)).flatten()
    # )

    return jnp.real(nn_list.der_0.chi_0(params_list["chi_0"], zeta_2, tm)).flatten()


def potential_guess(
    params_list,
    nn_list,
    tm_grid,
    tm_full_list,
    nodes_int,
    weights_int,
    reductions,
    point_coordinates,
):

    return integrate_fast(
        V_t,
        tm_grid,
        tm_full_list,
        nodes_int,
        weights_int,
        nn_list,
        params_list,
        point_coordinates,
        reductions,
    )


@partial(jax.jit, static_argnames=["f"])
def derivative_x(f, point):
    _, df_dx = jax.jvp(f, primals=(point,), tangents=(jnp.array([1.0, 0, 0]),))
    return df_dx


@partial(jax.jit, static_argnames=["f"])
def derivative_y(f, point):
    _, df_dx = jax.jvp(f, primals=(point,), tangents=(jnp.array([0, 1.0, 0]),))
    return df_dx


@partial(jax.jit, static_argnames=["f"])
def derivative_z(f, point):
    _, df_dx = jax.jvp(f, primals=(point,), tangents=(jnp.array([0, 0, 1.0]),))
    return df_dx


@partial(jax.jit, static_argnames=["f"])
def compute_gradients(f, point_coordinates):

    df_dx = jax.vmap(lambda point: derivative_x(f, point))(point_coordinates)
    df_dy = jax.vmap(lambda point: derivative_y(f, point))(point_coordinates)
    df_dz = jax.vmap(lambda point: derivative_z(f, point))(point_coordinates)

    gradients = jnp.hstack((df_dx, df_dy, df_dz))

    return gradients


def compute_gradient(f, point_coordinate):
    df_dx = derivative_x(f, point_coordinate)
    df_dy = derivative_y(f, point_coordinate)
    df_dz = derivative_z(f, point_coordinate)

    gradients = jnp.hstack((df_dx, df_dy, df_dz))

    return gradients


@jax.jit
def gradients_dot_unit_vectors(gradients, unit_vectors):
    return jnp.einsum("ij,ij->i", gradients, unit_vectors)


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
        return lambda x: f(x)[i]

    val = (
        directional_derivative(component(f, 0), point, e_x)
        + directional_derivative(component(f, 1), point, e_y)
        + directional_derivative(component(f, 2), point, e_z)
    )
    return val
