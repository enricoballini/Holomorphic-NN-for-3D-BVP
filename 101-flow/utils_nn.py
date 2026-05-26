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
from jax.tree_util import tree_map, tree_leaves, tree_reduce
from jax.flatten_util import ravel_pytree
import jax.scipy.sparse.linalg


import numpy as np


from utils_lapl import *

dprint = jax.debug.print
jax.config.update("jax_enable_x64", True)


# neural netwrork's arhitecture utils: ------------------------------------------------------


def split_dataset(
    point_coordinates, bc_vals, bc_type, normals, areas, weights_loss, tags, ratio=0.9
):
    """ """
    np.random.seed(0)

    tags_list = np.array([None] * point_coordinates.shape[0])
    for name, idx_tag in tags.items():
        tags_list[idx_tag] = name

    tmp_idx = np.arange(point_coordinates.shape[0])
    np.random.shuffle(tmp_idx)
    # print("NO SHUFFLING")

    point_coordinates = point_coordinates[tmp_idx]
    bc_vals = bc_vals[tmp_idx]
    bc_type = bc_type[tmp_idx]
    normals = normals[tmp_idx]
    areas = areas[tmp_idx]
    weights_loss = weights_loss[tmp_idx]
    tags_list = tags_list[tmp_idx]

    num_data = point_coordinates.shape[0]
    num_train = int(ratio * num_data)

    point_coordinates_train = point_coordinates[:num_train]
    bc_vals_train = bc_vals[:num_train]
    bc_type_train = bc_type[:num_train]
    normals_train = normals[:num_train]
    areas_train = areas[:num_train]
    weights_loss_train = weights_loss[:num_train]
    tags_list_train = tags_list[:num_train]

    point_coordinates_test = point_coordinates[num_train:]
    bc_vals_test = bc_vals[num_train:]
    bc_type_test = bc_type[num_train:]
    normals_test = normals[num_train:]
    areas_test = areas[num_train:]
    weights_loss_test = weights_loss[num_train:]
    tags_list_test = tags_list[num_train:]

    return (
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
    )


def generate_params(
    sizes, key, beta=0.5, scale_w=1, scale_b=0.01, dtype=jnp.complex128
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

    params_list["coeffs"] = jnp.ones(
        (
            len(
                tm_full_list,
            )
        )
    )

    return params_list


def activation_exp(z):
    return jnp.exp(z)


def activation_cos_sqrt(z):
    return jnp.cos(jnp.sqrt(z))


def activation_square(z):
    return z**2


def activation_prelu(x, a=0.1):
    real = jnp.maximum(0, x.real) + a * jnp.minimum(0, x.real)
    imag = jnp.maximum(0, x.imag) + a * jnp.minimum(0, x.imag)
    return real + 1j * imag


def make_list_chi_phi_and_der(
    chi_0,
):

    def scalarize(f):
        @jax.jit
        def wrapper(params, z, t):
            return f(params, z, t).squeeze()

        return wrapper

    def vmap_z(f):
        @jax.jit
        def wrapped(params, z, t):
            return jax.vmap(lambda zi: f(params, zi, t))(z)

        return wrapped

    def dz(f):
        @jax.jit
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
    ]

    der_0, der_1, der_2 = [], [], []

    for f in forwards:
        f_scalar = scalarize(f)
        der_0.append(f)
        der_1.append(vmap_z(dz(f_scalar)))
        der_2.append(vmap_z(dz(scalarize(dz(f_scalar)))))

    nn_list = namedtuple(
        "nn_list",
        ["chi_0"],
    )
    derivatives = namedtuple("derivatives", ["der_0"])

    return derivatives(
        der_0=nn_list(*der_0),
    )


# optimizers utils: --------------------------------------------------------------------


def loss_reg(params_list):
    return 0.001 * tree_reduce(
        lambda a, b: a + b, tree_map(lambda p: jnp.sum(p * p.conj()), params_list)
    )


def compute_loss_val(bc_type, nabla_V_g_n, bc_vals, weights_loss, epoch=None):

    # diff_1 = Dirichlet (missing)
    diff_2 = bc_type[:, 1] * (nabla_V_g_n - bc_vals)

    return (jnp.mean(weights_loss * diff_2**2)).astype(jnp.complex64)


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
    nodes_int,
    weights_int,
    der_displ_fcn=None,
    epoch=None,
):
    """ """

    tm_grid = jnp.array(tm_grid)

    V_g = potential_guess(
        params_list,
        nn_list,
        tm_grid,
        tm_full_list,
        nodes_int,
        weights_int,
        reductions,
        point_coordinates,
    )

    gradients_V = compute_gradients(
        lambda point: potential_guess(
            params_list,
            nn_list,
            tm_grid,
            tm_full_list,
            nodes_int,
            weights_int,
            reductions,
            jnp.atleast_2d(point),
        ),
        point_coordinates,
    )
    nabla_V_g_n = gradients_dot_unit_vectors(gradients_V, normals)

    loss_val = compute_loss_val(
        bc_type, nabla_V_g_n, bc_vals, weights_loss, epoch=epoch
    )

    loss_mean = 0.0001 * jnp.mean(V_g) ** 2

    return loss_val + loss_mean, [loss_val, loss_mean]


@partial(
    jax.jit,
    static_argnames=[
        "loss_l2",
        "nn_list",
        "tm_list",
        "delta_tm",
        "reductions",
        "der_displ_fcn",
    ],
)
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
    nodes_int,
    weights_int,
    der_displ_fcn,
):
    """
    Note: These are not the derivatives used in gradient-based optimization.
    There is an extra minus sign here.

    See https://docs.jax.dev/en/latest/advanced-autodiff.html
    Note the different behavious between vjp and jvp.

    However, for consistency of notation with other works, I do not include the complex conjugate at this stage; it will be introduced in the subsequent functions.
    """

    def loss_fn(params_list):
        loss_val, _ = loss_l2(
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
            nodes_int,
            weights_int,
            der_displ_fcn,
        )
        return loss_val

    _, vjp_fun = jax.vjp(loss_fn, params_list)
    grads_all = vjp_fun(jnp.array(1.0 + 0j))[0]

    return grads_all


# Adam: -----------------------------------------------------


def adam_init(params_list):
    """ """
    momentums_list = {}

    for name, params in zip(params_list.keys(), params_list.values()):
        m = tree_map(lambda param: jnp.zeros_like(param), params)
        v = tree_map(lambda param: jnp.zeros_like(param), params)
        t = 0
        momentums_list[name] = {"m": m, "v": v, "t": t}
    return momentums_list


@jax.jit
def update_adam(
    params_list,
    grads_list,
    momentums_list,
    lr,
    lr_coeffs_scaling,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    clip_value=1e-2,
):

    names = params_list.keys()

    for name in names:
        params = params_list[name]
        grads = grads_list[name]
        momentums = momentums_list[name]

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

        direction = tree_map(
            lambda m_hat_i, v_hat_i: (
                jnp.clip(
                    (m_hat_i / (jnp.sqrt(v_hat_i) + eps)).real, -clip_value, clip_value
                )
                + 1j
                * jnp.clip(
                    (m_hat_i / (jnp.sqrt(v_hat_i) + eps)).imag, -clip_value, clip_value
                )
            ),
            m_hat,
            v_hat,
        )

        if name == "coeffs":
            params_list[name] = tree_map(
                lambda param, d: jnp.maximum(
                    param - lr_coeffs_scaling * lr * d,
                    1e-8,
                ),
                params,
                direction,
            )
        else:
            params_list[name] = tree_map(
                lambda param, d: param - lr * d,
                params,
                direction,
            )

        momentums_list[name] = {"m": m, "v": v, "t": t}

    return params_list, momentums_list
