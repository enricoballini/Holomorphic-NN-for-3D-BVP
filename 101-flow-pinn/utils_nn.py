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
    point_coordinates,
    mask_inner_points,
    bc_vals,
    bc_type,
    normals,
    areas,
    weights_loss,
    tags,
    ratio=0.9,
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
    mask_inner_points = mask_inner_points[tmp_idx]
    bc_vals = bc_vals[tmp_idx]
    bc_type = bc_type[tmp_idx]
    normals = normals[tmp_idx]
    areas = areas[tmp_idx]
    weights_loss = weights_loss[tmp_idx]
    tags_list = tags_list[tmp_idx]

    num_data = point_coordinates.shape[0]
    num_train = int(ratio * num_data)

    point_coordinates_train = point_coordinates[:num_train]
    mask_inner_points_train = mask_inner_points[:num_train]
    bc_vals_train = bc_vals[:num_train]
    bc_type_train = bc_type[:num_train]
    normals_train = normals[:num_train]
    areas_train = areas[:num_train]
    weights_loss_train = weights_loss[:num_train]
    tags_list_train = tags_list[:num_train]

    point_coordinates_test = point_coordinates[num_train:]
    mask_inner_points_test = mask_inner_points[num_train:]
    bc_vals_test = bc_vals[num_train:]
    bc_type_test = bc_type[num_train:]
    normals_test = normals[num_train:]
    areas_test = areas[num_train:]
    weights_loss_test = weights_loss[num_train:]
    tags_list_test = tags_list[num_train:]

    return (
        point_coordinates_train,
        mask_inner_points_train,
        bc_vals_train,
        bc_type_train,
        normals_train,
        areas_train,
        weights_loss_train,
        tags_list_train,
        point_coordinates_test,
        mask_inner_points_test,
        bc_vals_test,
        bc_type_test,
        normals_test,
        areas_test,
        weights_loss_test,
        tags_list_test,
    )


# def generate_params(
#     sizes,
#     key,
#     beta=0.5,
#     scale_w=1,
#     scale_b=0.01,
#     dtype=jnp.float32,
# ):
#     """ """
#     keys = jax.random.split(key, 2 * (len(sizes) - 1))
#     keys = keys.reshape(len(sizes) - 1, -1)
#     params = []
#     for k, (size_in, size_out) in zip(
#         keys,
#         zip(sizes[:-1], sizes[1:]),
#     ):
#         k_real_W, k_imag_W = jax.random.split(k[0])
#         k_real_b, k_imag_b = jax.random.split(k[1])

#         std = beta / (2 * size_in * jnp.exp(beta))

#         W_real = scale_w * std * jax.random.normal(k_real_W, (size_out, size_in))
#         W_imag = scale_w * std * jax.random.normal(k_imag_W, (size_out, size_in))

#         b_real = jax.random.uniform(
#             k_real_b, (size_out,), minval=-scale_b, maxval=scale_b
#         )
#         b_imag = jax.random.uniform(
#             k_imag_b, (size_out,), minval=-scale_b, maxval=scale_b
#         )

#         W = W_real
#         b = b_real

#         params.append({"W": W, "b": b})
#     return params


def generate_params(
    sizes,
    key,
    dtype=jnp.float32,
    mode="uniform",
):
    """ """
    keys = jax.random.split(key, 2 * (len(sizes) - 1))
    keys = keys.reshape(len(sizes) - 1, -1)
    params = []

    for k, (size_in, size_out) in zip(keys, zip(sizes[:-1], sizes[1:])):

        if mode == "uniform":
            limit = jnp.sqrt(6.0 / (size_in + size_out))
            W = jax.random.uniform(
                k[0], (size_out, size_in), minval=-limit, maxval=limit, dtype=dtype
            )
        elif mode == "normal":
            std = jnp.sqrt(2.0 / (size_in + size_out))
            W = std * jax.random.normal(k[0], (size_out, size_in), dtype=dtype)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'uniform' or 'normal'.")

        b = jnp.zeros((size_out,), dtype=dtype)

        params.append({"W": W, "b": b})

    return params


# def generate_params(
#     sizes,
#     key,
#     dtype=jnp.float32,
#     mode="uniform",
# ):
#     """ """
#     keys = jax.random.split(key, len(sizes) - 1)
#     params = []

#     for k, (size_in, size_out) in zip(keys, zip(sizes[:-1], sizes[1:])):

#         if mode == "normal":
#             std = jnp.sqrt(2.0 / size_in)
#             W = std * jax.random.normal(k, (size_out, size_in), dtype=dtype)

#         elif mode == "uniform":
#             limit = jnp.sqrt(6.0 / size_in)
#             W = jax.random.uniform(
#                 k, (size_out, size_in), minval=-limit, maxval=limit, dtype=dtype
#             )

#         b = jnp.zeros((size_out,), dtype=dtype)
#         params.append({"W": W, "b": b})

#     return params


def generate_params_list(architectures, keys):
    """ """
    nn_names = architectures.keys()
    params_list = {name: None for name in nn_names}

    for nn_name, key in zip(nn_names, keys):
        params_list[nn_name] = generate_params(architectures[nn_name], key)

    return params_list


def activation_exp(z):
    return jnp.exp(z)


def activation_smooth(z):
    # return jax.nn.sigmoid(z)
    # return jax.nn.silu(z)
    return jax.nn.soft_sign(z)


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
        def wrapper(params, z):
            return f(params, z).squeeze()

        return wrapper

    def vmap_z(f):
        @jax.jit
        def wrapped(params, z):
            return jax.vmap(lambda zi: f(params, zi))(z)

        return wrapped

    def dz(f):
        @jax.jit
        def df(params, z):
            _, val = jax.jvp(
                lambda z_: f(params, z_),
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


def compute_loss_val(
    bc_type,
    nabla_V_g_n,
    lapl_V,
    bc_vals,
    weights_loss,
    mask_inner_pts,
    epoch=None,
):

    eps = 1e-16
    mask_boudary_pts = 1 - mask_inner_pts

    # diff_1 = Dirichlet (missing)
    diff_2 = (bc_type[:, 1] * (nabla_V_g_n - bc_vals)) * mask_boudary_pts

    val_boundary = jnp.sum((weights_loss * diff_2**2), axis=0) / (
        jnp.sum(mask_boudary_pts) + eps
    )

    diff_inner = (lapl_V - 0) * mask_inner_pts[:, None]
    val_residual = jnp.einsum("ij,ij->", diff_inner, diff_inner) / (
        jnp.sum(mask_inner_pts) + eps
    )
    # if epoch == 0:
    #     st()

    return (1 * val_boundary + 1 * val_residual).astype(jnp.complex64)


def loss_l2(
    params_list,
    nn_list,
    point_coordinates,
    mask_inner_pts,
    bc_vals,
    bc_type,
    normals,
    weights_loss,
    der_displ_fcn=None,
    epoch=None,
):
    """ """

    V_g = potential_guess(
        params_list,
        nn_list,
        point_coordinates,
    )

    gradients_V = compute_gradients(
        lambda point: potential_guess(
            params_list,
            nn_list,
            jnp.atleast_2d(point),
        ),
        point_coordinates,
    )

    gradients_V = gradients_V.reshape(-1, 3)
    nabla_V_g_n = gradients_dot_unit_vectors(gradients_V, normals)

    lapl_V = compute_lapl(
        lambda pt: potential_guess(
            params_list,
            nn_list,
            jnp.atleast_2d(pt),
        )[0],
        point_coordinates,
    )

    loss_val = compute_loss_val(
        bc_type,
        nabla_V_g_n,
        lapl_V,
        bc_vals,
        weights_loss,
        mask_inner_pts,
        epoch=epoch,
    )

    loss_mean = 0.0001 * jnp.mean(V_g) ** 2

    return loss_val + loss_mean, [loss_val, loss_mean]


@partial(
    jax.jit,
    static_argnames=[
        "loss_l2",
        "nn_list",
        "der_displ_fcn",
    ],
)
def compute_grads(
    loss_l2,
    params_list,
    nn_list,
    point_coordinates,
    mask_inner_pts,
    bc_vals,
    bc_type,
    normals,
    weights_loss,
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
            point_coordinates,
            mask_inner_pts,
            bc_vals,
            bc_type,
            normals,
            weights_loss,
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

        params_list[name] = tree_map(
            lambda param, d: param - lr * d,
            params,
            direction,
        )

        momentums_list[name] = {"m": m, "v": v, "t": t}

    return params_list, momentums_list
