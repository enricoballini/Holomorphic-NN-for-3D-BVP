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
from jax import tree_util
from jax.tree_util import tree_map
from jax.tree_util import tree_leaves
from jax.tree_util import tree_reduce
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
import numpy as np

import optax


import utils_data_and_folders_pinn
from utils_mechanics_pinn import *

dprint = jax.debug.print
# jax.config.update("jax_enable_x64", True)


def split_dataset(
    point_coordinates,
    mask_inner_pts,
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

    point_coordinates = point_coordinates[tmp_idx]
    mask_inner_pts = mask_inner_pts[tmp_idx]
    bc_vals = bc_vals[tmp_idx]
    bc_type = bc_type[tmp_idx]
    normals = normals[tmp_idx]
    areas = areas[tmp_idx]
    weights_loss = weights_loss[tmp_idx]
    tags_list = tags_list[tmp_idx]

    num_data = point_coordinates.shape[0]
    num_train = int(ratio * num_data)

    point_coordinates_train = point_coordinates[:num_train]
    mask_inner_pts_train = mask_inner_pts[:num_train]
    bc_vals_train = bc_vals[:num_train]
    bc_type_train = bc_type[:num_train]
    normals_train = normals[:num_train]
    areas_train = areas[:num_train]
    weights_loss_train = weights_loss[:num_train]
    tags_list_train = tags_list[:num_train]

    point_coordinates_test = point_coordinates[num_train:]
    mask_inner_pts_test = mask_inner_pts[num_train:]
    bc_vals_test = bc_vals[num_train:]
    bc_type_test = bc_type[num_train:]
    normals_test = normals[num_train:]
    areas_test = areas[num_train:]
    weights_loss_test = weights_loss[num_train:]
    tags_list_test = tags_list[num_train:]

    return (
        point_coordinates_train,
        mask_inner_pts_train,
        bc_vals_train,
        bc_type_train,
        normals_train,
        areas_train,
        weights_loss_train,
        tags_list_train,
        point_coordinates_test,
        mask_inner_pts_test,
        bc_vals_test,
        bc_type_test,
        normals_test,
        areas_test,
        weights_loss_test,
        tags_list_test,
    )


def make_batch_idx(num_data, batch_size, seed):
    """ """
    np.random.seed(seed)
    num_batches = np.maximum(int(np.floor(num_data / batch_size)), 1)

    idx_data = np.arange(num_data)
    np.random.shuffle(idx_data)
    indices_batches = np.split(idx_data[: num_batches * batch_size], num_batches) + [
        idx_data[num_batches * batch_size :]
    ]
    return indices_batches


# def generate_params(
#     sizes,
#     key,
#     beta=0.5,
#     scale_w=1,
#     scale_b=0.0,
#     dtype=jnp.complex64,
# ):
#     """ """
#     keys = jax.random.split(key, 2 * (len(sizes[0]) - 1))
#     keys = keys.reshape(len(sizes[0]) - 1, -1)
#     params = []
#     for k, (size_in, size_out, idx_split_in, idx_split_out) in zip(
#         keys, zip(sizes[0][:-1], sizes[0][1:], sizes[1][:-1], sizes[1][1:])
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
    mode="normal",
):
    """
    Args:
        sizes:  Tuple of (layer_sizes, split_indices), e.g. ([784, 256, 10], [...]).
        key:    JAX random key.
        dtype:  Data type for parameters (default: jnp.float32).
        mode:   'normal' => He normal | 'uniform' => He uniform.
    """
    keys = jax.random.split(key, 2 * (len(sizes[0]) - 1))
    keys = keys.reshape(len(sizes[0]) - 1, -1)
    params = []

    for k, (size_in, size_out, idx_split_in, idx_split_out) in zip(
        keys, zip(sizes[0][:-1], sizes[0][1:], sizes[1][:-1], sizes[1][1:])
    ):
        if mode == "normal":
            std = jnp.sqrt(2.0 / size_in)
            W = std * jax.random.normal(k[0], (size_out, size_in), dtype=dtype)

        elif mode == "uniform":
            limit = jnp.sqrt(6.0 / size_in)
            W = jax.random.uniform(
                k[0], (size_out, size_in), minval=-limit, maxval=limit, dtype=dtype
            )

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'normal' or 'uniform'.")

        b = jnp.zeros((size_out,), dtype=dtype)
        params.append({"W": W, "b": b})

    return params


def generate_params_list(architectures, keys, tm_full_list):
    """ """
    nn_names = architectures.keys()
    params_list = {name: None for name in nn_names}

    for nn_name, key in zip(nn_names, keys):
        params_list[nn_name] = generate_params(architectures[nn_name], key)

    params_list["coeffs"] = tree_map(lambda p: jnp.zeros_like(p), params_list["chi"])
    params_list["coeffs"][0]["W_1"] = jnp.ones((len(tm_full_list), 3))
    return params_list


def activation_exp(z):
    return jnp.exp(z)


def activation_cos_sqrt(z):
    return jnp.cos(jnp.sqrt(z))


def activation_pol(z):
    return z + z**2


def activation_sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))


def activation_smooth(z):
    return jax.nn.silu(z)


def activation_prelu(x, a=0.1):
    real = jnp.maximum(0, x.real) + a * jnp.minimum(0, x.real)
    imag = jnp.maximum(0, x.imag) + a * jnp.minimum(0, x.imag)
    return real + 1j * imag


def make_list_chi_phi_and_der(
    chi,
    # phi_0,
    # phi_1,
    # phi_2,
):

    def scalarize(f):
        def wrapper(params, reduction, point_coordinates, t):
            return f(params, reduction, point_coordinates, t).squeeze()

        return wrapper

    def vmap_point_coordinates(f):
        def wrapped(params, reduction, point_coordinates, t):
            return jax.vmap(lambda pt: f(params, reduction, pt, t))(point_coordinates)

        return wrapped

    def grad_single_pt(f):
        def df(params, reduction, point_coordinate, t):
            grad_val = jax.grad(lambda pt: f(params, reduction, pt[None], t))(
                point_coordinate
            )
            return grad_val

        return df

    forwards = [
        chi,
        # phi_0,
        # phi_1,
        # phi_2,
    ]

    der_0, der_1, der_2 = [], [], []

    for f in forwards:
        f_scalar = scalarize(f)
        der_0.append(f)
        der_1.append(vmap_point_coordinates(grad_single_pt(f_scalar)))
        der_2.append("TODO")

    nn_list = namedtuple("nn_list", ["chi"])  # , "phi_0", "phi_1", "phi_2"],
    derivatives = namedtuple("derivatives", ["der_0", "der_1", "der_2"])

    return derivatives(
        der_0=nn_list(*der_0),
        der_1=nn_list(*der_1),
        der_2=nn_list(*der_2),
    )


# optimizers utils: --------------------------------------------------------------------


def loss_reg(params_list):
    params_no_coeffs = {k: v for k, v in params_list.items() if k != "coeffs"}
    total = tree_reduce(
        lambda a, b: a + b,
        tree_map(lambda p: jnp.sum(p * jnp.conj(p)), params_no_coeffs),
    )
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params_no_coeffs))
    return 0.0 * total / n_params


def compute_loss_val(
    bc_type, displ_nn, sigma_n_nn, bc_vals, div_sigma, mask_inner_pts, epoch=None
):

    eps = 1e-16
    mask_boudary_pts = 1 - mask_inner_pts

    diff_outer = (
        bc_type[:, 0] * displ_nn + bc_type[:, 1] * sigma_n_nn - bc_vals
    ) * mask_boudary_pts[:, None]
    val_boundary = jnp.einsum("ij,ij->", diff_outer, diff_outer) / (
        jnp.sum(mask_boudary_pts) + eps
    )

    diff_inner = (div_sigma - 0) * mask_inner_pts[:, None]
    val_residual = jnp.einsum("ij,ij->", diff_inner, diff_inner) / (
        jnp.sum(mask_inner_pts) + eps
    )

    return (val_boundary + val_residual).astype(jnp.complex64)


def loss_l2(
    params_list,
    nn_list,
    tm_grid,
    tm_full_list,
    reductions,
    point_coordinates,
    mask_inner_pts,
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
    jacobian_field = define_strain_from_displ()

    sigma_tensor = compute_sigma_tensor_from_displ(
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

    sigma_n_nn = compute_sigma_n(sigma_tensor, normals)

    displ_nn = compute_displacement(
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

    div_sigma = compute_div_sigma(
        params_list, nn_list, nu, G, point_coordinates, jacobian_field
    )

    loss_val = compute_loss_val(
        bc_type, displ_nn, sigma_n_nn, bc_vals, div_sigma, mask_inner_pts
    )

    reg = loss_reg(params_list)
    loss_val = loss_val + reg

    return loss_val, [loss_val, reg]


def compute_grads(
    loss_l2,
    params_list,
    nn_list,
    tm_grid,
    tm_full_list,
    reductions,
    point_coordinates,
    mask_inner_pts,
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
        loss_val, _ = loss_l2(
            params_list,
            nn_list,
            tm_grid,
            tm_full_list,
            reductions,
            point_coordinates,
            mask_inner_pts,
            bc_vals,
            bc_type,
            normals,
            weights_loss,
            nu,
            G,
            nodes_int,
            weights_int,
        )
        return loss_val

    loss_val, vjp_fun = jax.vjp(loss_fn, params_list)
    grads_all = vjp_fun(jnp.array(1.0 + 0j))[0]

    return jnp.real(loss_val), grads_all


# Adam: -----------------------------------------------------


def adam_init_state(params_list, current_epoch):

    lr_1 = np.loadtxt("./data/lr_1_adam")
    lr_2 = np.loadtxt("./data/lr_2_adam")
    lr_3 = np.loadtxt("./data/lr_3_adam")

    num_steps_per_epoch = np.loadtxt("./data/num_steps_per_epoch").astype(int)

    schedule = optax.join_schedules(
        schedules=[
            lambda step: lr_1,
            lambda step: lr_2,
            lambda step: lr_3,
        ],
        boundaries=[
            int((20000 - current_epoch) * num_steps_per_epoch),
            int((30000 - current_epoch) * num_steps_per_epoch),
        ],
    )

    adam = optax.adam(learning_rate=schedule, nesterov=False)

    state = adam.init(params_list)

    return state, adam, schedule


# ---------------------------------------------


def load_or_generate_params_list(architectures, key, tm_full_list):
    try:
        Error
        current_epoch = np.loadtxt("./results/current_epoch")
        params_list = utils_data_and_folders.load_params_list(0)
        with open("./results/losses.pkl", "rb") as fle:
            losses = pickle.load(fle)
        _, adam, schedule = adam_init_state(params_list, current_epoch)
        momenta_list = utils_data_and_folders.load_state_list(0)
        print("\nPreviously trained parameters found\n")

    except:
        current_epoch = 0
        params_list = generate_params_list(architectures, key, tm_full_list)
        momenta_list, adam, schedule = adam_init_state(params_list, current_epoch)

        losses = {"train": [None], "test": [None]}
        print("\nNew parameters generated")

    time.sleep(0.4)
    return params_list, momenta_list, adam, schedule, losses, current_epoch
