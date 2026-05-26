""" """

import os
import pdb
from pdb import set_trace as st
import pickle
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np

import utils_data_and_folders
import utils_integral_geometry
import utils_nn

dprint = jax.debug.print


# min max utils: --------------------------------------------------


def update_mins_maxes_all_over_epochs(
    params_list,
    point_coordinates_train,
    mins_all_over_epochs,
    maxes_all_over_epochs,
):
    mins_all, maxes_all = compute_input_max_layer(params_list, point_coordinates_train)
    for name in mins_all_over_epochs.keys():
        mins_all_over_epochs[name] = np.vstack(
            (mins_all_over_epochs[name], mins_all[name])
        )
        maxes_all_over_epochs[name] = np.vstack(
            (maxes_all_over_epochs[name], maxes_all[name])
        )
    return mins_all_over_epochs, maxes_all_over_epochs


def compute_min_max_single_tm(
    params_list, nn_list, name, point_coordinates  # reductions, tm
):
    # reduction = reductions[0]  ###

    _, mins, maxes = getattr(nn_list.der_0, name)(params_list[name], point_coordinates)
    return mins, maxes


def compute_input_max_layer(params_list, point_coordinates):

    nn_list = define_nn_forwards_and_derivatives(return_min_max=True)

    nn_names = [key for key in params_list if key != "coeffs"]
    mins_all = dict.fromkeys(nn_names)
    maxes_all = dict.fromkeys(nn_names)

    for name in nn_names:
        mins_all[name], maxes_all[name] = compute_min_max_single_tm(
            params_list, nn_list, name, point_coordinates
        )

    return mins_all, maxes_all


# boundary data --------------------------------


def generate_boundary_data():
    r"""
    The boudary conditions are implemented in the following form:

        $a \circ u + b \circ \nabla u n = bc_val$

    where $\circ$ is the Hadamard product.

    It is up to the user to define properly the vectors a and b.
    """

    with open("./data/tags", "rb") as fle:
        tags = pickle.load(fle)

    tags = tree_map(lambda tag: tag.astype(int), tags)

    point_coordinates = np.loadtxt("./results/centroids_coordinates")
    normals = np.loadtxt("./results/face_normals")
    areas = np.loadtxt("./results/face_areas")
    idx_inner = np.loadtxt("./results/idx_inner").astype(int)

    n_pt = point_coordinates.shape[0]

    mask_inner_points = np.zeros(n_pt)
    mask_inner_points[idx_inner] = 1

    bc_type = np.zeros((n_pt, 2))

    # define vectors a:
    # this is for u

    # define vectors b:
    # this is for u dot n
    bc_type[tags["in"], 1] = 1
    bc_type[tags["out"], 1] = 1
    bc_type[tags["wall"], 1] = 1

    bc_vals = np.zeros((n_pt))
    bc_vals[tags["in"]] = -1
    bc_vals[tags["out"]] = 1

    return (
        point_coordinates,
        jnp.array(mask_inner_points),
        bc_vals,
        bc_type,
        normals,
        areas,
        tags,
    )


def compute_weights_loss(areas, tags):
    """ """
    weights_loss = np.ones(areas.shape[0])
    weights_loss[tags["in"]] = 10
    weights_loss[tags["out"]] = 10
    weights_loss[tags["wall"]] = 1
    return weights_loss


# model ----------------------------------------------------


def define_nn_forwards_and_derivatives(return_min_max=False):
    """ """

    @jax.jit
    def common(params, point_coordinates):

        z = point_coordinates

        for layer in params[:-1]:
            # z = utils_nn.activation_exp(z @ layer["W"].T + layer["b"])
            z = utils_nn.activation_smooth(z @ layer["W"].T + layer["b"])

        z = z @ params[-1]["W"].T + params[-1]["b"]

        return z

    @jax.jit
    def common_return_min_max(params, point_coordinates):
        """ """
        z = point_coordinates

        mins = jnp.zeros(len(params) - 1)
        maxes = jnp.zeros(len(params) - 1)

        for idx_layer, layer in enumerate(params[:-1]):
            # z = utils_nn.activation_exp(z @ layer["W"].T + layer["b"])
            z = utils_nn.activation_smooth(z @ layer["W"].T + layer["b"])

            length = jnp.sqrt(z**2).astype(jnp.float32)

            mins = mins.at[idx_layer].set(np.min(length))
            maxes = maxes.at[idx_layer].set(np.max(length))

        z = z @ params[-1]["W"].T + params[-1]["b"]

        return z, mins, maxes

    if not return_min_max:

        def chi_0_forward(params, z):
            """ """
            return common(params, z)

    else:

        def chi_0_forward(params, z):
            """ """
            return common_return_min_max(params, z)

    nn_list = utils_nn.make_list_chi_phi_and_der(
        chi_0_forward,
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
    point_coordinates_train,
    mask_inner_points_train,
    bc_vals_train,
    bc_type_train,
    normals_train,
    weights_loss_train,
    tags_list_train,
    point_coordinates_test,
    mask_inner_points_test,
    bc_vals_test,
    bc_type_test,
    normals_test,
    weights_loss_test,
    tags_list_test,
    test_every,
    losses,
    der_displ_fcn,
):
    """ """
    num_data = point_coordinates_train.shape[0]
    num_batches = np.maximum(int(np.floor(num_data / batch_size)), 1)

    momentums_list = utils_nn.adam_init(params_list)

    mins_all, maxes_all = compute_input_max_layer(params_list, point_coordinates_train)

    mins_all_over_epochs = mins_all
    maxes_all_over_epochs = maxes_all

    for epoch in range(epochs):
        idx_data = np.arange(num_data)
        np.random.shuffle(idx_data)
        indices_batches = np.split(
            idx_data[: num_batches * batch_size], num_batches
        ) + [idx_data[num_batches * batch_size :]]

        for indices_batch in indices_batches:

            grads_list = utils_nn.compute_grads(
                utils_nn.loss_l2,
                params_list,
                nn_list,
                point_coordinates_train[indices_batch],
                mask_inner_points_train[indices_batch],
                bc_vals_train[indices_batch],
                bc_type_train[indices_batch],
                normals_train[indices_batch],
                weights_loss_train[indices_batch],
                der_displ_fcn,
            )

            if epoch <= 1000:
                lr = lr_1
            elif 1000 <= epoch < 5000:
                lr = lr_2
            else:
                lr = lr_3

            params_list, momentums_list = utils_nn.update_adam(
                params_list,
                grads_list,
                momentums_list,
                lr,
                lr_coeffs_scaling,
            )

        if epoch % test_every == 0:
            loss_train, loss_components_train = utils_nn.loss_l2(
                params_list,
                nn_list,
                point_coordinates_train,
                mask_inner_points_train,
                bc_vals_train,
                bc_type_train,
                normals_train,
                weights_loss_train,
                der_displ_fcn,
            )
            loss_test, loss_components_test = utils_nn.loss_l2(
                params_list,
                nn_list,
                point_coordinates_test,
                mask_inner_points_test,
                bc_vals_test,
                bc_type_test,
                normals_test,
                weights_loss_test,
                der_displ_fcn,
            )

            if losses["train"][0] is None:
                losses["train"] = np.real(np.array(loss_components_train))
                losses["test"] = np.real(np.array(loss_components_test))
            else:
                losses["train"] = np.vstack(
                    (losses["train"], np.atleast_1d(loss_components_train))
                )
                losses["test"] = np.vstack(
                    (losses["test"], np.atleast_1d(loss_components_test))
                )

            print(
                f"{epoch}, train: {np.real(loss_train):.4e}, test: {np.real(loss_test):.4e}, lr={lr}"
            )

            mins_all_over_epochs, maxes_all_over_epochs = (
                update_mins_maxes_all_over_epochs(
                    params_list,
                    point_coordinates_train,
                    mins_all_over_epochs,
                    maxes_all_over_epochs,
                )
            )
    return (params_list, losses, mins_all_over_epochs, maxes_all_over_epochs)


if __name__ == "__main__":

    os.system("clear")

    point_coordinates, mask_inner_points, bc_vals, bc_type, normals, areas, tags = (
        generate_boundary_data()
    )
    print(
        f"\nnumber of data = number of boundary + inner points = {point_coordinates.shape[0]}"
    )

    nn_list = define_nn_forwards_and_derivatives()

    default_architecture = [3, 16, 16, 16, 1]

    architectures = {
        "chi_0": (default_architecture),
    }

    seeds = [[0, 1, 2, 3, 4, 5, 6]]
    np.savetxt("./results/idx_seeds", np.arange(len(seeds)))

    weights_loss = compute_weights_loss(areas, tags)

    losses = {"train": [None], "test": [None]}
    epochs = 2001

    lr_1 = 5e-3
    lr_2 = 1e-3
    lr_3 = 2e-4
    lr_coeffs_scaling = 1e-6

    batch_size = 64

    np.savetxt("./results/num_epochs", np.array([epochs]))

    test_every = 10
    np.savetxt("./results/test_every", np.array([test_every]))

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

        params_list = utils_nn.generate_params_list(architectures, key)

        (
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
        ) = utils_nn.split_dataset(
            point_coordinates,
            mask_inner_points,
            bc_vals,
            bc_type,
            normals,
            areas,
            weights_loss,
            tags,
        )

        t_0 = time.time()
        (
            params_list,
            losses,
            mins_all_over_epochs,
            maxes_all_over_epochs,
        ) = train_adam(
            epochs,
            lr_1,
            lr_2,
            lr_3,
            lr_coeffs_scaling,
            batch_size,
            params_list,
            nn_list,
            point_coordinates_train,
            mask_inner_points_train,
            bc_vals_train,
            bc_type_train,
            normals_train,
            weights_loss_train,
            tags_list_train,
            point_coordinates_test,
            mask_inner_points_test,
            bc_vals_test,
            bc_type_test,
            normals_test,
            weights_loss_test,
            tags_list_test,
            test_every,
            losses,
            der_displ_fcn=None,
        )

        training_time = time.time() - t_0
        np.savetxt("./results/trainig_time", np.array([training_time]))

        utils_data_and_folders.save_params_list(params_list, idx_seed)

        utils_data_and_folders.save_mins_maxes(
            mins_all_over_epochs, maxes_all_over_epochs
        )

        with open(f"./results/losses_seed{idx_seed}.pkl", "wb") as fle:
            pickle.dump(losses, fle)

    print("\nDone!")
