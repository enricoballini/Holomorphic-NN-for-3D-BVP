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
import case_settings

dprint = jax.debug.print


def train_adam(
    epochs,
    lr_1,
    lr_2,
    lr_3,
    lr_coeffs_scaling,
    batch_size,
    params_list,
    nn_list,
    tm_grid,
    tm_full_list,
    delta_tm,
    reductions,
    point_coordinates_train,
    bc_vals_train,
    bc_type_train,
    normals_train,
    weights_loss_train,
    tags_list_train,
    point_coordinates_test,
    bc_vals_test,
    bc_type_test,
    normals_test,
    weights_loss_test,
    tags_list_test,
    test_every,
    nodes_int,
    weights_int,
    der_displ_fcn,
):
    """ """
    num_data = point_coordinates_train.shape[0]
    num_batches = np.maximum(int(np.floor(num_data / batch_size)), 1)

    momentums_list = utils_nn.adam_init(params_list)

    losses_train = []
    losses_test = []

    mask_helper = None

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
                tm_grid,
                tm_full_list,
                delta_tm,
                reductions,
                point_coordinates[indices_batch],
                bc_vals[indices_batch],
                bc_type[indices_batch],
                normals[indices_batch],
                weights_loss[indices_batch],
                mask_helper,
                nodes_int,
                weights_int,
                der_displ_fcn,
            )

            if epoch <= 1000:
                lr = lr_1
            elif 1000 <= epoch < 2000:
                lr = lr_2
            else:
                lr = lr_3

            params_list, momentums_list = utils_nn.update_adam(
                params_list, grads_list, momentums_list, lr, lr_coeffs_scaling
            )

        if epoch % test_every == 0:
            loss_train, loss_components_train = utils_nn.loss_l2(
                params_list,
                nn_list,
                tm_grid,
                tm_full_list,
                delta_tm,
                reductions,
                point_coordinates_train,
                bc_vals_train,
                bc_type_train,
                normals_train,
                weights_loss_train,
                mask_helper,
                nodes_int,
                weights_int,
                der_displ_fcn,
            )
            loss_test, loss_components_test = utils_nn.loss_l2(
                params_list,
                nn_list,
                tm_grid,
                tm_full_list,
                delta_tm,
                reductions,
                point_coordinates_test,
                bc_vals_test,
                bc_type_test,
                normals_test,
                weights_loss_test,
                mask_helper,
                nodes_int,
                weights_int,
                der_displ_fcn,
            )
            losses_train.append(loss_components_train)
            losses_test.append(loss_components_test)

            print(
                f"{epoch}, train: {np.real(loss_train):.4e}, test: {np.real(loss_test):.4e}, lr={lr}"
            )
    return params_list, losses_train, losses_test


if __name__ == "__main__":

    point_coordinates, bc_vals, bc_type, normals, areas, tags = (
        case_settings.generate_boundary_data()
    )
    print(
        f"\nnumber of data = number of boundary points = {point_coordinates.shape[0]}"
    )

    reductions = utils_nn.define_reductions()

    nn_list = case_settings.define_nn_forwards_and_derivatives()

    default_architecture = np.loadtxt("./data/default_architecture").astype(int)
    default_split = np.loadtxt("./data/default_split").astype(int)
    architectures = {
        "chi_0": (default_architecture, default_split),
    }

    weights_loss = case_settings.compute_weights_loss(areas, tags)

    losses = {"train": [], "test": []}
    epochs = 3001

    lr_1 = 1e-4
    lr_2 = 5e-5
    lr_3 = 1e-5
    lr_coeffs_scaling = 1e-6  # 1

    batch_size = 32

    np.savetxt("./results/num_epochs", np.array([epochs]))

    test_every = 10
    np.savetxt("./results/test_every", np.array([test_every]))

    M = np.loadtxt("./data/M").astype(int)
    bias_rotation = np.loadtxt("./data/bias_rotation")
    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")

    tm_grid, delta_tm = utils_nn.define_tm(M, bias_rotation)

    tm_full_list = utils_integral_geometry.make_tm_full_list(tm_grid, nodes_int)

    seeds = np.loadtxt("./data/seeds", ndmin=2).astype(int)

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

        params_list = utils_nn.generate_params_list(architectures, key, tm_full_list)

        (
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
        ) = utils_nn.split_dataset(
            point_coordinates, bc_vals, bc_type, normals, areas, weights_loss, tags
        )

        t_1 = time.time()
        params_list, losses_train, losses_test = train_adam(
            epochs,
            lr_1,
            lr_2,
            lr_3,
            lr_coeffs_scaling,
            batch_size,
            params_list,
            nn_list,
            tm_grid,
            tm_full_list,
            delta_tm,
            reductions,
            point_coordinates_train,
            bc_vals_train,
            bc_type_train,
            normals_train,
            weights_loss_train,
            tags_list_train,
            point_coordinates_test,
            bc_vals_test,
            bc_type_test,
            normals_test,
            weights_loss_test,
            tags_list_test,
            test_every,
            nodes_int,
            weights_int,
            der_displ_fcn=None,
        )
        np.savetxt("./results/training_time", np.array([time.time() - t_1]))

        utils_data_and_folders.save_params_list(params_list, idx_seed)
        losses["train"].append(losses_train)
        losses["test"].append(losses_test)

    with open("./results/losses.pkl", "wb") as fle:
        pickle.dump(losses, fle)

    np.set_printoptions(threshold=99999)
    print("\n      h(t1),            tm")

    h_and_angle = np.hstack(
        (
            params_list["coeffs"].reshape((-1, 1)),
            tm_full_list.reshape(tm_full_list.shape[0], -1) * 180 / np.pi,
        )
    )
    print(h_and_angle)

    with open("results/h_and_angle.txt", "w") as f:
        f.write("\n      h(t1),         h(t2),         h(t3),         tm\n")
        f.write(np.array2string(h_and_angle))

    print("\nDone!")
