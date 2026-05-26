""" """

import os
import pdb
from pdb import set_trace as st
import pickle
import time

import jax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_map
import optax
import numpy as np

import utils_data_and_folders_pinn
import utils_integral_geometry
import utils_mechanics_pinn
import utils_nn_pinn

import case_settings_pinn

dprint = jax.debug.print


def update_mins_maxes_all_over_epochs(
    params_list,
    tm_full_list,
    reductions,
    point_coordinates_train,
    mins_all_over_epochs,
    maxes_all_over_epochs,
):
    mins_all, maxes_all = case_settings_pinn.compute_input_max_layer(
        params_list, point_coordinates_train
    )
    for name in mins_all_over_epochs.keys():
        mins_all_over_epochs[name] = np.vstack(
            (mins_all_over_epochs[name], mins_all[name])
        )
        maxes_all_over_epochs[name] = np.vstack(
            (maxes_all_over_epochs[name], maxes_all[name])
        )
    return mins_all_over_epochs, maxes_all_over_epochs


def train_adam(
    idx_seed,
    epochs,
    lr_coeffs_scaling,
    batch_size,
    params_list,
    state_list,
    adam,
    schedule,
    nn_list,
    tm_list,
    tm_full_list,
    delta_tm,
    reductions,
    point_coordinates_train,
    mask_inner_pts_train,
    bc_vals_train,
    bc_type_train,
    normals_train,
    weights_loss_train,
    tags_list_train,
    point_coordinates_test,
    mask_inner_pts_test,
    bc_vals_test,
    bc_type_test,
    normals_test,
    weights_loss_test,
    tags_list_test,
    test_every,
    nu,
    G,
    nodes_int,
    weights_int,
    der_displ_fcn,
    losses,
    current_epoch,
):
    """ """

    def make_update_fn(
        adam,
        nn_list,
        tm_list,
        tm_full_list,
        delta_tm,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    ):
        @jax.jit
        def update_adam(
            params_list,
            state_adam,
            point_coordinates_train_batch,
            mask_inner_pts_train_batch,
            bc_vals_train_batch,
            bc_type_train_batch,
            normals_train_batch,
            weights_loss_train_batch,
        ):
            def loss_fn(p_list):
                val, _ = utils_nn_pinn.loss_l2(
                    p_list,
                    nn_list,
                    tm_list,
                    tm_full_list,
                    reductions,
                    point_coordinates_train_batch,
                    mask_inner_pts_train_batch,
                    bc_vals_train_batch,
                    bc_type_train_batch,
                    normals_train_batch,
                    weights_loss_train_batch,
                    nu,
                    G,
                    nodes_int,
                    weights_int,
                )
                return jnp.real(val)

            def grads_fn(p_list):
                loss, grads = utils_nn_pinn.compute_grads(
                    utils_nn_pinn.loss_l2,
                    p_list,
                    nn_list,
                    tm_list,
                    tm_full_list,
                    reductions,
                    point_coordinates_train_batch,
                    mask_inner_pts_train_batch,
                    bc_vals_train_batch,
                    bc_type_train_batch,
                    normals_train_batch,
                    weights_loss_train_batch,
                    nu,
                    G,
                    nodes_int,
                    weights_int,
                )
                return loss, grads

            loss, grads = grads_fn(params_list)

            grads = tree_map(jnp.conj, grads)

            direction, state_adam_new = adam.update(
                grads,
                state_adam,
                params_list,
                value=loss,
                grad=grads,
                value_fn=loss_fn,
            )

            dir_h_scaled = direction["coeffs"][0]["W_1"] * lr_coeffs_scaling
            direction["coeffs"][0]["W_1"] = dir_h_scaled

            params_list_new = optax.apply_updates(params_list, direction)

            h = params_list_new["coeffs"][0]["W_1"]
            params_list_new["coeffs"][0]["W_1"] = jnp.maximum(jnp.real(h), 1e-8)

            thr = 1e-5
            direction = tree_map(
                lambda leaf: (
                    jnp.clip(leaf.real, min=-thr, max=thr)
                    + 1j * jnp.clip(leaf.imag, min=-thr, max=thr)
                ),
                direction,
            )

            return params_list_new, state_adam_new

        return update_adam

    update_adam = make_update_fn(
        adam,
        nn_list,
        tm_list,
        tm_full_list,
        delta_tm,
        reductions,
        nu,
        G,
        nodes_int,
        weights_int,
    )

    num_data = point_coordinates_train.shape[0]
    num_batches = np.maximum(int(np.floor(num_data / batch_size)), 1)

    mins_all, maxes_all = case_settings_pinn.compute_input_max_layer(
        params_list, point_coordinates_train
    )
    mins_all_over_epochs = mins_all
    maxes_all_over_epochs = maxes_all

    for epoch in range(epochs):
        idx_data = np.arange(num_data)
        np.random.shuffle(idx_data)
        indices_batches = np.split(
            idx_data[: num_batches * batch_size], num_batches
        ) + [idx_data[num_batches * batch_size :]]

        for indices_batch in indices_batches:
            params_list, state_list = update_adam(
                params_list,
                state_list,
                point_coordinates_train[indices_batch],
                mask_inner_pts_train[indices_batch],
                bc_vals_train[indices_batch],
                bc_type_train[indices_batch],
                normals_train[indices_batch],
                weights_loss_train[indices_batch],
            )

        if epoch % test_every == 0:
            loss_train, loss_components_train = utils_nn_pinn.loss_l2(
                params_list,
                nn_list,
                tm_list,
                tm_full_list,
                reductions,
                point_coordinates_train,
                mask_inner_pts_train,
                bc_vals_train,
                bc_type_train,
                normals_train,
                weights_loss_train,
                nu,
                G,
                nodes_int,
                weights_int,
            )
            loss_test, loss_components_test = utils_nn_pinn.loss_l2(
                params_list,
                nn_list,
                tm_list,
                tm_full_list,
                reductions,
                point_coordinates_test,
                mask_inner_pts_test,
                bc_vals_test,
                bc_type_test,
                normals_test,
                weights_loss_test,
                nu,
                G,
                nodes_int,
                weights_int,
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

            step = state_list[0].count
            lr = schedule(step)

            print(
                f"{epoch}, train: {np.real(loss_train):.4e}, test: {np.real(loss_test):.4e}, lr={lr:.3e}"
            )

            mins_all_over_epochs, maxes_all_over_epochs = (
                update_mins_maxes_all_over_epochs(
                    params_list,
                    tm_full_list,
                    reductions,
                    point_coordinates_train,
                    mins_all_over_epochs,
                    maxes_all_over_epochs,
                )
            )

        if epoch % 1000 == 0:
            utils_data_and_folders_pinn.save_params_list(params_list, 0)
            utils_data_and_folders_pinn.save_state_list(state_list, 0)

            utils_data_and_folders_pinn.save_mins_maxes(
                mins_all_over_epochs, maxes_all_over_epochs
            )

            with open(f"./results/losses_seed{idx_seed}_pinn.pkl", "wb") as fle:
                pickle.dump(losses, fle)

            np.savetxt("./results/current_epoch", np.array([epoch]))

    return (
        params_list,
        losses,
        mins_all_over_epochs,
        maxes_all_over_epochs,
    )


if __name__ == "__main__":

    nu = np.loadtxt("./data/nu")
    G = np.loadtxt("./data/G")

    point_coordinates, mask_inner_pts, bc_vals, bc_type, normals, areas, tags = (
        case_settings_pinn.generate_boundary_data()
    )
    print(
        f"\nnumber of data = number of boundary points = {point_coordinates.shape[0]}"
    )

    M = int(np.loadtxt("./data/M"))
    bias_rotation = np.loadtxt("./data/bias_rotation")

    tm_list, delta_tm = utils_nn_pinn.define_tm(M, bias_rotation)

    reductions = utils_nn_pinn.define_reductions()

    nn_list = case_settings_pinn.define_nn_forwards_and_derivatives()

    default_architecture = np.loadtxt("./data/default_architecture_pinn").astype(int)
    default_split = np.loadtxt("./data/default_split").astype(int)

    architectures = {
        "chi": (default_architecture, default_split),
    }

    seeds = np.loadtxt("./data/seeds", ndmin=2).astype(int)

    weights_loss = case_settings_pinn.compute_weights_loss(areas, tags)

    epochs_adam = 40001

    lr_1_adam = 1e-3
    lr_2_adam = 2e-4
    lr_3_adam = 5e-5

    lr_coeffs_scaling = 0.1

    np.savetxt("./data/lr_1_adam", np.array([lr_1_adam]))
    np.savetxt("./data/lr_2_adam", np.array([lr_2_adam]))
    np.savetxt("./data/lr_3_adam", np.array([lr_3_adam]))
    np.savetxt("./data/lr_coeffs_scaling", np.array([lr_coeffs_scaling]))

    batch_size = 128

    np.savetxt("./results/num_epochs_adam", np.array([epochs_adam]))

    num_steps_per_epoch = int(point_coordinates.shape[0] / batch_size)
    np.savetxt("./data/num_steps_per_epoch", np.array([num_steps_per_epoch]))

    test_every = 100
    np.savetxt("./results/test_every", np.array([test_every]))

    nodes_int = np.loadtxt("./data/nodes_int")
    weights_int = np.loadtxt("./data/weights_int")

    tm_full_list = utils_integral_geometry.make_tm_full_list(tm_list, nodes_int)

    der_displ_fcn = utils_mechanics_pinn.define_strain_from_displ()

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

        params_list, state_list, adam, schedule, losses, current_epoch = (
            utils_nn_pinn.load_or_generate_params_list(architectures, key, tm_full_list)
        )

        (
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
        ) = utils_nn_pinn.split_dataset(
            point_coordinates,
            mask_inner_pts,
            bc_vals,
            bc_type,
            normals,
            areas,
            weights_loss,
            tags,
        )

        print("\nTraining using ADAM:")
        t_0 = time.time()

        (
            params_list,
            losses,
            mins_all_over_epochs,
            maxes_all_over_epochs,
        ) = train_adam(
            idx_seed,
            epochs_adam,
            lr_coeffs_scaling,
            batch_size,
            params_list,
            state_list,
            adam,
            schedule,
            nn_list,
            tm_list,
            tm_full_list,
            delta_tm,
            reductions,
            point_coordinates_train,
            mask_inner_pts_train,
            bc_vals_train,
            bc_type_train,
            normals_train,
            weights_loss_train,
            tags_list_train,
            point_coordinates_test,
            mask_inner_pts_test,
            bc_vals_test,
            bc_type_test,
            normals_test,
            weights_loss_test,
            tags_list_test,
            test_every,
            nu,
            G,
            nodes_int,
            weights_int,
            der_displ_fcn,
            losses,
            current_epoch,
        )

        training_time = time.time() - t_0
        np.savetxt("./results/trainig_time_pinn", np.array([training_time]))

        utils_data_and_folders_pinn.save_params_list(params_list, idx_seed)

        utils_data_and_folders_pinn.save_mins_maxes(
            mins_all_over_epochs, maxes_all_over_epochs
        )

        with open(f"./results/losses_seed{idx_seed}_pinn.pkl", "wb") as fle:
            pickle.dump(losses, fle)

    np.set_printoptions(threshold=99999)
    print("\n      h(t1),         h(t2),         h(t3),         tm")
    h_and_angle = np.hstack(
        (
            params_list["coeffs"][0]["W_1"],
            tm_full_list.reshape(tm_full_list.shape[0], -1) * 180 / np.pi,
        )
    )
    print(h_and_angle)

    with open("results/h_and_angle.txt", "w") as f:
        f.write("\n      h(t1),         h(t2),         h(t3),         tm\n")
        f.write(np.array2string(h_and_angle))

    print("\nDone!")
