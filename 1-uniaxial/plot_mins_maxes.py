import os
from pdb import set_trace as st
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


fontsize = 28
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=fontsize)
params = {
    "text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}\usepackage{mathrsfs}\usepackage{amsfonts}"
}
plt.rcParams.update(params)
matplotlib.rcParams["axes.linewidth"] = 1.5


def print_min_max_weights():
    """TODO: move this function in an appropriate module"""
    import utils_data_and_folders
    from jax.tree_util import tree_map, tree_flatten

    seed = 0
    params_list = utils_data_and_folders.load_params_list(seed)

    mins_weights = tree_map(
        lambda p: np.min(np.real(p)) if p.size > 0 else 0, params_list
    )
    maxes_weights = tree_map(
        lambda p: np.max(np.real(p)) if p.size > 0 else 0, params_list
    )

    all_mins = np.array(tree_flatten(mins_weights)[0])
    all_maxes = np.array(tree_flatten(maxes_weights)[0])
    print("\nall_mins: ", all_mins)
    print("\nall_maxes: ", all_maxes)


def plot_mins_maxes_epochs(
    mins_all_over_epochs,
    maxes_all_over_epochs,
    test_every,
    nn_names,
    colors_mins,
    colors_maxes,
    linestyle_mins,
    linestyle_maxes,
):
    """ """

    x_plot = (
        np.arange(0, mins_all_over_epochs["chi"].shape[0] * test_every, test_every) + 1
    )
    fig, ax = plt.subplots(figsize=(16, 8))

    layer_markers = [
        "v",
        "",
        ".",
        "^",
        ">",
        "<",
        ",",
        "o",
        "1",
        "2",
        "3",
        "4",
        "s",
        "P",
    ]  # hardocded

    for ii, name in enumerate(mins_all_over_epochs):
        for idx_layer in range(mins_all_over_epochs[name].shape[1]):
            ax.plot(
                x_plot,
                mins_all_over_epochs[name][:, idx_layer],
                color=colors_mins[name],
                linestyle=linestyle_mins[name],
                linewidth=0.8,
                label=nn_names[ii] + f" layer {idx_layer}",
                marker=layer_markers[idx_layer],
                markersize=5,
            )

            ax.plot(
                x_plot,
                maxes_all_over_epochs[name][:, idx_layer],
                color=colors_maxes[name],
                linestyle=linestyle_maxes[name],
                linewidth=2.5,
                label=nn_names[ii] + f" layer {idx_layer}",
                marker=layer_markers[idx_layer],
                markersize=5,
            )

    ax.set_ylabel(r"$|\rho(z)|$")
    ax.set_xlabel(r"epoch")
    ax.set_yscale("log")
    ax.grid(linestyle="--", linewidth=0.1, alpha=0.15, which="both")

    ax.legend()

    plt.savefig(
        "./results/mins_maxes_epochs.pdf",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
    )


if __name__ == "__main__":

    with open("./results/mins_all_over_epochs.pkl", "rb") as fle:
        mins_all_over_epochs = pickle.load(fle)

    with open("./results/maxes_all_over_epochs.pkl", "rb") as fle:
        maxes_all_over_epochs = pickle.load(fle)

    test_every = np.loadtxt("./results/test_every").astype(int)

    nn_names = [
        r"$\chi_0$",
        r"$\phi_0$",
        r"$\chi_1$",
        r"$\phi_1$",
        r"$\chi_2$",
        r"$\phi_2$",
    ]
    colors_mins = {
        "chi": [0.2, 0.2, 0.2],
        "phi_0": [0.3, 0.3, 0.7],
        "phi_1": [0.5, 0.7, 0],
        "phi_2": [0.8, 0.7, 0],
    }
    colors_maxes = {
        "chi": [0.2, 0.2, 0.2],
        "phi_0": [0.3, 0.3, 0.7],
        "phi_1": [0.5, 0.7, 0],
        "phi_2": [0.9, 0.7, 0],
    }
    linestyle_mins = {
        "chi": "--",
        "phi_0": "--",
        "phi_1": "--",
        "phi_2": "--",
    }
    linestyle_maxes = {
        "chi": "-",
        "phi_0": "-",
        "phi_1": "-",
        "phi_2": "-",
    }

    plot_mins_maxes_epochs(
        mins_all_over_epochs,
        maxes_all_over_epochs,
        test_every,
        nn_names,
        colors_mins,
        colors_maxes,
        linestyle_mins,
        linestyle_maxes,
    )

    print_min_max_weights()

    print("\nDone!")

    # [
    #     [0, 0, 0],
    #     [1, 0.8, 0],
    #     [0.5, 0, 0.5],
    #     [0.5, 0.7, 0],
    #     [1, 0.8, 0],
    #     [0.5, 0, 0.5],
    #     [0.5, 0.7, 0],
    # ]
