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


def add_legend(handles, labels, ax, **kwargs):
    handles = np.array(handles)
    labels = np.array(labels)
    idx_keep = np.array([0, 1, 4, 5, 8, 9])
    handles = handles[idx_keep]
    labels = labels[idx_keep]

    ax.legend(
        handles,
        labels,
        **kwargs,
    )
    return ax


def plot_losses_epochs(
    idx_seeds,
    test_every,
    losses_names,
    colors_train,
    colors_test,
    colors_seed,
    markers_seed,
):
    """ """

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for idx_seed in idx_seeds:
        with open(f"./results/losses_seed{idx_seed}.pkl", "rb") as fle:
            losses = pickle.load(fle)

        x_plot = np.arange(0, losses["train"].shape[0] * test_every, test_every) + 1
        num_losses = losses["train"].shape[1]

        for idx_loss in range(num_losses):
            ax.plot(
                x_plot,
                losses["train"][:, idx_loss],
                # color=colors_train[idx_loss],
                color=colors_seed[idx_seed],
                linestyle="--",
                linewidth=2,
                # label=losses_names[idx_loss],
                label=f"Seed {idx_seed}, train",
                marker=markers_seed[idx_seed],
            )
            ax.plot(
                x_plot,
                losses["test"][:, idx_loss],
                # color=colors_test[idx_loss],
                color=colors_seed[idx_seed],
                linestyle="-",
                linewidth=2,
                # label=losses_names[idx_loss] + f", seed {idx_seed}",
                label=f"Seed {idx_seed}, test",
                marker=markers_seed[idx_seed],
            )

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=8))
    ax.grid(linestyle="--", linewidth=0.1, alpha=0.15, which="both")
    ax.set_ylabel(r"$\mathscr{L}$", fontsize=1.5 * fontsize)
    ax.set_xlabel("epoch")

    handles, labels = ax.get_legend_handles_labels()
    ax = add_legend(handles, labels, ax, fontsize=fontsize)
    plt.savefig(
        "./results/losses_epochs_shear.pdf",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    # OPTION 1 ---------------
    # # Dummy plot for each method label (one per method)
    # fig_legend, ax_legend = plt.subplots(figsize=(25, 10))

    # ax_legend.plot([], [], color=[0, 0, 0], linestyle="--", label="train")
    # ax_legend.plot([], [], color=[0, 0, 0], linestyle="-", label="test")

    # ax_legend.legend(
    #     fontsize=fontsize,
    #     loc="lower center",
    #     ncol=12,
    #     bbox_to_anchor=(0.5, -0.65),
    #     frameon=False,
    # )

    # filename = "./results/losses_epochs_legend.pdf"
    # fig_legend.savefig(filename, bbox_inches="tight")
    # plt.close(fig_legend)

    # os.system(f"pdfcrop --margins '0 -800 0 0' {filename} {filename}")
    # os.system(f"pdfcrop {filename} {filename}")

    # OPTION 2 ---------------
    fig_legend, ax_legend = plt.subplots(figsize=(25, 10))

    ax_legend = add_legend(
        handles,
        labels,
        ax_legend,
        fontsize=fontsize,
        loc="lower center",
        ncol=1,
        bbox_to_anchor=(-1, -0.65),
    )

    filename = "./results/losses_epochs_shear_legend.pdf"
    fig_legend.savefig(filename, bbox_inches="tight")
    plt.close(fig_legend)

    os.system(f"pdfcrop --margins '0 -430 -1500 0' {filename} {filename}")
    os.system(f"pdfcrop {filename} {filename}")


# ------------------------------------------------------------------------------
idx_seeds = np.atleast_1d(np.loadtxt("./results/idx_seeds").astype(int))


test_every = np.loadtxt("./results/test_every").astype(int)

losses_names = [r"$l_2^2$", r"$\log$", r"reg"]
colors_train = [[0.3, 0.3, 0.3], [0.8, 1, 0], [0.8, 0, 0.8]]
colors_test = [[0, 0, 0], [0.5, 0.7, 0], [0.5, 0, 0.5]]
colors_seed = [[0, 0, 0], [0.5, 0.7, 0], [1, 0.8, 0]]
markers_seed = ["o", "x", "s"]
plot_losses_epochs(
    idx_seeds,
    test_every,
    losses_names,
    colors_train,
    colors_test,
    colors_seed,
    markers_seed,
)

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
