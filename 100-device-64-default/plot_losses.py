import os
import pdb
from pdb import set_trace as st
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


fontsize = 28  # there is an issue with mathscr and fontsize
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=fontsize)
plt.rc("axes", labelsize=fontsize)
params = {
    "text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}\usepackage{mathrsfs}\usepackage{amsfonts}\usepackage{graphicx}\usepackage{relsize}\usepackage{tikz}"
}
plt.rcParams.update(params)
matplotlib.rcParams["axes.linewidth"] = 1.5


def plot_losses_epochs(losses, num_epochs_tot, idx_seeds, test_every):
    """ """

    x_plot = np.arange(0, num_epochs_tot, test_every) + 1
    fig, ax = plt.subplots(figsize=(12, 4))

    for idx_seed in idx_seeds:
        ax.plot(
            x_plot,
            losses["train"][idx_seed],
            color=[0.5, 0.5, 0.5],
            linestyle="--",
            linewidth=2,
            label="train",
        )
        ax.plot(
            x_plot,
            losses["test"][idx_seed],
            color=[0, 0, 0],
            linestyle="-",
            linewidth=2,
            label="test",
        )

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
    ax.grid(linestyle="--", linewidth=0.1, alpha=0.15, which="both")

    ax.set_ylabel(r"$\mathscr{L}$", fontsize=1.5 * fontsize)
    ax.set_xlabel("epoch")

    ax.legend()

    plt.savefig(
        "./results/losses_epochs_device_lapl.pdf",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    # Dummy plot for each method label (one per method)
    fig_legend, ax_legend = plt.subplots(figsize=(25, 10))

    ax_legend.plot([], [], color=[0, 0, 0], linestyle="--", label="train")
    ax_legend.plot([], [], color=[0, 0, 0], linestyle="-", label="test")

    ax_legend.legend(
        fontsize=fontsize,
        loc="lower center",
        ncol=12,
        bbox_to_anchor=(0.5, -0.65),
        frameon=False,
    )

    filename = "./results/losses_epochs_device_lapl_legend.pdf"
    fig_legend.savefig(filename, bbox_inches="tight")
    plt.close(fig_legend)

    os.system(f"pdfcrop --margins '0 -800 0 0' {filename} {filename}")
    os.system(f"pdfcrop {filename} {filename}")


# ------------------------------------------------------------------------------

with open("./results/losses.pkl", "rb") as fle:
    losses = pickle.load(fle)

num_epochs_tot = int(np.loadtxt("./results/num_epochs"))
idx_seeds = np.atleast_1d(np.loadtxt("./results/idx_seeds").astype(int))
test_every = np.loadtxt("./results/test_every").astype(int)

plot_losses_epochs(losses, num_epochs_tot, idx_seeds, test_every)

print("\nDone!")
