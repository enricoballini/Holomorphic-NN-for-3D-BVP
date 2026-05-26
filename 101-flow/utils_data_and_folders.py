import os
import pdb
from pdb import set_trace as st
import pickle


def setup_directories():
    """ """
    # os.system("rm -r ./data")
    # os.system("rm -r ./results")

    os.system("mkdir ./data")
    os.system("mkdir ./results")


def save_params_list(params_list, seed):
    """ """
    for name, params in zip(params_list.keys(), params_list.values()):
        with open(f"results/params_" + name + f"_seed{seed}.pkl", "wb") as fle:
            pickle.dump(params, fle)


def save_mins_maxes(mins, maxes):
    """ """
    with open(f"./results/mins_all_over_epochs.pkl", "wb") as fle:
        pickle.dump(mins, fle)

    with open(f"./results/maxes_all_over_epochs.pkl", "wb") as fle:
        pickle.dump(maxes, fle)


def load_params_list(seed):
    """ """
    params_list = {}
    files = sorted(
        [
            f
            for f in os.listdir("./results")
            if f.startswith("params_") and f.endswith(".pkl")
        ]
    )
    for fle_name in files:
        if f"_seed{seed}" in fle_name:
            nn_name = fle_name[len("params_") : fle_name.index(f"_seed{seed}")]
            with open(os.path.join("./results", fle_name), "rb") as fle:
                params = pickle.load(fle)
                params_list[nn_name] = params

    return params_list
