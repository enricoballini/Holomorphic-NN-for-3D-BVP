import os
import pdb
from pdb import set_trace as st
import pickle


def save_params_list(params_list, seed, subscript=""):
    """ """
    for name, params in zip(params_list.keys(), params_list.values()):
        with open(
            f"results/params" + subscript + "_" + name + f"_seed{seed}_pinn.pkl", "wb"
        ) as fle:
            pickle.dump(params, fle)


def save_momenta_list(momenta_list, seed):
    """ """
    for name, momentum in zip(momenta_list.keys(), momenta_list.values()):
        with open(f"results/momenta_" + name + f"_seed{seed}_pinn.pkl", "wb") as fle:
            pickle.dump(momentum, fle)


def save_state_list(state_list, seed, subscript=""):
    """ """
    with open(f"results/state_list" + subscript + f"_seed{seed}_pinn.pkl", "wb") as fle:
        pickle.dump(state_list, fle)


def save_mins_maxes(mins, maxes):
    """ """
    with open(f"./results/mins_all_over_epochs_pinn.pkl", "wb") as fle:
        pickle.dump(mins, fle)

    with open(f"./results/maxes_all_over_epochs_pinn.pkl", "wb") as fle:
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
        if f"_seed{seed}_pinn" in fle_name:
            nn_name = fle_name[len("params_") : fle_name.index(f"_seed{seed}_pinn")]
            with open(os.path.join("./results", fle_name), "rb") as fle:
                params = pickle.load(fle)
                params_list[nn_name] = params

    return params_list


def load_momenta_list(seed):
    """ """
    momenta_list = {}
    files = sorted(
        [
            f
            for f in os.listdir("./results")
            if f.startswith("momenta_") and f.endswith(".pkl")
        ]
    )
    for fle_name in files:
        if f"_seed{seed}" in fle_name:
            nn_name = fle_name[len("momenta_") : fle_name.index(f"_seed{seed}")]
            with open(os.path.join("./results", fle_name), "rb") as fle:
                params = pickle.load(fle)
                momenta_list[nn_name] = params

    return momenta_list


def load_state_list(seed):
    """ """
    with open(f"results/state_list_seed{seed}_pinn.pkl", "rb") as fle:
        state_list = pickle.load(fle)
    return state_list
