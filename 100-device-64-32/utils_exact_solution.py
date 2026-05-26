import os
from pdb import set_trace as st
import numpy as np
import jax.numpy as jnp


def compute_exact_solution(node_coordinates):

    scaling_nn = 300
    x = node_coordinates[:, 0] / scaling_nn
    y = node_coordinates[:, 1] / scaling_nn
    z = node_coordinates[:, 2] / scaling_nn
    a, b, c = np.sqrt(2**2 + 3**2), 2, 3
    return jnp.exp(a * x) * jnp.cos(b * y) * jnp.cos(c * z)
