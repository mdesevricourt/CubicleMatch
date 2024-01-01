from operator import is_
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def generate_random_price_vector(
    length: int, total_budget: int, key: Any
) -> tuple[jax.Array, Any]:
    """Generate a random price vector.

    Args:
        n_cubicles (int): The number of cubicles.
        n_half_days (int): The number of half days.
        seed (int, optional): The seed. Defaults to 0.

    Returns:
        jnp.ndarray: The random price vector.
        new_key (int): The new key.
    """
    key, sub_key = jax.random.split(key)
    price_vector = jax.random.uniform(sub_key, (length,)) * total_budget
    return price_vector, key


def filter_out_tabu_neighbors(neighbors: jax.Array, tabu_list: jax.Array) -> jax.Array:
    """Return the non-tabu neighbors.

    Args:
        neighbors (jax.Array): The neighbors.
        tabu_list (jax.Array): The tabu list.

    Returns:
        jax.Array: The non-tabu neighbors.
    """
    # find the non-tabu neighbors
    neighbors_ls = jnp.split(neighbors, neighbors.shape[0], axis=0)
    non_tabu_neighbors = []

    for neighbor in neighbors_ls:
        if not jnp.any(jnp.all(neighbor == tabu_list, axis=1)):
            non_tabu_neighbors.append(neighbor)

    return jnp.squeeze(jnp.array(non_tabu_neighbors))


def find_neighbor_with_smallest_error(neighbors: jax.Array, clearing_errors: jax.Array):
    """Return the neighbor with the smallest error.

    Args:
        neighbors (jax.Array): The neighbors.
        clearing_errors (jax.Array): The clearing errors.

    Returns:
        jax.Array: The neighbor with the smallest error.
    """
    # find the neighbor with the smallest error
    index = jnp.argmin(clearing_errors)

    return neighbors[index, :], clearing_errors[index]
