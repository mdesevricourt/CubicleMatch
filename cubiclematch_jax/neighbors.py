from ast import Call
from turtle import clear
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from cubiclematch_jax.demand import clearing_error_primitives


def find_gradient_neighbor(
    price_vector: jax.Array, step_size: jax.Array, excess_demand: jax.Array
):
    """Find a neighbor based on the excess demand and a step size.

    Args:
        price_vector (jax.Array): The price vector.
        step_size (jax.Array): The step size.
        excess_demand (jax.Array): The excess demand.

    Returns:
        jax.Array: The neighbor.
    """
    neighbor = price_vector + step_size * excess_demand

    neighbor = jnp.where(neighbor >= 0, neighbor, 0)

    return neighbor


find_gradient_neighbors = jax.vmap(find_gradient_neighbor, in_axes=(None, 0, None))


def find_IA_neighbor(
    price_vector: jax.Array,
    excess_demand: jax.Array,
    excess_budgets: jax.Array,
    clearing_error: Callable[[jax.Array], jax.Array],
):
    """Find individual adjustment neighbors neighbor based on the excess demand and a step size.

    Args:
        price_vector (jax.Array): The price vector.
        step_size (jax.Array): The step size.
        z (jax.Array): The modified excess demand.
        excess_budgets (jax.Array): The excess budgets.

    Returns:
        jax.Array: array of individual adjustment neighbors.
    """
    # find indexes of elements of z that are not 0
    indexes = [i for i in range(len(excess_demand)) if excess_demand[i] != 0]
    neighbors = []
    neigbor_types = []

    # individual adjustment neighbor
    for i in indexes:
        # create a copy of p
        p_neighbor = np.array(price_vector.copy())
        d_i = excess_demand[i]
        if d_i > 0:
            # if there is excess demand for i, increase p[i] until at least one fewer agent demands the item

            current_excess_budgets = excess_budgets.copy()
            while d_i >= 0:
                neigbor_types.append(
                    "individual adjustment neighbor with excess demand"
                )
                p_neighbor[i] += min(current_excess_budgets) + 0.01
                alpha_neighbor, z_neighbor, d_neighbor = clearing_error(p_neighbor)
                d_i = d_neighbor[i]
        elif d_i < 0:
            # if there is excess supply for i, decrease p[i] until at least one more agent demands the item
            neigbor_types.append("individual adjustment neighbor with excess supply")
            p_neighbor[i] = 0
            # update price
            self.prices_vec = p_neighbor
            alpha_neighbor, z_neighbor, d_neighbor = self.clearing_error()
            d_i = d_neighbor[i]

    return (neighbors,)
