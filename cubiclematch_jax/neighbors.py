from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from cubiclematch_jax.demand import find_agent_demand

Market_Function = Callable[[jax.Array], dict[str, jax.Array]]


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
    aggregate_quantities_price: Market_Function,
):
    """Find individual adjustment neighbors neighbor based on the excess demand. For cubicle-half-day that are over-supplied,
    we set the price to 0. For cubicle-half-day that are over-demanded, we increase the price until at least one more agent
    demands the item.

    Args:
        price_vector (jax.Array): The price vector.
        step_size (jax.Array): The step size.
        z (jax.Array): The modified excess demand.
        excess_budgets (jax.Array): The excess budgets.
        aggregate_quantities_price (Market_Function): Function that computes the aggregate quantities from prices
    Returns:
        neighbors (jax.Array): The neighbors.
        neigbor_types (list[str]): The type of each neighbor.

    """

    neighbors_ls = []
    neigbor_types = []

    # individual adjustment neighbor
    for (
        i,
        d_i,
    ) in enumerate(excess_demand):
        if d_i == 0:
            continue

        p_neighbor = price_vector
        if d_i < 0:
            # if there is excess supply for i, decrease p[i] until at least one more agent demands the item
            neigbor_types.append("IA neighbor (excess supply)")
            p_neighbor = price_vector.at[i].set(0)

        current_excess_budgets = excess_budgets.copy()
        while d_i > 0:
            neigbor_types.append("IA neighbor (excess demand)")
            p_neighbor = price_vector.at[i].add(min(current_excess_budgets) + 1)
            res = aggregate_quantities_price(p_neighbor)
            agents_demanding_i = find_agent_demand(i, res["demand"])
            current_excess_budgets = res["excess_budgets"][agents_demanding_i]
            d_i = res["excess_demand_vec"][i]

        neighbors_ls.append(p_neighbor)
    neighbors = jnp.array(neighbors_ls)

    return neighbors, neigbor_types
