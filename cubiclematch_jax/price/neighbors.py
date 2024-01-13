from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from cubiclematch_jax.demand.demand import find_agent_demand
from cubiclematch_jax.price.aux_func import filter_out_tabu_neighbors

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


def find_IA_neighbors(
    price_vector: jax.Array,
    excess_demand: jax.Array,
    excess_budgets: jax.Array,
    compute_agg_quantities: Callable[[jax.Array], dict[str, jax.Array]],
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
        if d_i > 0:
            neigbor_types.append("IA neighbor (excess demand)")
            d_i_target = d_i

            while d_i >= d_i_target:
                p_neighbor = price_vector.at[i].set(
                    p_neighbor[i] + min(current_excess_budgets) + 0.001
                )
                res = compute_agg_quantities(p_neighbor)
                agents_demanding_i = find_agent_demand(i, res["demand"])
                current_excess_budgets = res["excess_budgets"][agents_demanding_i]
                d_i = res["excess_demand_vec"][i]

        neighbors_ls.append(p_neighbor)
    neighbors = jnp.array(neighbors_ls)

    return neighbors


# %%


from typing import Callable

import jax
import jax.numpy as jnp


def find_individual_adjustment_neighbors_jit(
    price_vector: jax.Array,
    excess_demand: jax.Array,
    excess_budgets: jax.Array,
    compute_agg_quantities: Callable[[jax.Array], dict[str, jax.Array]],
):
    def adjust_price(i, price_vector):
        p_neighbor = price_vector

        def decrease_price(p):
            return p.at[i].set(0), None

        def increase_price(p):
            d_i = excess_demand[i]
            d_i_target = d_i
            current_excess_budgets = excess_budgets

            def body_fn(carry):
                p, d_i = carry
                p = p.at[i].set(p[i] + jnp.min(current_excess_budgets) + 0.001)
                res = compute_agg_quantities(p)
                agents_demanding_i = find_agent_demand(i, res["demand"])
                current_excess_budgets = res["excess_budgets"][agents_demanding_i]
                d_i = res["excess_demand_vec"][i]
                return (p, d_i), None

            p, d_i = jax.lax.while_loop(
                lambda carry: carry[1] >= d_i_target, body_fn, (p, d_i)
            )
            return p, None

        return jax.lax.cond(
            excess_demand[i] < 0, decrease_price, increase_price, p_neighbor
        )

    neighbors = jax.vmap(adjust_price, in_axes=(0, None))(
        jnp.arange(excess_demand.size), price_vector
    )
    return neighbors


def find_all_neighbors(
    price_vector: jax.Array,
    excess_demand: jax.Array,
    excess_budgets: jax.Array,
    step_sizes: jax.Array,
    evaluate_prices: Callable[[jax.Array], dict[str, jax.Array]],
    tabu_list: jax.Array,
):
    """Find all neighbors.

    Args:
        price_vector (jax.Array): The price vector.
        step_size (jax.Array): The step size.
        excess_demand (jax.Array): The excess demand.
        excess_budgets (jax.Array): The excess budgets.
        aggregate_quantities_price (Market_Function): Function that computes the aggregate quantities from prices
        tabu_list (jax.Array): The tabu list.

    Returns:
        neighbors (jax.Array): The neighbors.
    """
    gradient_neighbors = find_gradient_neighbors(
        price_vector, step_sizes, excess_demand
    )

    IA_neighbors = find_IA_neighbors(
        price_vector, excess_demand, excess_budgets, evaluate_prices
    )

    neighbors = jnp.vstack((gradient_neighbors, IA_neighbors))
    # neigbor_types = ["gradient"] * len(gradient_neighbors) + ["IA"] * len(IA_neighbors)

    vmap_evaluate_prices = jax.vmap(evaluate_prices)

    agg_quantites = vmap_evaluate_prices(neighbors)
    neighbors = filter_out_tabu_neighbors(neighbors, agg_quantites, tabu_list)

    return neighbors
