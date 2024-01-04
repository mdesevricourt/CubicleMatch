from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cubiclematch_jax.market_level import compute_aggregate_quantities_vec


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


def is_excess_demand_in_tabu_list(
    excess_demand: jax.Array, tabu_list: jax.Array
) -> jax.Array:
    """Return whether the excess demand is in the tabu list.

    Args:
        excess_demand (jax.Array): The excess demand.
        tabu_list (jax.Array): The tabu list.

    Returns:
        jax.Array: Whether the neighbor is in the tabu list.
    """
    return jnp.any(jnp.all(excess_demand == tabu_list, axis=1))


are_excess_demands_in_tabu_list = jax.vmap(
    is_excess_demand_in_tabu_list, in_axes=(0, None)
)


def filter_out_tabu_neighbors(
    neighbors: jax.Array,
    agg_quantities_dict: dict[str, jax.Array],
    tabu_list: jax.Array,
) -> jax.Array:
    """Return the non-tabu neighbors.

    Args:
        neighbors (jax.Array): The neighbors.
        agg_quantities_dict (jax.Array): The dictionary of aggregate quantities for the neighbors.
        tabu_list (jax.Array): The tabu list.

    Returns:
        jax.Array: The non-tabu neighbors.
        agg_quantities_dict (jax.Array): The dictionary of aggregate quantities for the non-tabu neighbors.
    """
    excess_demands = agg_quantities_dict["excess_demand_vec"]
    in_tabu = are_excess_demands_in_tabu_list(excess_demands, tabu_list)
    non_tabu_neighbors = neighbors[~in_tabu]
    for k, v in agg_quantities_dict.items():
        agg_quantities_dict[k] = v[~in_tabu]

    return non_tabu_neighbors


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


def total_budget(budgets: jax.Array) -> jax.Array:
    """Return the total budget.

    Args:
        budgets (jax.Array): The budgets.

    Returns:
        jax.Array: The total budget.
    """
    return jnp.sum(budgets)


def sort_neighbors_by_clearing_error(
    neighbors: jax.Array, agg_quantities_dict: dict[str, jax.Array]
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Sort the neighbors by clearing error and, in case of ties, by number of excess demands.

    Args:
        neighbors (jax.Array): The neighbors.
        agg_quantities_dict (dict[str, jax.Array]): The dictionary of aggregate quantities for the neighbors.

    Returns:
        jax.Array: The sorted neighbors.
        agg_quantities_dict (dict[str, jax.Array]): The dictionary of aggregate quantities for the sorted neighbors.
    """
    # sort the neighbors by clearing error
    clearing_errors = agg_quantities_dict["alpha"]
    number_excess_demands = agg_quantities_dict["number_excess_demands"]
    indices = jnp.lexsort((number_excess_demands, clearing_errors))
    neighbors = neighbors[indices, :]
    for k, v in agg_quantities_dict.items():
        agg_quantities_dict[k] = v[indices]

    return neighbors, agg_quantities_dict


def ACE_iteration(
    price_vectors: jax.Array,
    bundles: jax.Array,
    budgets: jax.Array,
    U_tilde: jax.Array,
    supply: jax.Array,
    tabu_list: jax.Array,
):
    """Perform one iteration of the ACE algorithm.

    Args:
        price_vector (jax.Array): The price vector.
        bundles (jax.Array): The bundles.
        budgets (jax.Array): The budgets.
        U_tilde (jax.Array): The utility function.
        supply (jax.Array): The supply.
        step_sizes (jax.Array): The step sizes for the gradient neighbors.
        tabu_list (jax.Array): The tabu list.

    Returns:
        new_price_vector (jax.Array): The new price vector.
        new_clearing_error (jax.Array): The new clearing error.

    """

    # find neighbors
    agg_quantities = compute_aggregate_quantities_vec(
        price_vectors, budgets, U_tilde, bundles, supply
    )

    price_vectors, agg_quantities = filter_out_tabu_neighbors(
        price_vectors, agg_quantities, tabu_list
    )

    # sort the neighbors by clearing error
    price_vectors, agg_quantities = sort_neighbors_by_clearing_error(
        price_vectors, agg_quantities
    )

    # return the neighbor with the smallest error
    return (
        price_vectors[0, :],
        agg_quantities["alpha"][0],
        agg_quantities["number_excess_demands"][0],
    )
