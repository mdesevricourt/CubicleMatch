"""Computes market level quantities from prices and primitives."""


from typing import Callable

import jax
import jax.numpy as jnp


def compute_aggregate_demand(demand_vector: jax.Array) -> jax.Array:
    """Return the aggregate demand, aka the bundle that maximizes utility subject to budget constraint.

    Args:
        demand_vector (jax.Array): A vector of demands.
    Returns:
        aggregate_demand (jax.Array): The aggregate demand.
    """
    # compute the aggregate demand
    aggregate_demand = jnp.sum(demand_vector, axis=0)
    return aggregate_demand


def compute_excess_demand(
    aggregate_demand: jax.Array,
    supply: jax.Array,
) -> jax.Array:
    """Return the excess demand, aka the aggregate demand minus the total supply.

    Args:
        aggregate_demand (jax.Array): The aggregate demand.
        supply (jax.Array): The total supply.
    Returns:
        excess_demand (jax.Array): The excess demand.
    """
    excess_demand = aggregate_demand - supply
    return excess_demand


def modified_excess_demand(
    excess_demand: jax.Array, price_vector: jax.Array
) -> jax.Array:
    """Return the modified excess demand, aka the excess demand modified so that it is non-negative.

    Args:
        excess_demand (jax.Array): The excess demand.
        price_vector (jax.Array): The price vector.
    Returns:
        z (jax.Array): The excess demand modified so that it is non-negative.
    """
    z = jnp.where(price_vector != 0, excess_demand, jnp.max(excess_demand, 0))
    return z


def compute_clearing_error(
    z: jax.Array,
):
    """Return the clearing error, aka the sum of the excess demand squared.

    Args:
       z (jax.Array): The modified excess demand.
    Returns:
        alpha (jax.Array): The clearing error.
    """

    return jnp.sum(z**2)


def compute_agg_quantities(
    price_vector: jax.Array,
    compute_demand_vector: Callable,
    supply: jax.Array,
):
    """Return the aggregate quantities.

    # Parameters
    price_vector (jax.Array): The price vector.
    bundles (jax.Array): The available bundles.
    compute_demand_vector (Callable): A function that computes the demand vector from the available bundles and their respective prices.
    supply (jax.Array): The total supply.

    # Returns
    res (dict): A dictionary containing the aggregate quantities.
    """

    demand, excess_budgets = compute_demand_vector(price_vector)
    agg_demand = compute_aggregate_demand(demand)
    excess_demand_vec = compute_excess_demand(agg_demand, supply)
    z = modified_excess_demand(excess_demand_vec, price_vector)
    alpha = compute_clearing_error(z)
    number_excess_demands = jnp.sum(excess_demand_vec > 0)

    res = {
        "demand": demand,
        "excess_budgets": excess_budgets,
        "agg_demand": agg_demand,
        "excess_demand_vec": excess_demand_vec,
        "z": z,
        "clearing_error": alpha,
        "number_excess_demand": number_excess_demands,
    }

    return res


compute_agg_quantities_vec = jax.vmap(
    compute_agg_quantities, in_axes=(0, None, None, None, None)
)
