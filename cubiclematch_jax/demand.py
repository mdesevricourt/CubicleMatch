import jax
import jax.numpy as jnp

from cubiclematch_jax.preferences import find_best_bundle
from cubiclematch_jax.price import affordable_bundles, price_bundles


def individual_demand(
    budget: jax.Array,
    U: jax.Array,
    u_cubicle: jax.Array,
    bundles: jax.Array,
    bundle_prices: jax.Array,
):
    """Return the demand of an individual, aka the bundle that maximizes utility subject to budget constraint.

    Args:
        budget (float): The budget.
        U (jax.Array): A utility matrix, that is a 2D square triangular numpy array of shape (number_of_half_days, number_of_half_days).
        u_cubicle (jax.Array): A utility vector of shape (number_of_cubicles,).
        bundles (jax.Array): A list of bundles.
        prices (jax.Array): A price vector of shape (number_of_half_days * number_of_cubicles,).
    Returns:
        jax.Array: The demand of an individual.
    """

    # find the affordable bundles
    affordable_bundles_ls = affordable_bundles(bundles, bundle_prices, budget)
    # compute the total utility of each affordable bundle
    best_affordable_bundle = find_best_bundle(affordable_bundles_ls, U, u_cubicle)
    # return the best affordable bundle
    return best_affordable_bundle


# demand_vector is the vector of demand across all agents
demand_vector = jax.vmap(individual_demand, in_axes=(0, 0, 0, None, None))


def aggregate_demand(
    budgets: jax.Array,
    U: jax.Array,
    u_cubicle: jax.Array,
    bundles: jax.Array,
    bundle_prices: jax.Array,
):
    """Return the aggregate demand, aka the bundle that maximizes utility subject to budget constraint.

    Args:
        budgets (jax.Array): A vector of budgets.
        U (jax.Array): A utility matrix, that is a 2D square triangular numpy array of shape (number_of_half_days, number_of_half_days).
        u_cubicle (jax.Array): A utility vector of shape (number_of_cubicles,).
        bundles (jax.Array): A list of bundles.
        prices (jax.Array): A price vector of shape (number_of_half_days * number_of_cubicles,).
    Returns:
        jax.Array: The aggregate demand.
    """
    # compute the demand vector
    demand = demand_vector(budgets, U, u_cubicle, bundles, bundle_prices)
    # compute the aggregate demand
    aggregate_demand = jnp.sum(demand, axis=0)

    return aggregate_demand


def excess_demand(
    budgets: jax.Array,
    U: jax.Array,
    u_cubicle: jax.Array,
    bundles: jax.Array,
    price_vector: jax.Array,
    supply: jax.Array,
):
    """Return the excess demand, aka the aggregate demand minus the total supply.

    Args:
        budgets (jax.Array): A vector of budgets.
        U (jax.Array): A utility matrix, that is a 2D square triangular numpy array of shape (number_of_half_days, number_of_half_days).
        u_cubicle (jax.Array): A utility vector of shape (number_of_cubicles,).
        bundles (jax.Array): A list of bundles.
        price_vector (jax.Array): A price vector of shape (number_of_half_days * number_of_cubicles,).
    Returns:
        jax.Array: The excess demand.
    """
    bundle_prices = price_bundles(bundles, price_vector)

    excess_demand = (
        aggregate_demand(budgets, U, u_cubicle, bundles, bundle_prices) - supply
    )

    return excess_demand
