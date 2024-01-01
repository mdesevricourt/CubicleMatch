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
) -> tuple[jax.Array, jax.Array]:
    """Return the demand of an individual, aka the bundle that maximizes utility subject to budget constraint.

    Args:
        budget (float): The budget.
        U (jax.Array): A utility matrix, that is a 2D square triangular numpy array of shape (number_of_half_days, number_of_half_days).
        u_cubicle (jax.Array): A utility vector of shape (number_of_cubicles,).
        bundles (jax.Array): A list of bundles.
        prices (jax.Array): A price vector of shape (number_of_half_days * number_of_cubicles,).
    Returns:
        best_affordable_bundle (jax.Array): The best affordable bundle.
        excess_budget (float): The excess budget.
    """

    # find the affordable bundles
    affordable_bundles_ls = affordable_bundles(bundles, bundle_prices, budget)
    # compute the total utility of each affordable bundle
    best_affordable_bundle, best_bundle_index = find_best_bundle(
        affordable_bundles_ls, U, u_cubicle
    )
    # find the price of the best affordable bundle
    price = bundle_prices[best_bundle_index]
    excess_budget = budget - price
    return best_affordable_bundle, excess_budget


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
        aggregate_demand (jax.Array): The aggregate demand.
        excess_budgets (jax.Array): The excess budgets.
    """
    # compute the demand vector
    demand, excess_budgets = demand_vector(
        budgets, U, u_cubicle, bundles, bundle_prices
    )
    # compute the aggregate demand
    aggregate_demand = jnp.sum(demand, axis=0)

    return aggregate_demand, excess_budgets


def compute_excess_demand(
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


def clearing_error(
    price_vector: jax.Array,
    excess_demand: jax.Array,
):
    """Return the clearing error, aka the sum of the excess demand squared.

    Args:
        budgets (jax.Array): A vector of budgets.
        excess_demand (jax.Array): The excess demand.
    Returns:
        jax.Array: The clearing error.
        z (jax.Array): The excess demand modified so that it is non-negative.
        excess_demand (jax.Array): The excess demand.
    """

    z = jnp.where(price_vector != 0, excess_demand, jnp.max(excess_demand, 0))

    alpha = jnp.sum(z**2)

    return alpha, z


def clearing_error_primitives(
    price_vector: jax.Array,
    budget: jax.Array,
    U: jax.Array,
    u_cubicle: jax.Array,
    bundles: jax.Array,
    supply: jax.Array,
):
    """Return the clearing error, aka the sum of the excess demand squared.

    Args:
        budgets (jax.Array): A vector of budgets.
        excess_demand (jax.Array): The excess demand.
    Returns:
        jax.Array: The clearing error.
        z (jax.Array): The excess demand modified so that it is non-negative.
        excess_demand (jax.Array): The excess demand.
    """
    excess_demand = compute_excess_demand(
        budget, U, u_cubicle, bundles, price_vector, supply
    )
    alpha, z = clearing_error(price_vector, excess_demand)
    return alpha, z, excess_demand


clearing_error_prices = jax.vmap(
    clearing_error_primitives, in_axes=(0, None, None, None, None, None)
)
