import jax
import jax.numpy as jnp

from cubiclematch_jax.preferences import find_best_bundle_tilde
from cubiclematch_jax.price import affordable_bundles


def individual_demand(
    budget: jax.Array,
    U_tilde: jax.Array,
    bundles: jax.Array,
    bundle_prices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the demand of an individual, aka the bundle that maximizes utility subject to budget constraint.

    Args:
        budget (float): The budget.
        U_tilde (jax.Array): A utility matrix, that is a 2D square triangular numpy array of shape (number_of_half_days, number_of_half_days).
        bundles (jax.Array): A list of bundles.
        prices (jax.Array): A price vector of shape (number_of_half_days * number_of_cubicles,).
    Returns:
        best_affordable_bundle (jax.Array): The best affordable bundle.
        excess_budget (float): The excess budget.
    """

    # find the affordable bundles
    affordable_bundles_ls = affordable_bundles(bundles, bundle_prices, budget)
    # compute the total utility of each affordable bundle
    best_affordable_bundle, best_bundle_index = find_best_bundle_tilde(
        affordable_bundles_ls, U_tilde=U_tilde
    )
    # find the price of the best affordable bundle
    price = bundle_prices[best_bundle_index]
    excess_budget = budget - price
    return best_affordable_bundle, excess_budget


# demand_vector is the vector of demand across all agents
demand_vector = jax.vmap(individual_demand, in_axes=(0, 0, None, None))

# %


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


def find_agent_demand(index: int, demanded_bundles: jax.Array) -> jax.Array:
    """For a given good given by index, return the indices of the agents that demand it.

    Args:
        index (int): The index of the good.
        demanded_bundles (jax.Array): The demanded bundles.

    Returns:
        indices (jax.Array): The indices of the agents that demand the good.
    """
    indices = jnp.where(demanded_bundles[:, index] == 1)[0]
    return indices
