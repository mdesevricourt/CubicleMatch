import jax
import jax.numpy as jnp

from cubiclematch_jax.demand import rank_bundles
from cubiclematch_jax.demand.preferences import find_best_bundle_from_ordering
from cubiclematch_jax.demand.price import filter_affordable_bundles
from cubiclematch_jax.demand.utility import vmap_utility_over_half_days


def ind_demand_from_preference(
    bundle_prices: jax.Array,
    budget: jax.Array,
    preference_ordering: jax.Array,
    available_bundles: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the demand of an individual, aka the bundle that maximizes utility subject to budget constraint.

    Args:
        bundles_prices (jax.Array): A price vector of shape (number_of_half_days * number_of_cubicles,).
        budget (float): The budget.
        preference_ordering (jax.Array): A preference ordering over bundles.
        available_bundles (jax.Array): A list of available bundles.
    Returns:
        best_affordable_bundle (jax.Array): The best affordable bundle.
        excess_budget (float): The excess budget.
    """

    # find the affordable bundles
    affordable_bundles_ls = filter_affordable_bundles(
        available_bundles, bundle_prices, budget
    )
    # compute the total utility of each affordable bundle
    best_affordable_bundle, best_bundle_index = find_best_bundle_from_ordering(
        affordable_bundles_ls, preference_ordering=preference_ordering
    )
    # find the price of the best affordable bundle
    price = bundle_prices[best_bundle_index]
    excess_budget = budget - price
    return best_affordable_bundle, excess_budget


vmap_ind_demand_from_preference = jax.vmap(
    ind_demand_from_preference, in_axes=(None, 0, 0, None)
)


def ind_demand_from_utility(
    bundle_prices: jax.Array,
    budget: jax.Array,
    utility_matrix: jax.Array,
    available_bundles: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the demand of an individual, aka the bundle that maximizes utility subject to budget constraint.

    Args:
        bundles_prices (jax.Array): A price vector of shape (number_of_half_days * number_of_cubicles,).
        budget (float): The budget.
        utility_matrix (jax.Array): A utility matrix of shape (number_of_half_days * number_of_cubicles, number_of_agents).
        available_bundles (jax.Array): A list of available bundles.
    Returns:
        best_affordable_bundle (jax.Array): The best affordable bundle.
        excess_budget (float): The excess budget.
    """

    # find the affordable bundles
    affordable_bundles = filter_affordable_bundles(
        available_bundles, bundle_prices, budget
    )
    # compute the total utility of each affordable bundle
    bundles_utility = vmap_utility_over_half_days(affordable_bundles, utility_matrix)
    sorted_index = rank_bundles.sorted_index_bundles(
        affordable_bundles, bundles_utility
    )
    # best bundle
    best_bundle_index = sorted_index[0]

    best_bundle = available_bundles[best_bundle_index]
    price = bundle_prices[best_bundle_index]
    excess_budget = budget - price
    return best_bundle, excess_budget


vmap_ind_demand_from_utility = jax.vmap(
    ind_demand_from_utility, in_axes=(None, 0, 0, None)
)
