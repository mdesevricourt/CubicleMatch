import jax
import jax.numpy as jnp

from cubiclematch_jax.preferences import find_best_bundle_from_ordering
from cubiclematch_jax.price import affordable_bundles


def compute_individual_demand(
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
    affordable_bundles_ls = affordable_bundles(available_bundles, bundle_prices, budget)
    # compute the total utility of each affordable bundle
    best_affordable_bundle, best_bundle_index = find_best_bundle_from_ordering(
        affordable_bundles_ls, preference_ordering=preference_ordering
    )
    # find the price of the best affordable bundle
    price = bundle_prices[best_bundle_index]
    excess_budget = budget - price
    return best_affordable_bundle, excess_budget


calculate_demand_vector = jax.vmap(
    compute_individual_demand, in_axes=(None, 0, 0, None)
)
