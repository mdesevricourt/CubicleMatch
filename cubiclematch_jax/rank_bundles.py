from operator import is_

import jax
import jax.numpy as jnp
from numpy import sort


def rank_bundles(
    bundles: jax.Array,
    utilities: jax.Array,
) -> jax.Array:
    """Rank the bundles according to the total utility.

    Args:
        bundles (jnp.ndarray): The bundles.
        utilities (jnp.ndarray): The utilities of the bundles.

    Returns:
        jnp.ndarray: The ranked bundles.
    """
    # sort bundles by utility
    _is_zero = jnp.all(bundles == 0, axis=1)
    _utility_greater_than_0 = utilities > 0
    is_relevant = jnp.logical_or(_is_zero, _utility_greater_than_0)

    relevant_bundles = bundles[is_relevant]
    relevant_utilities = utilities[is_relevant]

    sorted_indices = jnp.argsort(relevant_utilities, axis=0)[::-1]

    sorted_bundles = relevant_bundles[sorted_indices]

    return sorted_bundles
