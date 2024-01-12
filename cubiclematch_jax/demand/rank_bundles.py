from operator import is_

import jax
import jax.numpy as jnp
from numpy import sort


def rank_bundles(
    bundles: jax.Array,
    utilities: jax.Array,
) -> jax.Array:
    """Rank the bundles according to the total utility. In case of a tie, the bundle with the fewest items is ranked higher.
    Bundles that yield negative or zero utility are removed, unless the bundle is empty.

    Args:
        bundles (jnp.ndarray): The bundles.
        utilities (jnp.ndarray): The utilities of the bundles.

    Returns:
        jnp.ndarray: The ranked bundles.
    """
    # sort bundles by utility and then by number of items (descending)

    _is_zero = jnp.all(bundles == 0, axis=1)
    _utility_greater_than_0 = utilities > 0
    is_relevant = jnp.logical_or(_is_zero, _utility_greater_than_0)

    relevant_bundles = bundles[is_relevant]
    relevant_utilities = utilities[is_relevant]
    num_items = jnp.sum(relevant_bundles, axis=1)

    # sorted bundles by utility and then by number of items (descending)
    sorted_bundles = relevant_bundles[jnp.lexsort((num_items, -relevant_utilities))]

    return sorted_bundles
