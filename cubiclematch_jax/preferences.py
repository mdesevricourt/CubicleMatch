import jax
import jax.numpy as jnp
import numpy as np

Bundle = (
    jax.Array
)  # a bundle is a vector of 0s and 1s of length number_of_half_days * number_of_cubicles


def find_best_bundle_from_ordering(bundles: jax.Array, preference_ordering: jax.Array):
    """This function finds the best bundle from a list of bundles and a preference ordering.

    Args:
        bundles (jax.Array): A list of bundles.
        preference_ordering (jax.Array): A preference ordering.

    Returns:
        best_bundle(jax.Array): The best bundle.
    """

    # find position of each bundle in the preference ordering
    positions = find_positions_bundles(bundles, preference_ordering)

    # find the index of the best bundle
    best_bundle_index = jnp.argmin(positions)

    return bundles[best_bundle_index], best_bundle_index


def find_position_bundle_ordering(bundle: jax.Array, preference_ordering: jax.Array):
    """This function finds the position of a bundle in a preference ordering.

    Args:
        bundle (jax.Array): A bundle.
        preference_ordering (jax.Array): A preference ordering.

    Returns:
        position (int): The position of the bundle in the preference ordering.
    """

    # find position of bundle in the preference ordering
    matches = (preference_ordering == bundle).all(axis=1)
    position = jnp.argmax(matches)
    default_value = preference_ordering.shape[0] + 1

    return jnp.where(matches.any(), position, default_value)


find_positions_bundles = jax.vmap(find_position_bundle_ordering, in_axes=(0, None))


find_ind_demands_from_available_bundles = jax.vmap(
    find_best_bundle_from_ordering, in_axes=(0, 0)
)
