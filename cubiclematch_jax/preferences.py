import jax
import jax.numpy as jnp

Bundle = (
    jax.Array
)  # a bundle is a vector of 0s and 1s of length number_of_half_days * number_of_cubicles


def collapse_bundle(bundle: Bundle, num_half_days: int, num_cubicles: int) -> jax.Array:
    """Return the collapsed bundle, that is the bundle, ignoring the cubicle dimension, that is the sum of the cubicle bundles."""

    # reshape the bundle to be a 2D array with shape (number_of_half_days, number_of_cubicles)
    bundle = jnp.reshape(bundle, (num_cubicles, num_half_days))

    # sum the bundle across the cubicle dimension
    collapsed_bundle = jnp.sum(bundle, axis=0)
    # take the minimum of the collapsed bundle and 1
    collapsed_bundle = jnp.minimum(collapsed_bundle, 1)

    return collapsed_bundle


def utility_over_half_days(collapsed_bundle: jax.Array, U: jax.Array) -> jax.Array:
    """Return the utility of a person for a given assignment across all half-days

    Args:
        collapsed_bundle (jax.Array): A collapsed bundle, that is a vector of 0s and 1s of length number_of_half_days.
        U (jax.Array): A utility matrix, that is a 2D square triangular numpy array of shape (number_of_half_days, number_of_half_days).

    Returns:
        jax.Array: A utility vector of shape (1,).
    """

    # reshape the collapsed bundle to be a 2D array with shape (number_of_half_days, 1)
    collapsed_bundle = jnp.reshape(collapsed_bundle, (-1, 1))

    # compute the utility
    utility = jnp.matmul(jnp.matmul(collapsed_bundle.T, U), collapsed_bundle)

    return utility


def utility_over_cubicles(bundle: jax.Array, u_cubicle: jax.Array) -> jax.Array:
    """Return the utility of a person from the cubicle.

    Args:
        bundle (jax.Array): A bundle, that is a vector of 0s and 1s of length number_of_half_days * number_of_cubicles.
        u_cubicle (jax.Array): A utility vector of shape (number_of_cubicles,).

    Returns:
        jax.Array: A utility vector of shape (1,).
    """
    number_of_cubicles = u_cubicle.shape[0]
    # reshape the bundle to be a 2D array with shape (number_of_cubicles, number_of_half_days)
    bundle = jnp.reshape(bundle, (number_of_cubicles, -1))

    # sum the bundle across the half-day dimension
    bundle = jnp.sum(bundle, axis=1)

    # compute the utility
    utility = jnp.dot(bundle, u_cubicle)
    # reduce the utility to a scalar

    return jnp.squeeze(utility)


def total_utility_bundle(
    bundle: jax.Array, U: jax.Array, u_cubicle: jax.Array
) -> jax.Array:
    """Return the total utility of a person from the cubicle.

    Args:
        bundle (jax.Array): A bundle, that is a vector of 0s and 1s of length number_of_half_days * number_of_cubicles.
        U (jax.Array): A utility matrix, that is a 2D square triangular numpy array of shape (number_of_half_days, number_of_half_days).
        u_cubicle (jax.Array): A utility vector of shape (number_of_cubicles,).

    Returns:
        jax.Array: A utility vector of shape (1,).
    """
    collapsed_bundle = collapse_bundle(bundle, U.shape[0], u_cubicle.shape[0])
    utility = jnp.squeeze(
        utility_over_half_days(collapsed_bundle, U)
        + utility_over_cubicles(bundle, u_cubicle)
    )
    return utility


total_utility_bundles = jax.vmap(total_utility_bundle, in_axes=(0, None, None))


def find_best_bundle(bundles: jax.Array, U: jax.Array, u_cubicle: jax.Array):
    """Return the best bundle from a list of bundles.

    Args:
        bundles (jax.Array): A list of bundles.
        U (jax.Array): A utility matrix, that is a 2D square triangular numpy array of shape (number_of_half_days, number_of_half_days).
        u_cubicle (jax.Array): A utility vector of shape (number_of_cubicles,).

    Returns:
        jax.Array: The best bundle.
    """

    # compute the total utility of each bundle
    utilities = total_utility_bundles(bundles, U, u_cubicle)

    # find the index of the best bundle
    best_bundle_index = jnp.argmax(utilities)

    # return the best bundle
    return bundles[best_bundle_index]
