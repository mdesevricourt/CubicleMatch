import jax
import jax.numpy as jnp


def price_bundle(bundle: jax.Array, prices: jax.Array) -> jax.Array:
    """Return the price of a bundle.

    Args:
        bundle (jax.Array): A bundle, that is a vector of 0s and 1s of length number_of_half_days * number_of_cubicles.
        prices (jax.Array): A price vector of shape (number_of_half_days * number_of_cubicles,).

    Returns:
        jax.Array: A price vector of shape (1,).
    """
    # compute the price
    total_price = jnp.dot(bundle, prices)

    return total_price


price_bundles = jax.vmap(price_bundle, in_axes=(0, None))


def affordable_bundles(
    bundles: jax.Array, bundle_prices: jax.Array, budget: jax.Array
) -> jax.Array:
    """Return the affordable bundles from a list of bundles.

    Args:
        bundles (jax.Array): A list of bundles.
        prices (jax.Array): A price vector of shape (number_of_half_days * number_of_cubicles,).
        budget (float): The budget.

    Returns:
        jax.Array: A list of affordable bundles.
    """
    # find the affordable bundles
    y = bundles * 0
    is_affordable = bundle_prices <= budget
    is_affordable = jnp.expand_dims(is_affordable, axis=1)
    affordable_bundles = jnp.where(is_affordable, bundles, y)

    return affordable_bundles
