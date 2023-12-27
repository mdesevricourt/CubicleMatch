import pprint

pp = pprint.PrettyPrinter(indent=4)
import itertools

import jax
import jax.numpy as jnp
import numpy as np


def available_bundles(num_half_days: int, num_cubicles: int):
    """Return the list of all possible bundles, under the assumption that each cubicle is available for each half-day.

    Args:
        num_half_days (int): The number of half-days.
        num_cubicles (int): The number of cubicles.

    Returns:
        list[list[int]]: A list of bundles.
    """
    # create a list of bundles

    all_combinations = [
        jnp.array(t)
        for t in itertools.product([0, 1], repeat=num_half_days * num_cubicles)
    ]
    all_combinations = jnp.array(all_combinations)

    @jax.jit
    def check_consistency(bundle: jax.Array):
        """Return True if the bundle does not repeat half-days across cubicles, False otherwise."""

        # reshape the bundle to be a 2D array with shape (number_of_cubicles, number_of_half_days)
        bundle = jnp.reshape(bundle, (num_cubicles, num_half_days))

        # sum the bundle across the half-day dimension
        sum_consistency = jnp.sum(bundle, axis=0)

        return jnp.max(sum_consistency) <= 1

    check_consistency_bundles = jax.vmap(check_consistency)
    consistency_vector = check_consistency_bundles(all_combinations)
    bundles = all_combinations[consistency_vector]
    return bundles


if __name__ == "__main__":
    print(available_bundles(5, 8))
