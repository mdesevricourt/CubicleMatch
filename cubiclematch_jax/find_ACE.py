"""This module implements the ACE algorithm for finding the best matching of agents to cubicles."""

import jax
import jax.numpy as jnp

from cubiclematch_jax.aux_func import ACE_iteration


def ACE_algorithm(
    price_vector: jax.Array,
    bundles: jax.Array,
    budgets: jax.Array,
    U_tilde: jax.Array,
    supply: jax.Array,
    step_sizes: jax.Array,
    tabu_list: jax.Array,
    max_iter: int,
    tol: float,
):
    """Find the best matching of agents to cubicles.

    Args:
        price_vector (jax.Array): The initial price vector.
        bundles (jax.Array): The bundles.
        budgets (jax.Array): The budgets.
        U_tilde (jax.Array): The utility function.
        supply (jax.Array): The supply.
        step_sizes (jax.Array): The step sizes for the gradient neighbors.
        tabu_list (jax.Array): The tabu list.
        max_iter (int): The maximum number of iterations.
        tol (float): The tolerance level for the clearing error.

    Returns:
        price_vector (jax.Array): The final price vector.
        clearing_error (jax.Array): The final clearing error.

    """
    # initialize
    clearing_error = tol + 1
    iter = 0

    # iterate
    while clearing_error > tol and iter < max_iter:
        price_vector, clearing_error = ACE_iteration(
            price_vector, bundles, budgets, U_tilde, supply, step_sizes, tabu_list
        )
        iter += 1

    return price_vector, clearing_error
