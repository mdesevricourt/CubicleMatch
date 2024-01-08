"""This module implements the ACE algorithm for finding the best matching of agents to cubicles."""

import time
from typing import Callable

import jax
import jax.numpy as jnp

from cubiclematch_jax.aux_func import (
    select_best_neighbor,
    sort_neighbors_by_clearing_error,
)


def ACE_iteration_extended(
    key,
    evaluate_prices: Callable,
    gen_rand_price_vec: Callable,
    find_neighbors: Callable,
    max_C: int = 5,
):
    """Perform one iteration of the ACE algorithm.

    Args:
        key (jax.random.PRNGKey): The key.
        evaluate_price_vec (Callable): The function for evaluating the price vector.
        gen_rand_price_vec (Callable): The function for generating a random price vector.
        find_neighbors (Callable): The function for finding the neighbors.
        max_C (int, optional): The maximum number of iterations. Defaults to 5.

    Returns:
        price_vector (jax.Array): The final price vector.
        clearing_error (jax.Array): The final clearing error
        number_excess_demand (jax.Array): The number of excess demand.
        key (jax.random.PRNGKey): The new key.

    """
    evaluate_price_vecs = jax.vmap(evaluate_prices, in_axes=0)
    c = 0

    current_p, key = gen_rand_price_vec(key)

    agg_quantities = evaluate_prices(current_p)

    best_p = current_p
    search_error = agg_quantities["clearing_error"]
    search_number_excess_demand = agg_quantities["number_excess_demand"]
    tabu_list = jnp.array([agg_quantities["excess_demand_vec"]])

    while c < max_C and search_error > 0:
        print(
            f"c: {c}, search_error: {search_error}, search_number_excess_demand: {search_number_excess_demand}"
        )
        c += 1
        neighbors = find_neighbors(current_p, agg_quantities, tabu_list)
        if len(neighbors) == 0:
            break
        agg_quantities = evaluate_price_vecs(neighbors)
        # sort prices by clearing error and number of excess demand
        current_p, agg_quantities = select_best_neighbor(neighbors, agg_quantities)

        current_clearing_error = agg_quantities["clearing_error"]
        current_number_excess_demand = agg_quantities["number_excess_demand"]
        current_excess_demand_vec = agg_quantities["excess_demand_vec"]

        tabu_list = jnp.vstack((tabu_list, current_excess_demand_vec))

        better_p_found = current_clearing_error < search_error or (
            current_clearing_error <= search_error
            and current_number_excess_demand < search_number_excess_demand
        )

        if better_p_found:
            best_p = current_p
            search_error = current_clearing_error
            search_number_excess_demand = current_number_excess_demand
            c = 0

    return best_p, search_error, search_number_excess_demand, key


def ACE_algorithm(
    key,
    ACE_iteration: Callable,
    max_hours: float,
    verbose: bool = False,
):
    """Find the best matching of agents to cubicles.

    Args:
        key (jax.random.PRNGKey): The initial random key.
        ACE_iteration (Callable): The ACE iteration function. It takes in a key and returns a price vector, clearing error, number of excess demand, and a new key.
        max_hours (float): The maximum number of hours.
        tol (float): The tolerance.

    """
    start_time = time.time()
    current_time = start_time

    best_error = jnp.inf
    best_number_excess_demand = jnp.inf
    best_p = jnp.array([])

    while current_time - start_time < max_hours * 3600 and best_error > 0:
        if verbose:
            print(f"best_error: {best_error}")
            print(f"best_number_excess_demand: {best_number_excess_demand}")
            print(f"best_p: {best_p}")
        p, error, number_excess_demand, key = ACE_iteration(key)
        if error < best_error or (
            error <= best_error and number_excess_demand < best_number_excess_demand
        ):
            best_error = error
            best_number_excess_demand = number_excess_demand
            best_p = p

        current_time = time.time()

    return best_p, best_error, best_number_excess_demand, key
