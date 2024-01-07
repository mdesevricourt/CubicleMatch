from typing import Protocol, Union

import jax
import jax.numpy as jnp

from cubiclematch_jax.aux_func import compute_total_budget, generate_random_price_vector
from cubiclematch_jax.find_ACE import ACE_algorithm, ACE_iteration_extended
from cubiclematch_jax.individual_demand import calculate_demand_vector
from cubiclematch_jax.market_level import compute_agg_quantities
from cubiclematch_jax.neighbors import find_all_neighbors


class Agent(Protocol):
    name: str
    budget: float
    preferences: jax.Array


def main(
    key,
    agents: list[Agent],
    bundles: jax.Array,
    supply: jax.Array,
    settings: Union[dict[str, float], None] = None,
):
    """Run the ACE algorithm.

    Args:
        key (jax.random.PRNGKey): The key.
        settings (Union[dict[str, float], None], optional): The settings. Defaults to None.

    Returns
        jnp.ndarray: The price vector.
        jnp.ndarray: The excess demand vector.
        jnp.ndarray: The excess budgets.
        jnp.ndarray: The number of iterations.
    """

    if settings is None:
        settings = {}
    settings = settings.copy()
    max_hour = settings.pop("max_hour", 0.1)
    max_C = settings.pop("max_C", 5.0)
    num_gradient_num = int(settings.pop("num_gradient_num", 10))
    step_sizes = jnp.logspace(-1, 1, num_gradient_num)
    budgets = jnp.array([agent.budget for agent in agents])
    total_budget = compute_total_budget(budgets)
    preference_orderings = jnp.array([agent.preferences for agent in agents])
    length = len(supply)

    def compute_demand_vector(price_vec):
        return calculate_demand_vector(
            price_vec,
            budgets,
            preference_ordering=preference_orderings,
            available_bundles=bundles,
        )

    def evaluate_price_vec(price_vec):
        return compute_agg_quantities(
            price_vec,
            bundles,
            compute_demand_vector,
            supply,
        )

    def gen_rand_price_vec(key):
        return generate_random_price_vector(length, total_budget, key)

    def find_neighbors(price_vec, agg_dict, tabu_list):
        find_all_neighbors(
            price_vector=price_vec,
            excess_demand=agg_dict["excess_demand_vec"],
            excess_budgets=agg_dict["excess_budgets"],
            step_sizes=step_sizes,
            compute_agg_quantities=evaluate_price_vec,
            tabu_list=tabu_list,
        )

    def ACE_iteration(key):
        return ACE_iteration_extended(
            key,
            evaluate_price_vec,
            gen_rand_price_vec,
            find_neighbors,
            max_C=int(max_C),
        )

    res = ACE_algorithm(key, ACE_iteration, max_hours=max_hour)

    return res
