from typing import Protocol, Sequence, Union

import jax
import jax.numpy as jnp

from cubiclematch_jax.aggregates.market_level import compute_agg_quantities
from cubiclematch_jax.demand.individual_demand import \
    vmap_ind_demand_from_utility
from cubiclematch_jax.demand.price import price_bundles
from cubiclematch_jax.find_ACE import ACE_algorithm, ACE_iteration_extended
from cubiclematch_jax.price.aux_func import generate_random_price_vector
from cubiclematch_jax.price.neighbors import find_all_neighbors


class Agent(Protocol):
    name: str
    budget: float
    utility_matrix: jax.Array


def main_algorithm(
    key,
    agents: Sequence[Agent],
    bundles: jax.Array,
    supply: jax.Array,
    settings: Union[dict[str, float], None] = None,
    verbose: bool = False,
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
    step_sizes = jnp.logspace(0, 10, num_gradient_num)
    budgets = jnp.array([agent.budget for agent in agents])
    max_budget = float(jnp.max(budgets))
    utility_matrices = jnp.array([agent.utility_matrix for agent in agents])
    length = len(supply)

    if settings:
        raise ValueError(f"Unknown settings: {settings}")

    def compute_demand_vector(price_vec):
        bundles_prices = price_bundles(bundles, price_vec)
        return vmap_ind_demand_from_utility(
            bundles_prices,
            budgets,
            utility_matrices,
            bundles,
        )

    @jax.jit
    def evaluate_prices(price_vec):
        return compute_agg_quantities(
            price_vec,
            compute_demand_vector,
            supply,
        )

    @jax.jit
    def gen_rand_price_vec(key):
        return generate_random_price_vector(length, max_budget, key)

    def find_neighbors(price_vec, agg_dict, tabu_list):
        return find_all_neighbors(
            price_vector=price_vec,
            excess_demand=agg_dict["excess_demand_vec"],
            excess_budgets=agg_dict["excess_budgets"],
            step_sizes=step_sizes,
            evaluate_prices=evaluate_prices,
            tabu_list=tabu_list,
        )
    def ACE_iteration(key):
        return ACE_iteration_extended(
            key,
            evaluate_prices,
            gen_rand_price_vec,
            find_neighbors,
            max_C=int(max_C),
        )

    res = ACE_algorithm(key, ACE_iteration, max_hours=max_hour, verbose=verbose)
    final_agg_quantities = evaluate_prices(res["price_vec"])
    return res, final_agg_quantities
