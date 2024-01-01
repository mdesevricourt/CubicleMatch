"""Computes market level quantities from prices and primitives."""

from ast import mod

import jax
import jax.numpy as jnp

from cubiclematch_jax.demand import (
    compute_aggregate_demand,
    compute_clearing_error,
    compute_excess_demand,
    demand_vector,
    modified_excess_demand,
)
from cubiclematch_jax.price import price_bundles


def aggregate_quantities(
    prices: jax.Array,
    budgets: jax.Array,
    U_tilde: jax.Array,
    bundles: jax.Array,
    supply: jax.Array,
):
    bundle_prices = price_bundles(bundles, prices)
    demand, excess_budgets = demand_vector(budgets, U_tilde, bundles, bundle_prices)
    agg_demand = compute_aggregate_demand(demand)
    excess_demand_vec = compute_excess_demand(agg_demand, supply)
    z = modified_excess_demand(excess_demand_vec, prices)
    alpha = compute_clearing_error(z)

    res = {
        "demand": demand,
        "excess_budgets": excess_budgets,
        "agg_demand": agg_demand,
        "excess_demand_vec": excess_demand_vec,
        "z": z,
        "alpha": alpha,
    }

    return res
