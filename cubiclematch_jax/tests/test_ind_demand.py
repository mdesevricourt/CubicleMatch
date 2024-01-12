import jax.numpy as jnp

from cubiclematch_jax.demand.individual_demand import (
    calculate_demand_vector,
    compute_individual_demand,
)


def test_individual_demand_budget_constraint():
    # Test that the function respects the budget constraint
    bundle_prices = jnp.array([10, 20, 30])
    budget = jnp.array(15)
    preference_ordering = jnp.array(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    )  # preference in descending order
    available_bundles = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    best_bundle, excess_budget = compute_individual_demand(
        bundle_prices, budget, preference_ordering, available_bundles
    )
    assert jnp.array_equal(
        best_bundle, jnp.array([1, 0, 0])
    )  # Should choose the first bundle which is within budget
    assert excess_budget == 5  # Budget - Cost of bundle


def test_compute_individual_demand():
    # Test that the function chooses the bundle with the highest preference within budget
    bundle_prices = jnp.array([5, 15, 25])
    budget = jnp.array(21)
    preference_ordering = jnp.array(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    )  # preference in descending order
    available_bundles = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    best_bundle, excess_budget = compute_individual_demand(
        bundle_prices, budget, preference_ordering, available_bundles
    )
    assert jnp.array_equal(
        best_bundle, jnp.array([0, 1, 0])
    )  # Should choose the second bundle, which is the most preferred within budget
    assert excess_budget == 6  # Budget - Cost of bundle


def test_calculate_demand_vector(verbose=False):
    # Test that the function chooses the bundle with the highest preference within budget
    bundle_prices = jnp.array([5, 15, 25])
    budgets = jnp.array([21, 21])
    bundle1 = jnp.array([1, 0, 0])
    bundle2 = jnp.array([0, 1, 0])
    bundle3 = jnp.array([0, 0, 1])
    preference_1 = jnp.array([bundle1, bundle2, bundle3])
    preference_2 = jnp.array([bundle3, bundle2, bundle1])
    preference_orderings = jnp.array([preference_1, preference_2])
    available_bundles = jnp.array([bundle1, bundle2, bundle3])

    best_bundles, excess_budgets = calculate_demand_vector(
        bundle_prices, budgets, preference_orderings, available_bundles
    )
    if verbose:
        print(best_bundles)
    assert jnp.array_equal(best_bundles[0], bundle1)

    assert jnp.array_equal(best_bundles[1], bundle2)


def test_calculate_demand_vector_one_agent(verbose=False):
    # Test that the function chooses the bundle with the highest preference within budget
    bundle_prices = jnp.array([5, 15, 25])
    budgets = jnp.array([21])
    bundle1 = jnp.array([1, 0, 0])
    bundle2 = jnp.array([0, 1, 0])
    bundle3 = jnp.array([0, 0, 1])
    preference_1 = jnp.array([bundle1, bundle2, bundle3])
    preference_orderings = jnp.array([preference_1])
    available_bundles = jnp.array([bundle1, bundle2, bundle3])

    best_bundles, excess_budgets = calculate_demand_vector(
        bundle_prices, budgets, preference_orderings, available_bundles
    )
    if verbose:
        print(best_bundles)
        agg_demand = jnp.sum(best_bundles, axis=0)
        print(f"Aggregate demand: {agg_demand}")
    assert jnp.array_equal(best_bundles[0], bundle1)
    assert jnp.array_equal(excess_budgets[0], 16)


if __name__ == "__main__":
    test_individual_demand_budget_constraint()
    test_compute_individual_demand()
    test_calculate_demand_vector(verbose=True)
    test_calculate_demand_vector_one_agent(verbose=True)
