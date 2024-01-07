import jax.numpy as jnp
import pytest

from cubiclematch_jax.individual_demand import compute_individual_demand


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


def test_individual_demand_preference_ordering():
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


if __name__ == "__main__":
    pytest.main([__file__])
