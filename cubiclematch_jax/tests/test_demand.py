import jax
import jax.numpy as jnp
import pytest

from cubiclematch_jax.demand import demand_vector, find_agent_demand, individual_demand
from cubiclematch_jax.market_level import compute_agg_quantities
from cubiclematch_jax.price import price_bundles
from cubiclematch_jax.utility import create_U_tilde


def test_individual_demand():
    budget = jnp.array(30)
    U = jnp.array(
        [
            [10, 5, 5, 0, 0],
            [0, 10, 5, 0, 0],
            [0, 0, 10, 5, 0],
            [0, 0, 0, 10, 5],
            [0, 0, 0, 0, 10],
        ]
    )

    u_cubicle = jnp.array([0, 1])
    bundle1 = jnp.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
    bundle2 = jnp.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    bundle3 = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    bundle4 = bundle2

    bundles = jnp.array([bundle1, bundle2, bundle3, bundle4])
    prices = jnp.array([10, 5, 5, 0, 0, 10, 5, 0, 0, 0])
    bundle_prices = price_bundles(bundles, prices)
    U_tilde = create_U_tilde(U, u_cubicle)
    expected = bundle2
    demanded_bundle, _ = individual_demand(budget, U_tilde, bundles, bundle_prices)
    assert jnp.allclose(demanded_bundle, expected)


def test_vector_demand():
    budget1 = 50
    budget2 = 25
    budgets = jnp.array([budget1, budget2])

    U1 = jnp.array(
        [
            [10, 5, 5, 0, 0],
            [0, 10, 5, 0, 0],
            [0, 0, 10, 5, 0],
            [0, 0, 0, 10, 5],
            [0, 0, 0, 0, 10],
        ]
    )
    U2 = U1

    U_ls = [U1, U2]
    u_cubicle1 = jnp.array([5, 0])
    u_cubicle2 = jnp.array([0, 5])
    u_cubicle_ls = [u_cubicle1, u_cubicle2]

    U_tilde = jnp.array(
        [create_U_tilde(U, u_cubicle) for U, u_cubicle in zip(U_ls, u_cubicle_ls)]
    )
    U = jnp.array(U_ls)
    u_cubicle = jnp.array(u_cubicle_ls)

    bundle1 = jnp.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    bundle2 = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    bundle3 = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    bundle4 = jnp.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    bundle5 = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0])
    bundles = jnp.array([bundle1, bundle2, bundle3, bundle4, bundle5])

    prices = jnp.array([10, 10, 10, 10, 10, 5, 5, 5, 5, 5])
    bundle_prices = price_bundles(bundles, prices)

    bundles = jnp.array([bundle1, bundle2, bundle3, bundle4, bundle5])

    expected1 = bundle1
    expected2 = bundle2
    expected = jnp.array([expected1, expected2])

    demanded_bundles, _ = demand_vector(budgets, U_tilde, bundles, bundle_prices)
    print(demanded_bundles)
    assert jnp.allclose(demanded_bundles, expected)


# Test cases
def test_find_agent_demand_all_demand():
    demanded_bundles = jnp.array([[1, 1], [1, 1]])
    index = 0
    assert jnp.array_equal(
        find_agent_demand(index, demanded_bundles), jnp.array([0, 1])
    )


def test_find_agent_demand_no_demand():
    demanded_bundles = jnp.array([[0, 0], [0, 0]])
    index = 0
    assert jnp.array_equal(find_agent_demand(index, demanded_bundles), jnp.array([]))


def test_find_agent_demand_mixed_demand():
    demanded_bundles = jnp.array([[1, 0], [0, 1], [1, 1]])
    index = 1
    assert jnp.array_equal(
        find_agent_demand(index, demanded_bundles), jnp.array([1, 2])
    )


if __name__ == "__main__":
    test_individual_demand()
    test_vector_demand()
    test_find_agent_demand_all_demand()
    test_find_agent_demand_no_demand()
    test_find_agent_demand_mixed_demand()
