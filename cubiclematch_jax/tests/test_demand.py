import jax
import jax.numpy as jnp

from cubiclematch_jax.demand import aggregate_demand, demand_vector, individual_demand
from cubiclematch_jax.preferences import create_U_tilde
from cubiclematch_jax.price import price_bundles


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
    demanded_bundle, _ = individual_demand(budget, U, u_cubicle, bundles, bundle_prices)
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

    demanded_bundles, _ = demand_vector(budgets, U, u_cubicle, bundles, bundle_prices)
    print(demanded_bundles)
    assert jnp.allclose(demanded_bundles, expected)


def test_aggregate_demand(verbose=False):
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
    agg_demand, _ = aggregate_demand(budgets, U, u_cubicle, bundles, bundle_prices)
    if verbose:
        print(agg_demand)
    assert jnp.allclose(agg_demand, expected1 + expected2)


if __name__ == "__main__":
    test_individual_demand()
    test_vector_demand()
    test_aggregate_demand(verbose=True)
