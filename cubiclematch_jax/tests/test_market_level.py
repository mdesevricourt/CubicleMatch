import jax
import jax.numpy as jnp

from cubiclematch_jax.market_level import (
    compute_aggregate_quantities,
    compute_aggregate_quantities_vec,
)


def test_compute_aggregate_quantities(verbose=False):
    """Test the compute_aggregate_quantities_vec function"""
    price_vector = jnp.array([1, 2])
    budgets = jnp.array([1, 2])
    U1 = jnp.array([[1, 0], [0, 2]])
    U_tilde = jnp.array([U1, U1])
    bundle1 = jnp.array([0, 0])
    bundle2 = jnp.array([1, 1])
    bunlde3 = jnp.array([1, 0])
    bundle4 = jnp.array([0, 1])
    bundles = jnp.array([bundle1, bundle2, bunlde3, bundle4])
    supply = jnp.array([1, 1])

    res = compute_aggregate_quantities(price_vector, budgets, U_tilde, bundles, supply)
    if verbose:
        print(res)
    assert jnp.allclose(res["agg_demand"], jnp.array([1, 1]))
    assert jnp.allclose(res["excess_demand_vec"], jnp.array([0, 0]))
    assert jnp.allclose(res["z"], jnp.array([0, 0]))
    assert jnp.allclose(res["alpha"], jnp.array(0))


def test_compute_aggregate_quantities_vec(verbose=False):
    """Test the compute_aggregate_quantities_vec function"""
    price_vectors = jnp.array([[1, 2], [1, 2]])
    budgets = jnp.array([1, 2])
    U1 = jnp.array([[1, 0], [0, 2]])
    U_tilde = jnp.array([U1, U1])
    bundle1 = jnp.array([0, 0])
    bundle2 = jnp.array([1, 1])
    bunlde3 = jnp.array([1, 0])
    bundle4 = jnp.array([0, 1])
    bundles = jnp.array([bundle1, bundle2, bunlde3, bundle4])
    supply = jnp.array([1, 1])

    res = compute_aggregate_quantities_vec(
        price_vectors, budgets, U_tilde, bundles, supply
    )
    if verbose:
        print(res)


if __name__ == "__main__":
    test_compute_aggregate_quantities(verbose=True)
    test_compute_aggregate_quantities_vec(verbose=True)
