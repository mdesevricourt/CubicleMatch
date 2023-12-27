from tabnanny import verbose

import jax
import jax.numpy as jnp

from cubiclematch_jax.price import affordable_bundles, price_bundles


def test_price_bundles():
    bundle1 = jnp.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
    bundle2 = jnp.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    prices = jnp.array([10, 5, 0, 0, 0, 10, 5, 0, 0, 0])

    expected = jnp.array([30, 25])

    assert jnp.allclose(price_bundles(jnp.array([bundle1, bundle2]), prices), expected)


def test_affordable_bundles(verbose=False):
    bundle1 = jnp.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
    bundle2 = jnp.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    bundle3 = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    bundles = jnp.array([bundle1, bundle2, bundle3])
    prices = jnp.array([10, 5, 5, 0, 0, 10, 5, 0, 0, 0])
    budget = jnp.array(30)
    bundle_prices = price_bundles(bundles, prices)

    bundle0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    expected = jnp.array([bundle1, bundle2, bundle0])
    if verbose:
        print(affordable_bundles(bundles, bundle_prices, budget))

    assert jnp.allclose(affordable_bundles(bundles, bundle_prices, budget), expected)


if __name__ == "__main__":
    test_price_bundles()
    test_affordable_bundles(verbose=True)
