import jax
import jax.numpy as jnp

from cubiclematch_jax.rank_bundles import rank_bundles


def test_rank_bundles(verbose: bool = False):
    bundle1 = [1, 2, 3]
    bundle2 = [1, 2, 4]
    bundle3 = [1, 2, 5]
    bundle4 = [1, 2, 6]
    bundle0 = [0, 0, 0]

    bundles = jnp.array([bundle1, bundle2, bundle3, bundle4, bundle0])

    utilities = jnp.array([1, 2, 0, 4, 0])

    ranked_bundles = rank_bundles(bundles, utilities)
    if verbose:
        print(utilities > 0)
        print(ranked_bundles)

    assert jnp.all(ranked_bundles == jnp.array([bundle4, bundle2, bundle1, bundle0]))


if __name__ == "__main__":
    test_rank_bundles(verbose=True)
