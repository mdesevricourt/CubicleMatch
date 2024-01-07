import jax
import jax.numpy as jnp

from cubiclematch_jax.preferences import (
    find_best_bundle_from_ordering,
    find_ind_demands_from_available_bundles,
    find_position_bundle_ordering,
    find_positions_bundles,
)


def test_find_position_bundle(verbose=False):
    bundle1 = jnp.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    bundle2 = jnp.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    bundle3 = jnp.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1])

    bundles = jnp.array([bundle1, bundle2, bundle3])
    preference_ordering = jnp.array([bundle3, bundle1, bundle2])

    position1 = find_position_bundle_ordering(bundle1, preference_ordering)
    position2 = find_position_bundle_ordering(bundle2, preference_ordering)
    position3 = find_position_bundle_ordering(bundle3, preference_ordering)

    if verbose:
        print(position1)
        print(position2)
        print(position3)

    assert jnp.allclose(position1, jnp.array([1]))
    assert jnp.allclose(position2, jnp.array([2]))
    assert jnp.allclose(position3, jnp.array([0]))


def test_find_positions_bundles(verbose=False):
    bundle1 = jnp.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
    bundle2 = jnp.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
    bundle3 = jnp.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)

    bundles = jnp.array([bundle1, bundle2, bundle3])
    preference_ordering = jnp.array([bundle3, bundle1, bundle2])

    positions = find_positions_bundles(bundles, preference_ordering)

    if verbose:
        print(positions)

    assert jnp.allclose(positions, jnp.array([1, 2, 0]))


def test_find_best_bundle_from_ordering(verbose=False):
    bundle1 = jnp.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    bundle2 = jnp.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    bundle3 = jnp.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1])

    bundles1 = jnp.array([bundle1, bundle2, bundle3])
    bundles2 = jnp.array([bundle1, bundle3, bundle2])
    bundles3 = jnp.array([bundle2, bundle1])
    preference_ordering = jnp.array([bundle3, bundle1, bundle2])

    best_bundle1, _ = find_best_bundle_from_ordering(bundles1, preference_ordering)
    best_bundle2, _ = find_best_bundle_from_ordering(bundles2, preference_ordering)
    best_bundle3, _ = find_best_bundle_from_ordering(bundles3, preference_ordering)

    if verbose:
        print(best_bundle1)
        print(best_bundle2)
        print(best_bundle3)

    assert jnp.allclose(best_bundle1, bundle3)
    assert jnp.allclose(best_bundle2, bundle3)
    assert jnp.allclose(best_bundle3, bundle1)


def test_find_ind_demands():
    bundle1 = jnp.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    bundle2 = jnp.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    bundle3 = jnp.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1])

    bundles = jnp.array([bundle1, bundle2, bundle3])
    bundles_ls = jnp.array([bundles, bundles])

    preference_ordering1 = jnp.array([bundle3, bundle1, bundle2])
    preference_ordering2 = jnp.array([bundle2, bundle3, bundle2])

    preference_orderings = jnp.array([preference_ordering1, preference_ordering2])

    demand_vec = find_ind_demands_from_available_bundles(
        bundles_ls, preference_orderings
    )

    print(demand_vec)


if __name__ == "__main__":
    test_find_position_bundle(verbose=True)
    test_find_positions_bundles(verbose=True)
    test_find_best_bundle_from_ordering(verbose=True)
    test_find_ind_demands()
