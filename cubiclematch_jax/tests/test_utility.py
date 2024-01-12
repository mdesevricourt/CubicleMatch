import jax.numpy as jnp

from cubiclematch_jax.demand.utility import (
    collapse_bundle,
    create_total_utility_matrix,
    create_utility_matrix_slots,
    find_best_bundle,
    total_utility_bundle,
    total_utility_bundles,
    utility_over_cubicles,
    utility_over_half_days,
)


def test_collapse_bundle():
    """Test the collapse_bundle function"""
    bundle = jnp.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 1])
    num_half_days = 5
    num_cubicles = 2
    collapsed_bundle = collapse_bundle(bundle, num_half_days, num_cubicles)

    assert jnp.allclose(collapsed_bundle, jnp.array([1, 1, 1, 1, 1]))

    bundle = jnp.array([1, 0, 0, 0, 0, 1, 0, 1, 1, 1])
    num_half_days = 5
    num_cubicles = 2
    collapsed_bundle = collapse_bundle(bundle, num_half_days, num_cubicles)

    assert jnp.allclose(collapsed_bundle, jnp.array([1, 0, 1, 1, 1]))


def test_utility_over_half_days():
    collaposed_bundle = jnp.array([1, 1, 0, 1, 1])

    U = jnp.array(
        [
            [10, 5, 0, 0, 0],
            [0, 10, -3, 0, 0],
            [0, 0, 10, 6, 0],
            [0, 0, 0, 20, 8],
            [0, 0, 0, 0, 10],
        ]
    )

    utility = utility_over_half_days(collaposed_bundle, U)

    assert jnp.allclose(utility, jnp.array([63]))


def test_utility_over_cubicles():
    bundle = jnp.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])

    u_cubicle = jnp.array([10, 5])

    utility = utility_over_cubicles(bundle, u_cubicle)

    assert jnp.allclose(utility, jnp.array([60]))


def test_total_utility_cubicle():
    bundle = jnp.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1])

    U = jnp.array(
        [
            [10, 5, 0, 0, 0],
            [0, 10, -3, 0, 0],
            [0, 0, 10, 6, 0],
            [0, 0, 0, 20, 8],
            [0, 0, 0, 0, 10],
        ]
    )

    u_cubicle = jnp.array([10, 5])

    utility_hd = utility_over_half_days(collapse_bundle(bundle, U.shape[0], 2), U)
    utility_cubicle = utility_over_cubicles(bundle, u_cubicle)

    utility = total_utility_bundle(bundle, U, u_cubicle)
    assert jnp.allclose(utility_hd, jnp.array([63]))
    assert jnp.allclose(utility_cubicle, jnp.array([60]))
    assert jnp.allclose(utility, jnp.array([123]))
    assert jnp.allclose(utility, utility_hd + utility_cubicle)


def test_total_utility_cubicles():
    bundle1 = jnp.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
    bundle2 = jnp.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    bundle3 = jnp.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1])

    bundles = jnp.array([bundle1, bundle2, bundle3])

    U = jnp.array(
        [
            [10, 5, 0, 0, 0],
            [0, 10, -3, 0, 0],
            [0, 0, 10, 6, 0],
            [0, 0, 0, 20, 8],
            [0, 0, 0, 0, 10],
        ]
    )
    u_cubicle = jnp.array([10, 5])

    utilities = total_utility_bundles(bundles, U, u_cubicle)
    utility1 = total_utility_bundle(bundle1, U, u_cubicle)
    utility2 = total_utility_bundle(bundle2, U, u_cubicle)
    utility3 = total_utility_bundle(bundle3, U, u_cubicle)

    assert jnp.allclose(utilities, jnp.array([utility1, utility2, utility3]))


def test_find_best_bundle():
    bundle1 = jnp.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    bundle2 = jnp.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    bundle3 = jnp.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1])

    bundles = jnp.array([bundle1, bundle2, bundle3])

    U = jnp.array(
        [
            [10, 5, 0, 0, 0],
            [0, 10, 3, 0, 0],
            [0, 0, 10, 6, 0],
            [0, 0, 0, 20, 8],
            [0, 0, 0, 0, 10],
        ]
    )
    u_cubicle = jnp.array([10, 5])
    bundles = jnp.array([bundle1, bundle2, bundle3])
    best_bundle, _ = find_best_bundle(bundles, U, u_cubicle)
    expected_best_bundle = jnp.array([bundle3])
    assert jnp.allclose(best_bundle, expected_best_bundle)


def test_U_tilde():
    U = jnp.array(
        [
            [10, 5, 0],
            [0, 10, 3],
            [0, 0, 5],
        ]
    )
    u_cubicle = jnp.array([10, 5])
    expected_U_tilde = jnp.array(
        [
            [
                20,
                5,
                0,
                0,
                5,
                0,
            ],
            [
                0,
                20,
                3,
                5,
                0,
                3,
            ],
            [
                0,
                0,
                15,
                0,
                3,
                0,
            ],
            [
                0,
                0,
                0,
                15,
                5,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                15,
                3,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                10,
            ],
        ]
    )
    assert jnp.allclose(create_total_utility_matrix(U, u_cubicle), expected_U_tilde)

    U = jnp.array(
        [
            [10, 5, 0],
            [0, 10, 0],
            [0, 0, 0],
        ]
    )
    u_cubicle = jnp.array([10, 5])

    expected_U_tilde = jnp.array(
        [
            [
                20,
                5,
                0,
                0,
                5,
                0,
            ],
            [
                0,
                20,
                0,
                5,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                15,
                5,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                15,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ]
    )
    assert jnp.allclose(create_total_utility_matrix(U, u_cubicle), expected_U_tilde)


def test_create_utility_matrix_over_slots(verbose=False):
    u_vec = jnp.array([10, 5, 10, 0, 0, 1, 1, 1, 0, 0])
    bonus_full_day = jnp.array(6.0)

    expected = jnp.array(
        [
            [10, 6, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 10, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    output = create_utility_matrix_slots(u_vec, bonus_full_day)
    if verbose:
        print(output)
    assert jnp.allclose(output, expected)


if __name__ == "__main__":
    test_collapse_bundle()
    test_utility_over_half_days()
    test_utility_over_cubicles()
    test_total_utility_cubicle()
    test_total_utility_cubicles()
    test_find_best_bundle()
    test_U_tilde()
    test_create_utility_matrix_over_slots(verbose=True)
