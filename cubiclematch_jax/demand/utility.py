import jax
import jax.numpy as jnp
import numpy as np

Bundle = (
    jax.Array
)  # a bundle is a vector of 0s and 1s of length number_of_half_days * number_of_cubicles


def collapse_bundle(bundle: Bundle, num_half_days: int, num_cubicles: int) -> jax.Array:
    """Return the collapsed bundle, that is the bundle, ignoring the cubicle dimension, that is the sum of the cubicle bundles."""

    # reshape the bundle to be a 2D array with shape (number_of_half_days, number_of_cubicles)
    bundle = jnp.reshape(bundle, (num_cubicles, num_half_days))

    # sum the bundle across the cubicle dimension
    collapsed_bundle = jnp.sum(bundle, axis=0)
    # take the minimum of the collapsed bundle and 1
    collapsed_bundle = jnp.minimum(collapsed_bundle, 1)

    return collapsed_bundle


def utility_over_half_days(collapsed_bundle: jax.Array, U: jax.Array) -> jax.Array:
    """Return the utility of a person for a given assignment across all half-days

    Args:
        collapsed_bundle (jax.Array): A collapsed bundle, that is a vector of 0s and 1s of length number_of_half_days.
        U (jax.Array): A utility matrix, that is a 2D square triangular numpy array of shape (number_of_half_days, number_of_half_days).

    Returns:
        jax.Array: A utility vector of shape (1,).
    """

    # reshape the collapsed bundle to be a 2D array with shape (number_of_half_days, 1)
    collapsed_bundle = jnp.reshape(collapsed_bundle, (-1, 1))

    # compute the utility
    utility = jnp.matmul(jnp.matmul(collapsed_bundle.T, U), collapsed_bundle)

    return utility


def utility_over_cubicles(bundle: jax.Array, u_cubicle: jax.Array) -> jax.Array:
    """Return the utility of a person from the cubicle.

    Args:
        bundle (jax.Array): A bundle, that is a vector of 0s and 1s of length number_of_half_days * number_of_cubicles.
        u_cubicle (jax.Array): A utility vector of shape (number_of_cubicles,).

    Returns:
        jax.Array: A utility vector of shape (1,).
    """
    number_of_cubicles = u_cubicle.shape[0]
    # reshape the bundle to be a 2D array with shape (number_of_cubicles, number_of_half_days)
    bundle = jnp.reshape(bundle, (number_of_cubicles, -1))

    # sum the bundle across the half-day dimension
    bundle = jnp.sum(bundle, axis=1)

    # compute the utility
    utility = jnp.dot(bundle, u_cubicle)
    # reduce the utility to a scalar

    return jnp.squeeze(utility)


def total_utility_bundle(
    bundle: jax.Array, U: jax.Array, u_cubicle: jax.Array
) -> jax.Array:
    """Return the total utility of a person from the cubicle.

    Args:
        bundle (jax.Array): A bundle, that is a vector of 0s and 1s of length number_of_half_days * number_of_cubicles.
        U (jax.Array): A utility matrix, that is a 2D square triangular numpy array of shape (number_of_half_days, number_of_half_days).
        u_cubicle (jax.Array): A utility vector of shape (number_of_cubicles,).

    Returns:
        jax.Array: A utility vector of shape (1,).
    """
    collapsed_bundle = collapse_bundle(bundle, U.shape[0], u_cubicle.shape[0])
    utility = jnp.squeeze(
        utility_over_half_days(collapsed_bundle, U)
        + utility_over_cubicles(bundle, u_cubicle)
    )
    return utility


total_utility_bundles = jax.vmap(total_utility_bundle, in_axes=(0, None, None))


def find_best_bundle(bundles: jax.Array, U: jax.Array, u_cubicle: jax.Array):
    """Return the best bundle from a list of bundles.

    Args:
        bundles (jax.Array): A list of bundles.
        U (jax.Array): A utility matrix, that is a 2D square triangular numpy array of shape (number_of_half_days, number_of_half_days).
        u_cubicle (jax.Array): A utility vector of shape (number_of_cubicles,).

    Returns:
        jax.Array: The best bundle.
        best_bundle_index: The index of the best bundle.
    """

    # compute the total utility of each bundle
    utilities = total_utility_bundles(bundles, U, u_cubicle)

    # find the index of the best bundle
    best_bundle_index = jnp.argmax(utilities)

    # return the best bundle
    return bundles[best_bundle_index], best_bundle_index


def create_total_utility_matrix(
    U_jax: jax.Array, u_cubicle_jax: jax.Array
) -> jax.Array:
    """Return the utility matrix of the cubicle match problem.

    Args:
        U (jax.Array): A utility matrix, that is a 2D square triangular numpy array of shape (number_of_half_days, number_of_half_days).
        u_cubicle (jax.Array): A floating point number representing the utility associated with a cubicle.

    Returns:
        jax.Array: The utility matrix of the cubicle match problem.
    """
    # repeat U along the diagonal the same number of times as the number of cubicles
    U_tilde = np.zeros(
        (
            U_jax.shape[0] * u_cubicle_jax.shape[0],
            U_jax.shape[1] * u_cubicle_jax.shape[0],
        )
    )
    U = np.array(U_jax)
    u_cubicle = np.array(u_cubicle_jax)

    for i in range(u_cubicle.shape[0]):
        matrix = np.array(U)
        diagonal = np.where(matrix.diagonal() == 0, 0, matrix.diagonal() + u_cubicle[i])
        np.fill_diagonal(matrix, diagonal)

        U_tilde[
            i * U.shape[0] : (i + 1) * U.shape[0], i * U.shape[0] : (i + 1) * U.shape[0]
        ] = matrix

    n_half_days = U.shape[0]
    shape_U_tilde = U_tilde.shape

    for i in range(shape_U_tilde[0]):
        for j in range(i + 1, shape_U_tilde[1]):
            i_orig = i % n_half_days
            j_orig = j % n_half_days
            if j_orig == i_orig:
                continue

            U_tilde[i, j] = U[i_orig, j_orig] + U[j_orig, i_orig]

    U_tilde2 = jnp.array(U_tilde)
    return U_tilde2


def create_utility_matrix_slots(
    utility_half_days: jax.Array, bonus_full_day: jax.Array
) -> jax.Array:
    """Return the utility matrix over half_days for the cubicle match problem. It takes into account utility over each half-day and the
    bonus for a full day.

    Args
    ----
    utility_half_days (jax.Array): A utility vector of utility over each half-day.

    bonus_full_day (jax.Array): A utility bonus (scalar) for a full day."""

    U = np.zeros((utility_half_days.shape[0], utility_half_days.shape[0]))
    # the diagonal of the utility matrix is the utility over half-days
    np.fill_diagonal(U, utility_half_days)
    # add the bonus for a full day
    for i in range(U.shape[0]):
        if i % 2 == 0:
            both_positive = utility_half_days[i] > 0 and utility_half_days[i + 1] > 0
            if both_positive:
                U[i, i + 1] = bonus_full_day

    return jnp.array(U)


def compute_utility_tilde(bundle: jax.Array, U_tilde: jax.Array):
    """Return the utility of a person from a bundle in the cubicle match problem.

    Args:
        bundle (jax.Array): A bundle, that is a vector of 0s and 1s of length number_of_half_days * number_of_cubicles.
        U_tilde (jax.Array): The utility matrix of the cubicle match problem.

    Returns:
        jax.Array: A utility vector of shape (1,).
    """

    # compute the utility
    utility = jnp.matmul(jnp.matmul(bundle.T, U_tilde), bundle)

    return jnp.squeeze(utility)


compute_utility_tilde_vec = jax.vmap(compute_utility_tilde, in_axes=(0, None))


def find_best_bundle_tilde(bundles: jax.Array, U_tilde: jax.Array):
    """Return the best bundle from a list of bundles in the cubicle match problem.

    Args:
        bundles (jax.Array): A list of bundles.
        U_tilde (jax.Array): The utility matrix of the cubicle match problem.

    Returns:
        best_bundle(jax.Array): The best bundle.
        best_bundle_index: The index of the best bundle.
    """

    # compute the total utility of each bundle
    utilities = compute_utility_tilde_vec(bundles, U_tilde)

    # find the index of the best bundle
    best_bundle_index = jnp.argmax(utilities)

    # return the best bundle
    return bundles[best_bundle_index], best_bundle_index
