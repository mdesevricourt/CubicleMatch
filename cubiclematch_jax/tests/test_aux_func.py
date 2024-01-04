import jax
import jax.numpy as jnp

from cubiclematch_jax.aux_func import (
    filter_out_tabu_neighbors,
    find_neighbor_with_smallest_error,
    generate_random_price_vector,
)
from cubiclematch_jax.demand import compute_aggregate_demand


def test_generate_random_price_vector():
    """Test the generate_random_price_vector function"""
    key = jax.random.PRNGKey(0)
    length = 10
    total_budget = 100
    price_vector, key = generate_random_price_vector(length, total_budget, key)

    assert price_vector.shape == (length,)
    assert jnp.all(price_vector >= 0)
    assert jnp.all(price_vector <= total_budget)

    price_vector_2, key = generate_random_price_vector(length, total_budget, key)

    assert not jnp.allclose(price_vector, price_vector_2)


def test_filter_out_tabu_neighbors():
    """Test the filter_out_tabu_neighbors function"""
    neighbors = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [8, 9, 10]])
    tabu_list = jnp.array([[1, 2, 3], [7, 8, 9]])
    agg_dict = {"excess_demand_vec": neighbors}

    non_tabu_neighbors = filter_out_tabu_neighbors(neighbors, agg_dict, tabu_list)

    assert jnp.allclose(non_tabu_neighbors, jnp.array([[4, 5, 6], [8, 9, 10]]))


def test_find_neighbor_with_smallest_error():
    """Test the find_neighbor_with_smallest_error function"""
    neighbors = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [8, 9, 10]])
    clearing_errors = jnp.array([1, 2, 3, 4])
    neighbor, error = find_neighbor_with_smallest_error(neighbors, clearing_errors)

    assert jnp.allclose(neighbor, jnp.array([1, 2, 3]))
    assert jnp.allclose(error, jnp.array(1))


if __name__ == "__main__":
    test_generate_random_price_vector()
    test_filter_out_tabu_neighbors()
    test_find_neighbor_with_smallest_error()
