from math import e

import numpy as np

from cubiclematch_jax.supply import generate_vectors


def test_generate_vectors():
    vectors = generate_vectors(2, 10, 3)

    assert np.all(np.sum(vectors, axis=1) <= 2)

    # at least one vector is all zeros
    assert np.any(np.sum(vectors, axis=1) == 0)

    # least one vector is [1, 1] followed by 28 zeros
    expected_vector = np.zeros(30)
    expected_vector[0] = 1
    expected_vector[1] = 1

    assert np.any(np.all(vectors == expected_vector, axis=1))

    # no vector is [1, 1, 1] followed by 27 zeros

    expected_vector = np.zeros(30)
    expected_vector[0] = 1
    expected_vector[1] = 1
    expected_vector[2] = 1

    assert not np.any(np.all(vectors == expected_vector, axis=1))

    # no vector has two ones exactely 10 positions apart

    expected_vector = np.zeros(30)
    expected_vector[0] = 1
    expected_vector[10] = 1

    assert not np.any(np.all(vectors == expected_vector, axis=1))

    assert np.sum(np.sum(vectors, axis=1) == 0) == 1


if __name__ == "__main__":
    test_generate_vectors()
