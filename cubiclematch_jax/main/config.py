"""This file includes information how the algorithm should be run."""

from pathlib import Path

import jax.numpy as jnp

# Cubicles are numbered from 0 to 7. The following line indicates which cubicles are included in the matching. Here cubicles 1 to 7 are included.
cubicles_included = jnp.array([False, True, True, True, True, True, True, True])

data_path = Path(r"C:\Users\itism\Documents\GitHub\CubicleMatch\data")
num_half_days = 10
max_num_half_days = 6
