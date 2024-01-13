"""This file includes information how the algorithm should be run."""

from pathlib import Path

import jax.numpy as jnp

# Cubicles are numbered from 0 to 7. The following line indicates which cubicles are included in the matching. Here cubicles 1 to 7 are included.
cubicles_included = jnp.array([False, True, True, True, True, True, True, True])

data_path = Path(r"/home/mc2534/CubicleMatch/data")
survey_path = data_path / "Cubicles_spring24_January-12-2024_17.07.csv"
main_data = data_path / "main_data.csv"
num_half_days = 10
max_num_half_days = 6
bundles_path = data_path / "vectors_6_10_1.npy"