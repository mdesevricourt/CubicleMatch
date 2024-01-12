"""The first step of the main algorithm is to create all the bundles corresponding to the cubicles included."""

import logging

import jax.numpy as jnp

from cubiclematch_jax.main.config import (
    cubicles_included,
    data_path,
    max_num_half_days,
    num_half_days,
)
from cubiclematch_jax.supply.supply import generate_and_save_bundles

num_cubicles = int(jnp.sum(cubicles_included))

bundles = generate_and_save_bundles(
    max_num_half_days=max_num_half_days,
    total_num_cubicles=num_cubicles,
    total_num_half_days=num_half_days,
    data_path=data_path,
)
