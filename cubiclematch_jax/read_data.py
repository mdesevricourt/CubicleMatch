from pathlib import Path
from venv import create

import jax
import jax.numpy as jnp
import pandas as pd

from cubiclematch_jax.demand.utility import (  # type: ignore
    create_total_utility_matrix,
    create_utility_matrix_slots,
)

data_path = Path(
    r"C:\Users\itism\Documents\GitHub\CubicleMatch\data\Cubicles_spring24_January-8-2024_15.54.csv"
)

df = pd.read_csv(data_path)
print(df.head())
print(df.columns)
print(df.iloc[1])
print(df.iloc[2])
# drop first two rows
# df = df.drop([0, 1])


def row_to_utility(row):
    name = row["Q1"] + " " + row["Q2"]
    # capitalise first letter of each word
    name = name.title()

    # create vector from Q5_1_1 to Q5_10_1
    utility_slots = []
    for i in range(1, 11):
        utility_slots.append(row[f"Q5_{i}_1"])

    full_day_bonus = row["Q6_1_1"]
    utility_cubicles = []

    for i in range(1, 9):
        utility_cubicles.append(row[f"Q7_{i}_1"])

    return (
        name,
        jnp.array(utility_slots, dtype=jnp.float32),
        jnp.array(full_day_bonus, dtype=jnp.float32),
        jnp.array(utility_cubicles, dtype=jnp.float32),
    )


print(row_to_utility(df.iloc[2]))


def row_to_utility_matrix(row, cubicle_included=None):
    if cubicle_included is None:
        cubicle_included = jnp.array([True] * 8)

    name, utility_slots, full_day_bonus, utility_cubicles = row_to_utility(row)

    utility_matrix = create_utility_matrix_slots(utility_slots, full_day_bonus)
    utility_cubicles = utility_cubicles[cubicle_included]
    total_utility_matrix = create_total_utility_matrix(utility_matrix, utility_cubicles)

    return name, total_utility_matrix


cubicles_included = jnp.array([False, True, True, True, True, True, True, True])

print(row_to_utility_matrix(df.iloc[2]))
print(
    row_to_utility_matrix(
        df.iloc[2],
        cubicle_included=cubicles_included,
    )
)
