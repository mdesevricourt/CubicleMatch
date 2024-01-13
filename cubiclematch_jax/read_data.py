from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from cubiclematch_jax.demand.rank_bundles import rank_bundles
from cubiclematch_jax.demand.utility import (  # type: ignore
    create_utility_matrix_slots, vmap_utility_over_half_days)
from cubiclematch_jax.main.config import survey_path

df = pd.read_csv(survey_path)
print(df.head())
print(df.columns)
print(df.iloc[1])
print(df.iloc[2])
# drop first two rows
# df = df.drop([0, 1])
budget_dictionary = {
    "1": 100,
    "2": 101,
    "3": 102,
    "4": 102,
    "5": 102,
}

def row_to_utility(row):
    name = row["Q1"]
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


# print(row_to_utility(df.iloc[2]))


def transform_row(row, cubicle_included=None, bundles = None):
    if cubicle_included is None:
        cubicle_included = jnp.array([True] * 8)

    name, utility_slots, full_day_bonus, utility_cubicles = row_to_utility(row)

    utility_matrix = create_utility_matrix_slots(utility_slots, full_day_bonus)
    utility_cubicles = utility_cubicles[cubicle_included]
    # total_utility_matrix = create_total_utility_matrix(utility_matrix, utility_cubicles)
    year = str(row["Q3"])
    base_budget = budget_dictionary[year]
    # budget is the base budget plus random number between 0 and 1
    budget = base_budget + np.random.rand()
    
    dictionary = {
        "name": name,
        "budget": budget,
        "utility_matrix": utility_matrix,
        "utility_cubicles": utility_cubicles,
    }
    ranked_bundles = None
    if bundles is not None: # rank bundles according to utility matrix
        bundles_utility = vmap_utility_over_half_days(bundles, utility_matrix)
        ranked_bundles = rank_bundles(bundles, bundles_utility)
        dictionary["ranked_bundles"] = ranked_bundles
    
    row["name"] = name
    row["budget"] = budget
    row["utility_matrix"] = utility_matrix
    row["utility_cubicles"] = utility_cubicles
    row["ranked_bundles"] = ranked_bundles

    return row


# cubicles_included = jnp.array([False, True, True, True, True, True, True, True])

# print(transform_row(df.iloc[2]))
