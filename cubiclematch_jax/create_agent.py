

import jax
import jax.numpy as jnp


class Agent:
    def __init__(self, name: str, budget: float, preferences: jax.Array, utilit_matrix: jax.Array) -> None:
        self.name = name
        self.budget = budget
        self.preferences = preferences
        self.utility_matrix = utilit_matrix

def agent_from_row(row):
    name = row["name"]
    budget = row["budget"]
    preferences = jnp.array(row["ranked_bundles"])
    u = jnp.array(row["utility_matrix"])
    return Agent(name, budget, preferences, u)