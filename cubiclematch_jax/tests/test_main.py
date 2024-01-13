import jax
import jax.numpy as jnp

from cubiclematch import agent
from cubiclematch_jax.main.main import main_algorithm


class Agent:
    name: str
    budget: float
    preferences: jax.Array

    def __init__(self, name: str, budget: float, preferences: jax.Array):
        self.name = name
        self.budget = budget
        self.preferences = preferences


bundle0 = jnp.array([0.0, 0.0, 0.0])
bundle1 = jnp.array([1.0, 0.0, 0.0])
bundle2 = jnp.array([0.0, 1.0, 0.0])
bundle3 = jnp.array([0.0, 0.0, 1.0])
bundle4 = jnp.array([1.0, 1.0, 0.0])
bundle5 = jnp.array([1.0, 0.0, 1.0])
bundle6 = jnp.array([0.0, 1.0, 1.0])
bundle7 = jnp.array([1.0, 1.0, 1.0])
bundles = jnp.array(
    [bundle0, bundle1, bundle2, bundle3, bundle4, bundle5, bundle6, bundle7]
)
supply = jnp.array([1.0, 1.0, 1.0])
bundles_reversed = bundles[::-1]

agent1 = Agent("agent1", 100.0, preferences=bundles_reversed)
agent2 = Agent("agent2", 101.0, preferences=bundles_reversed)
agent3 = Agent("agent3", 102.0, preferences=bundles_reversed)


def test_main():
    key = jax.random.PRNGKey(10)

    agents = [agent1, agent2, agent3]

    res, agg_quant = main_algorithm(key, agents, bundles, supply, verbose=True)
    print(f"price_vec: {res[0]}")
    print(f"excess_demand_vec: {res[1]}")
    print(f"number_excess_demand: {res[2]}")
    print(f"key: {res[3]}")
    print(f"agg_quant: {agg_quant}")


if __name__ == "__main__":
    test_main()
