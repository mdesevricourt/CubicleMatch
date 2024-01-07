import jax
import jax.numpy as jnp

from cubiclematch_jax.main import main


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


def test_main():
    key = jax.random.PRNGKey(0)

    agents = [agent1]

    price_vec, excess_demand, excess_budgets, num_iterations = main(
        key,
        agents,
        bundles,
        supply,
    )


if __name__ == "__main__":
    test_main()
