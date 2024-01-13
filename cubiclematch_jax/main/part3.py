import logging
import pickle

import jax
import jax.numpy as jnp
import pandas as pd
from config import bundles_path, main_data

from cubiclematch_jax import create_agent
from cubiclematch_jax.main.main import main_algorithm

df = pd.read_pickle(main_data)

# logging includes time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

logging.info("Creating agents")

agents = [create_agent.agent_from_row(row) for _, row in df.iterrows()]

seed = 56789101112

key = jax.random.PRNGKey(seed)
supply = jnp.array([7.0] * 10)
bundles = jnp.load(bundles_path)
algorithm_settings = {}
algorithm_settings["max_iterations"] = 1000
algorithm_settings["max_hours"] = 72
logging.info("Running main algorithm")
res, agg_quant = main_algorithm(key, agents, bundles, supply, verbose=True)

logging.info(f"Results")
logging.info(f'price_vec: {res["price_vec"]}')
logging.info(f'clearing error: {res["clearing_error"]}')
logging.info(f'excess_demand_vec: {res["excess_demand_vec"]}')

# save results using pickly in data folder
logging.info("Saving results")
with open(main_data.parent / "results_{seed}_.pickle", "wb") as f:
    pickle.dump(res, f)
