""" This module tests the performance of the package."""

import numpy as np
import matplotlib.pyplot as plt
import time

from market import Market
from cubicle import Cubicle
from agent import Agent


def test_performance_priced_bundles():
    """Tests the performance of the priced_bundles method of the Market class."""

    # create some sample agents
    U_Alice = np.zeros((4, 4))
    np.fill_diagonal(U_Alice, [1, 0, 1, 0])
    U_Bob = np.zeros((4, 4))
    np.fill_diagonal(U_Bob, [0, 1, 0, 1])
    agents = [Agent("Alice", U_Alice, 100), Agent("Bob", U_Bob, 101)]

    # create 5 cubicles with different prices
    cubicles = [
        Cubicle("C1", [10, 20, 30, 40]),
        Cubicle("C2", [15, 25, 35, 45]),
        Cubicle("C3", [20, 30, 40, 50]),
        Cubicle("C4", [25, 35, 45, 55]),
        Cubicle("C5", [30, 40, 50, 60]),
    ]

    # create a Market instance with the agents and cubicles
    market = Market(agents, cubicles, maxing_price=False)

    # measure the time it takes to compute the priced bundles
    start = time.time()
    priced_bundles = market.priced_bundles
    end = time.time()
    print(
        f"Time to compute priced bundles: {end - start} seconds when maxing_price = False"
    )

    # measure the time it takes to compute the priced bundles when maxing_price = True
    market = Market(agents, cubicles, maxing_price=True)
    start = time.time()
    priced_bundles = market.priced_bundles
    end = time.time()

    print(
        f"Time to compute priced bundles: {end - start} seconds when maxing_price = True"
    )

    return


if __name__ == "__main__":
    test_performance_priced_bundles()
