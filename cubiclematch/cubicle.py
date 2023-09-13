""" Creates the cublicle object."""

import numpy as np


class Cubicle:
    def __init__(self, number, numberofhalfdays=None, prices=None) -> None:
        self.number = str(number)

        if prices is None:
            self.numberofhalfdays = numberofhalfdays
            self.prices = np.ones(numberofhalfdays)
        else:
            self.numberofhalfdays = len(prices)
            self.prices = prices

        self._assigned_agents = []

    @property
    def prices(self):
        return self._prices

    @prices.setter
    def prices(self, prices):
        # check that prices is the right length
        assert len(prices) == self.numberofhalfdays
        self._prices = prices

    def __repr__(self) -> str:
        return f"Cubicle {self.number}"

    def price_bundle(self, bundle):
        """Return the price of a bundle of half-days"""
        return np.dot(self.prices, bundle)

    @property
    def assigned_agents(self):
        return self._assigned_agents

    @assigned_agents.setter
    def assigned_agents(self, assigned_agents):
        self._assigned_agents = assigned_agents
