""" Creates the cublicle object."""

import numpy as np


class Cubicle:
    def __init__(self, number, prices) -> None:
        self.number = number
        self._prices = prices
        self.numberofhalfdays = len(prices)

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
    