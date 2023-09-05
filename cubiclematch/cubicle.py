""" Creates the cublicle object."""

import numpy as np


class Cubicle:
    def __init__(self, number, numberofhalfdays) -> None:
        self.number = str(number)
        self._prices = np.zeros(numberofhalfdays)
        self.numberofhalfdays = numberofhalfdays
        

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
    