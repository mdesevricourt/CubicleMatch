"""Create a market class to store the information of the market to the matching algorithm to solve their demand"""

import numpy as np
import random
import itertools

import cubicle, agent


class Market:
    def __init__(self, agents, cublicles) -> None:
        self.agents = agents
        self.cublicles = cublicles
        self.numberofagents = len(agents)
        self.numberofcublicles = len(cublicles)
        self.numberofhalfdays = cublicles[0].numberofhalfdays
        self.bundles = [np.array(t) for t in itertools.product([0, 1], repeat=self.numberofhalfdays)]
        

    @property
    def prices_array(self):
        """Returns the prices of the cublicles in a numpy array"""
        return np.array([cubicle.prices for cubicle in self.cublicles])
    
    @prices_array.setter
    def update_prices_array(self, prices):
        """Set the prices of the cublicles"""
        # check that prices is the right length
        assert len(prices) == self.numberofcublicles
        for cubicle, price in zip(self.cublicles, prices):
            cubicle.prices = price

    @property
    def prices_vec(self):
        """Returns the prices of the cublicles in a numpy array"""
        return np.array([cubicle.prices for cubicle in self.cublicles]).flatten()
    
    @prices_vec.setter
    def update_prices_vec(self, prices):
        """Set the prices of the cublicles"""
        
        # divide the prices into the cublicles
        prices = np.array(prices).reshape(self.numberofcublicles, self.numberofhalfdays)
        # use the update prices_array method
        self.update_prices_array(prices)

    def price_bundle(self, bundle):
        """Return the lowest price of a bundle of half-days across all cublicles, as wee as the name of the cublicle that has the lowest price"""
        # create a dictionary of the prices of all the cublicles
        prices = {}
        for cubicle in self.cublicles:
            prices[cubicle.number] = cubicle.price_bundle(bundle)
        
        # get the number of cublicle that has the lowest price
        
        cubicle = min(prices, key=prices.get)
        # get the price of the cublicle that has the lowest price
        price = prices[cubicle]

        return price, cubicle

    @property
    def priced_bundles(self):
        """Return the prices of all the bundles, along with the cublicle that has the lowest price for that bundle"""
        
        bundles = self.bundles
        # for each bundle, finds the cubicle that has the lowest price for that bundle, get its name and the price
        # create a list of triplet (bundle, cubicle, price)
        princed_bundles = []

        for bundle in bundles:
            price, cubicle = self.price_bundle(bundle)
            princed_bundles.append((bundle, cubicle, price))

        # sort the list of triplet by price from lowest to highest
        princed_bundles = sorted(princed_bundles, key=lambda x: x[2])

        return princed_bundles

    

        
        
