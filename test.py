import unittest 

from market import Market
from cubicle import Cubicle
from agent import Agent

import numpy as np

class TestAgent(unittest.TestCase):
    def setUp(self) -> None:
        U = np.zeros((2, 2))
        np.fill_diagonal(U, [2, 1])
        self.agent_Alice = Agent("Alice", U, 3)
        return 
    
    def test_utility(self):
        # utility of Alice for [0, 0] should be 0
        self.assertEqual(self.agent_Alice.utility([0, 0]), 0)
        # utility of Alice for [1, 0] should be 2
        self.assertEqual(self.agent_Alice.utility([1, 0]), 2)
        # utility of Alice for [0, 1] should be 1
        self.assertEqual(self.agent_Alice.utility([0, 1]), 1)
        # utility of Alice for [1, 1] should be 3
        self.assertEqual(self.agent_Alice.utility([1, 1]), 3)
        return
    
    def test_find_agent_demand(self):
        # create a list of priced bundles
        priced_bundles = [([0, 0], "1", 1), ([1, 0], "1", 2), ([0, 1], "2", 2), ([1, 1], "1", 3), ([1, 1], "2", 3)]
        bundle, cublicle, price = self.agent_Alice.find_agent_demand(priced_bundles)
        # bundle should be [1, 1]
        self.assertEqual(bundle, [1, 1])
        # cublicle should be "1"
        self.assertEqual(cublicle, "1")
        # other priced_bundles
        priced_bundles = [([0, 0], "1", 1), ([1, 0], "1", 2), ([0, 1], "2", 2), ([1, 1], "1", 5), ([1, 1], "2", 4)]
        bundle, cublicle, price = self.agent_Alice.find_agent_demand(priced_bundles)
        # bundle should be [1,0]
        self.assertEqual(bundle, [1, 0])
        # cublicle should be "1"
        self.assertEqual(cublicle, "1")

        return

class TestMarket(unittest.TestCase):

    # create a setUp method to initialize a Market instance with some sample data
    def setUp(self):
        # create some sample agents
        U_Alice = np.zeros((4, 4))
        np.fill_diagonal(U_Alice, [1, 0, 1, 0])
        U_Bob = np.zeros((4, 4))
        np.fill_diagonal(U_Bob, [0, 1, 0, 1])
        budgets = 10
        agents = [Agent("Alice", U_Alice, 100), Agent("Bob", U_Bob, 100)]
        # create some sample cubicles with different prices
        cubicles = [Cubicle("C1", [10, 20, 30, 40]), Cubicle("C2", [15, 25, 35, 45])]
        # create a Market instance with the agents and cubicles
        self.market = Market(agents, cubicles)

    # create a test method for each method of the Market class
    def test_prices_array(self):
        # test that the prices_array property returns a numpy array of the prices of the cubicles
        expected = np.array([[10, 20, 30, 40], [15, 25, 35, 45]])
        actual = self.market.prices_array
        # use assertArrayEqual to compare numpy arrays
        np.testing.assert_array_equal(actual, expected)

    def test_prices_array(self):
        # test that the update_prices_array method sets the prices of the cubicles correctly
        # create a new array of prices to update
        new_prices = np.array([[11, 21, 31, 41], [16, 26, 36, 46]])
        # call the update_prices_array method with the new prices
        self.market.prices_array = new_prices
        # check that the prices_array property reflects the changes
        expected = new_prices
        actual = self.market.prices_array
        # use assertArrayEqual to compare numpy arrays
        np.testing.assert_array_equal(actual, expected)

    def test_prices_vec(self):
        # test that the prices_vec property returns a numpy array of the prices of the cubicles flattened
        expected = np.array([10, 20, 30, 40, 15, 25, 35, 45])
        actual = self.market.prices_vec
        # use assertArrayEqual to compare numpy arrays
        np.testing.assert_array_equal(actual, expected)

    def test_prices_vec(self):
        # test that the update_prices_vec method sets the prices of the cubicles correctly
        # create a new array of prices to update (flattened)
        new_prices = np.array([11, 21, 31, 41, 16, 26, 36, 46])
        # call the update_prices_vec method with the new prices
        self.market.prices_vec = new_prices
        # check that the prices_vec property reflects the changes
        expected = new_prices
        actual = self.market.prices_vec
        # use assertArrayEqual to compare numpy arrays
        np.testing.assert_array_equal(actual, expected)

    def test_price_bundle(self):
        # test that the price_bundle method returns the lowest price of a bundle of half-days across all cubicles,
        # as well as the name of the cubicle that has the lowest price
        # create a sample bundle of half-days (a numpy array of zeros and ones)
        bundle = np.array([1, 0, 1, 0])
        # call the price_bundle method with the bundle
        price, cubicle = self.market.price_bundle(bundle)
        # check that the price is correct (the sum of the first and third elements of the lowest-priced cubicle)
        expected_price = 10 + 30 # C1 has the lowest price for this bundle
        actual_price = price
        # use assertEqual to compare scalars
        self.assertEqual(actual_price, expected_price)
        # check that the cubicle is correct
        expected_cubicle = "C1"
        actual_cubicle = cubicle
        # use assertEqual to compare strings
        self.assertEqual(actual_cubicle, expected_cubicle)

    def test_aggregate_demand(self):
        # test that the aggregate_demand method returns a dictionary of the aggregate demand for each cubicle
        # create a sample bundle of half-days (a numpy array of zeros and ones)
        expected_aggregate_demand = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        actual_aggregate_demand = self.market.aggregate_demand()
        print(self.market.cublicles_names)
        self.assertEqual(actual_aggregate_demand, expected_aggregate_demand.tolist())
    

if __name__ == "__main__":
    unittest.main()



# class TestMarket(unittest.TestCase):


