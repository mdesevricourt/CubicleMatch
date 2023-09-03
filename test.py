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


if __name__ == "__main__":
    unittest.main()



# class TestMarket(unittest.TestCase):


