import unittest 

from market import Market
from cubicle import Cubicle
from agent import Agent

import numpy as np

class TestAgent(unittest.TestCase):
    def setUp(self) -> None:
        U = np.zeros((2, 2))
        np.fill_diagonal(U, [1, 1])
        agent = Agent("Alice", np.array([[0, 1], [0, 0]]), 100)
        return 
class TestMarket(unittest.TestCase):


